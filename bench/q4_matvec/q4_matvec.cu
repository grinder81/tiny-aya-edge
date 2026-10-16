// q4_matvec.cu : Q4_0 matvec microbenchmark.
// y[M] = W[M,K] * x[K] with W stored as Q4_0-style blocks (32 weights, one fp16 scale, nibbles = q-8).
// Goal: measure achieved GB/s for a memory-bound quantized matvec against the device roofline,
// then compare the design with ggml-cuda (mmvq.cu) and ggml-metal (kernel_mul_mv_q4_0_f32).
// Layout here is SoA (scales[] and qs[] separate) for aligned 16-byte loads; ggml uses 18-byte AoS blocks.
// Build: make ARCH=sm_120   Run: ./q4_matvec [M] [K] [peak_GBps]
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

constexpr int QK = 32;   // weights per block

// One warp per row. Lane L handles blocks L, L+32, L+64, ... of that row.
__global__ void q4_matvec_kernel(const uint4* __restrict__ qs,     // [M * nb] 16 bytes per block
                                 const half*  __restrict__ scales, // [M * nb]
                                 const float* __restrict__ x,      // [K]
                                 float* __restrict__ y,            // [M]
                                 int M, int nb) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= M) return;
    const uint4* rq = qs + (size_t)warp * nb;
    const half*  rs = scales + (size_t)warp * nb;
    float acc = 0.f;
    for (int b = lane; b < nb; b += 32) {
        const uint4 q = rq[b];
        const float d = __half2float(rs[b]);
        const float* xb = x + b * QK;
        const uint32_t w[4] = {q.x, q.y, q.z, q.w};
        float s = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t byte = (w[i] >> (8 * j)) & 0xFF;
                const int lo = (int)(byte & 0x0F) - 8;   // weight index i*4+j
                const int hi = (int)(byte >> 4)   - 8;   // weight index i*4+j+16
                s += lo * xb[i * 4 + j] + hi * xb[i * 4 + j + 16];
            }
        }
        acc += d * s;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (lane == 0) y[warp] = acc;
}

// Host-side Q4_0 quantization, same rule as ggml: d = amax/-8 where amax is the signed value with largest |w|.
static void quantize_q4_0(const std::vector<float>& W, int M, int K,
                          std::vector<uint8_t>& qs, std::vector<half>& scales, std::vector<float>& Wdq) {
    const int nb = K / QK;
    qs.assign((size_t)M * nb * 16, 0); scales.assign((size_t)M * nb, __float2half(0.f)); Wdq.assign(W.size(), 0.f);
    for (int m = 0; m < M; ++m) for (int b = 0; b < nb; ++b) {
        const float* w = &W[(size_t)m * K + b * QK];
        float amax = 0.f, maxv = 0.f;
        for (int j = 0; j < QK; ++j) if (fabsf(w[j]) > amax) { amax = fabsf(w[j]); maxv = w[j]; }
        const float d = maxv / -8.f; const float id = d ? 1.f / d : 0.f;
        const half dh = __float2half(d); const float dr = __half2float(dh);
        scales[(size_t)m * nb + b] = dh;
        uint8_t* q = &qs[((size_t)m * nb + b) * 16];
        for (int j = 0; j < QK / 2; ++j) {
            int q0 = (int)(w[j] * id + 8.5f);        q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
            int q1 = (int)(w[j + 16] * id + 8.5f);   q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
            q[j] = (uint8_t)(q0 | (q1 << 4));
            Wdq[(size_t)m * K + b * QK + j]      = (q0 - 8) * dr;
            Wdq[(size_t)m * K + b * QK + j + 16] = (q1 - 8) * dr;
        }
    }
}

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 4096;
    const int K = argc > 2 ? atoi(argv[2]) : 4096;
    const double peak = argc > 3 ? atof(argv[3]) : 1792.0;   // RTX 5090 GDDR7 spec, GB/s
    if (K % QK) { fprintf(stderr, "K must be a multiple of 32\n"); return 1; }
    const int nb = K / QK;

    std::mt19937 rng(42); std::normal_distribution<float> nd(0.f, 0.02f), nx(0.f, 1.f);
    std::vector<float> W((size_t)M * K), x(K), Wdq;
    for (auto& v : W) v = nd(rng);
    for (auto& v : x) v = nx(rng);
    std::vector<uint8_t> qs; std::vector<half> scales;
    quantize_q4_0(W, M, K, qs, scales, Wdq);

    // host references: dequantized (what the kernel should match) and original fp32 (quantization error)
    std::vector<double> y_ref(M, 0.0), y_fp32(M, 0.0);
    for (int m = 0; m < M; ++m) { double a = 0, b = 0; for (int k = 0; k < K; ++k) { a += (double)Wdq[(size_t)m*K+k]*x[k]; b += (double)W[(size_t)m*K+k]*x[k]; } y_ref[m] = a; y_fp32[m] = b; }

    uint4* d_qs; half* d_s; float *d_x, *d_y;
    CK(cudaMalloc(&d_qs, qs.size())); CK(cudaMalloc(&d_s, scales.size() * sizeof(half)));
    CK(cudaMalloc(&d_x, K * sizeof(float))); CK(cudaMalloc(&d_y, M * sizeof(float)));
    CK(cudaMemcpy(d_qs, qs.data(), qs.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_s, scales.data(), scales.size() * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x, x.data(), K * sizeof(float), cudaMemcpyHostToDevice));

    const int threads = 256, warps_per_block = threads / 32;
    const int blocks = (M + warps_per_block - 1) / warps_per_block;
    for (int i = 0; i < 10; ++i) q4_matvec_kernel<<<blocks, threads>>>(d_qs, d_s, d_x, d_y, M, nb);
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const int iters = 200;
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) q4_matvec_kernel<<<blocks, threads>>>(d_qs, d_s, d_x, d_y, M, nb);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); ms /= iters;

    std::vector<float> y(M); CK(cudaMemcpy(y.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost));
    double max_err = 0, max_qerr = 0, norm = 0;
    for (int m = 0; m < M; ++m) { max_err = fmax(max_err, fabs(y[m] - y_ref[m])); max_qerr = fmax(max_qerr, fabs(y[m] - y_fp32[m])); norm = fmax(norm, fabs(y_ref[m])); }

    const double bytes = (double)qs.size() + scales.size() * 2.0 + K * 4.0 + M * 4.0;
    const double gbps = bytes / (ms * 1e-3) / 1e9;
    printf("M=%d K=%d  blocks/row=%d  bytes=%.1f MB\n", M, K, nb, bytes / 1e6);
    printf("time %.4f ms  ->  %.1f GB/s  = %.1f%% of %.0f GB/s peak\n", ms, gbps, 100.0 * gbps / peak, peak);
    printf("max |y - y_dequant_ref| = %.3e (rel %.2e)   max |y - y_fp32| = %.3e (quantization error)\n", max_err, max_err / norm, max_qerr);
    printf("next: compare with ggml-cuda mmvq.cu (dp4a on int8-quantized x, AoS 18-byte blocks) and llama-bench tg numbers\n");
    return 0;
}
