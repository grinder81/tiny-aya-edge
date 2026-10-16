ARCH ?= sm_120        # RTX 5090 (Blackwell consumer). Use ARCH=native if your nvcc supports it.
NVCC ?= nvcc
all: q4_matvec
q4_matvec: q4_matvec.cu
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -o $@ $<
run: q4_matvec
	./q4_matvec 4096 4096
	./q4_matvec 12288 3072
	./q4_matvec 3072 12288
clean:
	rm -f q4_matvec
