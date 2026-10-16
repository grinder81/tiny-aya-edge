# Android phone (Snapdragon 8 Elite) setup

Buy used: Galaxy S25 or OnePlus 13 (Snapdragon 8 Elite, Hexagon v79). The llama.cpp Hexagon backend
targets 8 Gen 3, 8 Elite and 8 Elite Gen 5; the 8 Elite is the sweet spot for price and support.

## Host tools
- Android Studio or command-line tools; NDK r27+; `adb` on PATH; enable USB debugging on the phone.
- Optional: Termux on the phone for quick CPU-only builds.

## llama.cpp builds (three backends)
Follow the upstream docs; flags move. Start from `docs/backend/snapdragon/` and `docs/backend/hexagon/`
in the llama.cpp repo (there is a `CMakeUserPresets.json` with an `arm64-android-snapdragon-release` preset).
```bash
# CPU + OpenCL (Adreno). Hexagon needs the Qualcomm Hexagon SDK installed; see the docs for the preset.
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 \
  -DGGML_OPENCL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-android -j
adb push build-android/bin/llama-bench build-android/bin/llama-cli /data/local/tmp/
adb push models/tiny-aya-global-Q4_K_M.gguf /data/local/tmp/
adb shell "cd /data/local/tmp && chmod +x llama-bench && ./llama-bench -m tiny-aya-global-Q4_K_M.gguf -p 512 -n 128"
```
Record for each backend: pp512 t/s, tg128 t/s, and which ops fell back to CPU on Hexagon
(`GGML_HEX_VERBOSE=1`). That fallback list goes upstream as an issue, with numbers.

## Battery and thermal readings (the device runner uses these)
```bash
adb shell dumpsys battery                                   # level, voltage, temperature
adb shell cat /sys/class/power_supply/battery/charge_counter  # micro-amp-hours; delta over a run = energy proxy
adb shell cat /sys/class/thermal/thermal_zone*/temp           # find the CPU/GPU/NPU zones once, then poll them
```
Protocol: airplane mode, fixed brightness, battery between 40 and 80 percent, 30-minute cool-down
between runs, three runs per configuration.

## ExecuTorch on Android (once Cohere2 export lands)
`examples/demo-apps/android/LlamaDemo` in the executorch repo, XNNPACK backend. Later: QNN backend.
