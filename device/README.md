# Device runner

Phones: iPhone 15 (A16, 6 GB RAM, LPDDR5 about 51 GB/s) and a Snapdragon 8 Elite device. On the
iPhone the 6 GB ceiling means BF16 cannot load, Q8_0 (3.6 GB plus cache) is tight, and 4-bit builds
are the realistic class; the ExecuTorch and llama.cpp apps need the Increased Memory Limit entitlement.
That constraint is part of the result, not an inconvenience.

Both platforms, same protocol:
- Fixed prompt set (20 prompts, mixed languages, 128 generated tokens each) for TTFT and decode speed.
- 10-minute sustained generation run; log decode tok/s every 10 seconds to see the throttle curve.
- Peak RSS; battery counters before and after (Android charge_counter; iOS battery level); temperature.
- Airplane mode, fixed brightness, battery 40 to 80 percent, 30-minute cool-down, N=3.
- Output: JSONL rows in the harness schema with task = "device_cost" and metric in
  {ttft_ms, decode_tps, sustained_tps_p50, peak_rss_mb, energy_j_per_1k_tok, temp_max_c}.

Parity gate: 200 prompts, greedy, token-for-token agreement between the desktop runtime and
the device runtime for the same artifact. Report the rate alongside every quality number.

## Implementation and measurement readiness

This directory currently contains the protocol only: there is no runnable iPhone
harness and no verified iPhone measurement. P13 requires physical device access
and a reproducible app build. Before collecting results, record the iPhone model,
iOS version, Xcode/SDK and app build, llama.cpp revision and local tokenizer patches,
model SHA256, context/KV settings, prompt/tokenizer/chat-template identities,
sampling settings, warm-up schedule and the memory entitlement actually signed
into the app. An arbitrary existing app is exploratory unless its runtime and
measurement behavior can be verified. Do not extrapolate RTX results to the phone.

Keep the protocol above unchanged, but distinguish what the instrument actually
observes:

- Record battery level before/after, elapsed time and charging state. Apple's
  [UIDevice.batteryLevel](https://developer.apple.com/documentation/uikit/uidevice/batterylevel)
  returns charge level, with an unavailable value when monitoring is disabled.
  A level difference is a battery-percentage-point observation, not a joule
  measurement. Do not turn it into `energy_j_per_1k_tok` using an assumed capacity.
- If using Instruments, preserve the exported trace and its units. Apple's
  [Power Profiler documentation](https://developer.apple.com/documentation/xcode/measuring-your-app-s-power-use-with-power-profiler)
  describes system power relative to battery energy per hour; its power-impact
  values are device-dependent. Do not relabel those values as app-only joules or
  compare them across device models as an absolute energy scale.
- Record measured temperature in Celsius only with a documented instrument.
  A categorical thermal state is useful context but cannot become `temp_max_c`.
- Preserve the memory measurement's API/tool, units and scope. Resident memory,
  physical footprint and device-wide memory are distinct; do not silently label
  one as another or infer peak runtime memory from model file bytes.
- Record actual generated tokens and stop reason. If generation stops before
  128 tokens, retain that outcome; do not pretend 128 tokens were generated or
  silently change stopping to fill the budget. Separate load, prompt processing,
  first-token and decode timing boundaries, with prompt/output lengths retained.

Unavailable measurements are recorded as unavailable with a reason in raw run
metadata and omitted from numerical metric rows; they are never zero. In
particular, `energy_j_per_1k_tok` needs an independently documented energy
measurement method. Battery/thermal observations can remain supporting evidence
without those numerical claims. This clarification introduces no device result,
changes no canonical benchmark, and does not close P13.
