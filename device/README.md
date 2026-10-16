# Device runner

Both platforms, same protocol:
- Fixed prompt set (20 prompts, mixed languages, 128 generated tokens each) for TTFT and decode speed.
- 10-minute sustained generation run; log decode tok/s every 10 seconds to see the throttle curve.
- Peak RSS; battery counters before and after (Android charge_counter; iOS battery level); temperature.
- Airplane mode, fixed brightness, battery 40 to 80 percent, 30-minute cool-down, N=3.
- Output: JSONL rows in the harness schema with task = "device_cost" and metric in
  {ttft_ms, decode_tps, sustained_tps_p50, peak_rss_mb, energy_j_per_1k_tok, temp_max_c}.

Parity gate: 200 prompts, greedy, token-for-token agreement between the desktop runtime and
the device runtime for the same artifact. Report the rate alongside every quality number.
