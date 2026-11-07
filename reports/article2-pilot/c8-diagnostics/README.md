# C8 degrade-one diagnostic exports

The sixteen native diagnostics of the [frozen C8 execution plan](https://github.com/grinder81/polyglot-quant-eval/blob/main/results/article2-pilot/c8-plan.json) (SHA-256 `e7ccdb50…`) exist as actual GGUF files: the BF16 control with exactly one tensor degraded to Q2_K or Q3_K, no importance matrix, every other tensor unchanged. Each `<stem>/manifest.json` here is the manifest the export tool wrote, committed byte-for-byte before any comparative probe runs, as the T3 build manifests under `../builds` were; the discovery driver refuses to start unless every committed manifest equals the one in the workspace. The manifests therefore carry the workspace paths of the frozen plan. Each records the quantizer revision and binary hash, the exact command, the output SHA-256 and size, the degraded tensor's payload size, parity hashes of three native rows, and the payload hash of every unchanged tensor.

Nine exports completed on 1 November 2026; the tenth was interrupted by the host-storage exhaustion described in [storage-incident](../storage-incident/README.md), and the remaining seven ran on 7 November after the recovery gate was satisfied, under the backing-volume guard. All sixteen outputs were hashed inside the distro against their manifests on 7 November with no mismatch ([output-verification-2026-11-07.json](output-verification-2026-11-07.json)). The ledger holds sixteen passed native rows and one failed row (the interrupted attempt, 31.38 s charged), 706.13 s of the 90-minute native cap spent.

| Tensor | Native type | Output bytes | Degraded payload bytes | Export seconds | Output SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| token_embd.weight | Q2_K | 5,812,903,424 | 176,160,768 | 17.6 | `27f2c8ddacca…` |
| token_embd.weight | Q3_K | 5,867,429,376 | 230,686,720 | 14.6 | `535588282592…` |
| blk.0.attn_v.weight | Q2_K | 6,708,731,392 | 344,064 | 15.8 | `21014512f8bd…` |
| blk.0.attn_v.weight | Q3_K | 6,708,837,888 | 450,560 | 46.7 | `9b851de361f3…` |
| blk.3.attn_v.weight | Q2_K | 6,708,731,392 | 344,064 | 39.5 | `d2500a0531d7…` |
| blk.3.attn_v.weight | Q3_K | 6,708,837,888 | 450,560 | 85.5 | `9152fd982001…` |
| blk.0.ffn_down.weight | Q2_K | 6,672,793,088 | 7,397,376 | 51.3 | `cdc5be8afd4e…` |
| blk.0.ffn_down.weight | Q3_K | 6,675,082,752 | 9,687,040 | 53.6 | `ee23af2a8481…` |
| blk.17.ffn_down.weight | Q2_K | 6,672,793,088 | 7,397,376 | 51.9 | `23d44881e351…` |
| blk.17.ffn_down.weight | Q3_K | 6,675,082,752 | 9,687,040 | 16.1 | `1da94376d402…` |
| blk.35.ffn_down.weight | Q2_K | 6,672,793,088 | 7,397,376 | 15.7 | `21111999b253…` |
| blk.35.ffn_down.weight | Q3_K | 6,675,082,752 | 9,687,040 | 33.8 | `059bb2e5572d…` |
| blk.17.ffn_gate.weight | Q2_K | 6,672,793,088 | 7,397,376 | 26.4 | `69fbc66a1eef…` |
| blk.17.ffn_gate.weight | Q3_K | 6,675,082,752 | 9,687,040 | 49.6 | `d65a32559940…` |
| blk.17.ffn_up.weight | Q2_K | 6,672,793,088 | 7,397,376 | 47.4 | `9c4329bd758d…` |
| blk.17.ffn_up.weight | Q3_K | 6,675,082,752 | 9,687,040 | 47.0 | `73c8ba87e930…` |

Export seconds are the native budget charged per export (quantizer wall time), not file-writing time. These files are diagnostics for the degrade-one screen, not shipped models; weights are not committed. No comparative native or proxy measurement existed when these manifests were committed.
