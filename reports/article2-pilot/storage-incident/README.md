# C8 storage interruption and recovery gate

The C8 run stopped during degrade-one exports, before any comparative native KL or proxy measurement. The last successful status observation reported eight passed exports and about 360 seconds of completed native budget. Those are last-observed values, pending recovery and inspection of the original ledger; no export result is reconstructed from them.

C: had only 10,911,744 bytes free when investigated. The Ubuntu disk reported read/write I/O errors, an aborted ext4 journal and a read-only remount; export and resource-sampler processes ended with bus errors. Ubuntu then failed to start. Host free space subsequently rose to about 35 GB without this agent deleting files. The cause of that increase is unconfirmed. The [kernel excerpt](kernel-excerpt.log) and [observations](observations.json) preserve the diagnostic evidence. A backup through `wsl --export --vhd` failed with a sharing violation; no completed backup is claimed.

The resource checks covered the WSL virtual disk and C8 artifact cap but omitted the actual backing Windows volume. That implementation gap allowed host exhaustion despite apparent Linux free space. Microsoft documents that WSL's displayed virtual capacity can exceed available host storage: [WSL disk-space guidance](https://learn.microsoft.com/en-us/windows/wsl/disk-space).

## Recovery before experiments

1. Preserve the original stopped Ubuntu disk. A dedicated recovery directory was created on the user's Z: network drive, which has over 6 TB available. No disk migration or repair has run.
2. Obtain approval for the system-wide WSL/Docker interruption needed to release the stale disk handle. Other workloads may use Docker; the experiment's authorization does not presume permission to interrupt them.
3. After shutdown, create and verify a backup before any repair. Inspect the exact Ubuntu device, then choose the least invasive filesystem recovery supported by the observed condition. Microsoft's repair procedure requires an elevated shell and can require closing Docker. Do not guess a block-device name from the historical log.
4. Recover the C8 ledger, manifests, failure records and sampler logs. Retain failed elapsed time and the frozen 90/30-minute cumulative caps; do not reset the experiment budget or remove an active operation without a separately recorded recovery receipt.
5. Establish sufficient actual host headroom before resuming. Task-derived exports and the completed T3 reference cache are candidates for verified archival; original weights and unrelated data are protected. Z: is a network drive, so it is not treated as a drop-in location for a running WSL disk.
6. Freeze the execution repair before comparative outcomes. The new host-storage guard requires an explicit backing-volume path in WSL, checks a 20 GiB reserve plus each proposed output, samples headroom during operations, fails on unreadable storage, and records its observed minimum. Set `C8_HOST_STORAGE_PATH=/mnt/c` only after verifying that C: still backs this distro. Scientific targets, sources, repetitions, formats and nomination rules remain frozen.

The completed T3 curve and source-bound C8 plan are already committed in the companion [evaluation PR](https://github.com/grinder81/polyglot-quant-eval/pull/50). Q3_K_M is the selected operating point under the observed English floor. This interruption supplies no evidence for or against language-informed allocation. The pilot and decision tasks remain unfinished; the holdout remains untouched.

The user's latest allowance instruction supersedes the earlier 10% threshold: stop below 2% weekly allowance. No usage reset was requested or redeemed.
