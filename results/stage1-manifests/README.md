# Stage 1 artifact identities / checkpoint C1

These three manifests consolidate the merged download and conversion evidence
using the fields in `builds/manifest_template.json`. Paths are relative to the
installation root, not this repository. No model weights are committed.

| Manifest | Purpose |
| --- | --- |
| original-bf16.json | Original checkpoint with individually hashed configuration, tokenizer and weight files |
| control-bf16-gguf.json | Same-source local conversion and mandatory runtime EOS override |
| official-q4-k-m.json | Downloaded compressed artifact with independently pinned distribution revision |

Null means unknown or not applicable as explained in each manifest's notes.
The original is a directory: `artifact.files` provides hashes and sizes; its
aggregate SHA256 is null, and its size is the sum of the eight selected files.
The GGUF manifests identify single files. Whole-file bits per weight is descriptive
and includes metadata; it is not the precision of every tensor. Original directory
bits per weight is left null because it mixes weights and auxiliary files.

The official distribution revision does not identify its original weight source.
Its base revision, quantizer and calibration remain unknown. Comparing it against
our BF16 control is therefore an artifact-plus-quantization comparison, not a
strictly isolated quantizer experiment. A matched local quantization is required
for the latter claim. No invented calibration settings or build dates are filled in.

## C1 evidence index

- Environment and fixed toolchain: `reports/stage1-installed/README.md`, package
  freezes and final preflight JSON.
- Original revision, filenames, sizes and local SHA256: manifests and
  `reports/stage1-downloaded/verified-files.json`.
- Full configuration and actual original/GGUF tensor types:
  `reports/stage1-downloaded/inspection.json`.
- Conversion command, converter revision, control hash and stopping correction:
  `reports/stage1-bf16-control/README.md` and adjacent evidence.

This completes the evidence assembly for artifact/environment checkpoint C1.
It does not complete endpoint/scoring checkpoint C2: prompt and tokenizer parity,
peak runtime memory, harness configuration and scoring checks remain open.

Validation for this consolidation: JSON parsing, hashes/sizes/revisions checked
against merged evidence, required EOS override checked, and official unknown
provenance checked to remain null. Model inference was not rerun for a manifest-only
change. The successful and failed smoke runs remain preserved in their reports.
