# Stage 1 artifact audit: metadata only

The setup-results PR is merged. This next, deliberately small step pins public
Hub metadata for the original Global checkpoint and official Q4_K_M artifact.
See `stage1-artifact-audit.json` for exact revisions, selected filenames, advertised
byte sizes, and Hub LFS hashes. No model weights were downloaded.

The official Q4_K_M file exists and is advertised as 2,143,977,056 bytes. This is
file size, not measured tensor precision or total deployment package size.
The original checkpoint has two weight shards totaling 6,698,488,528 bytes;
configuration and tokenizer files are additional. Only required runtime files
are selected; images, alternate quantizations and official BF16 GGUF are excluded.

## Access verified after local login

A metadata HEAD request for the pinned original `config.json`, using the WSL
converter environment's default Hugging Face credentials, returned HTTP 401.
After the user logged in locally, metadata HEAD requests from the same environment
succeeded for the pinned configuration and both weight shards. Returned sizes were
1,961, 4,992,396,352, and 1,706,092,176 bytes respectively, matching the pinned
metadata. The access blocker is cleared. These requests did not download weights
or verify their contents. No credentials were recorded and no terms were accepted
by this audit.

## Evidence boundaries and next step

- Remote LFS hashes are expected values, not completed local checksum verification.
- Original configuration, actual weight dtype, GGUF tensor precision mix, and
  tokenizer/chat-template contents remain unverified until file access/download.
- Official GGUF base revision, quantizer version, command, and calibration remain
  unknown. The serving runtime version cannot fill these fields.
- Generate the same-runtime BF16 control from our pinned original after download;
  an official BF16 GGUF with unverified lineage is not a substitute.
- Recheck physical storage, download only selected files, verify local hashes,
  and create full manifests using `builds/manifest_template.json` in the next PR.

This audit is partial progress on issue #3, not completion of checkpoint C1.
