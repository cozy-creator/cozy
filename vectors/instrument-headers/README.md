# Evaluation instrument carrier headers

These are real pinned safetensors headers and the Qwen8B shard index, with no tensor
payload. `MANIFEST.json` records exact header hashes, full file geometry and the
whole-file SHA256 declared by HuggingFace LFS. The latter is not a locally verified
payload hash; normal native source admission must verify it after download.

The sparse files used by the re-bank/preflight proofs exist only to validate header
geometry. They are never passed to conversion or presented as trained checkpoints.

The three profiles use the existing canonical component identity converter and keep
source keys/dtypes unchanged. Qwen is one `judge` component; Whisper is one `asr`.
Construction order was produced by the actual Eval 0.4.0 factories on Torch 2.13.0's
meta device with Transformers 5.16.1, then filtered to keys actually present in the
source. Its independent digest is banked in the manifest. No factory runs in TensorFS.

Eval owns the remaining preparation: Qwen constructor rotary buffers and the 2B tied
output destination, Whisper F32-to-F16 conversion and its tied output destination,
and all processor/tokenizer/model configuration. Those rules do not live here.

Run `python3 scripts/rebank-instruments.py --tfs target/debug/tfs` and
`cargo test -p tensorfs-core --test instrument_profiles -- --test-threads=1` to check
the native bank, exact profile assignment/order and refused incomplete/foreign inputs.
