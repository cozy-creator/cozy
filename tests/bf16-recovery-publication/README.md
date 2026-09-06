# Exact BF16 checkpoint recovery

This action harness was used to recover the original H3 BF16 checkpoint after its
producer container stopped before publication. It is a task-owned executable,
not a new product upload path. The existing Creator client owns authorization,
publication, verification, and finalization.

The source preparation roots were restored from the existing NFS repository cache
into a new isolated TensorFS Store using public TensorFS 0.3.12. Two legitimate
unpublished local checkpoints retained those verified closures. Existing published
H3 tools 2.2.6 and Runtime 0.2.26 reproduced the exact original BF16 manifest,
`sha256:c3b72104aff017e77fc9f5b427c8cd0c5f8dc7c4cbe604584e89f9fc24f1025a`.
The original Creator object inventory and the recovered native inventory match
exactly: 5,424 objects, 195,021,215,719 bytes.

Actual Hub finalization reused 5,422 held objects (195,020,590,816 bytes) and required
only two successful metadata PUTs: the 624,739-byte header and 164-byte manifest.
Both were read back over HTTPS and checked against their exact SHA256 and length.
No tensor payload was reuploaded. The native inheritance receipt reports zero
payload reads or hashes and no added payload objects. Full source restoration had
already verified every declared source object in the isolated Store.

The helper scopes mutations to `paul/minimax-h3`, publication operation
`bf16-recovery-20260906`, and that exact manifest. It takes the directory containing
the original sorted `objects.jsonl` as its sole argument and accepts one typed JSON
command per stdin line: open, grants, verify, metadata, or finalize. Grants contain
capabilities: stdout must be a private pipe, never a terminal or log. Account
credentials stay in Creator's existing accountauth implementation. The action
driver allowed uploads of only the exact two metadata files and refused any
unexpected missing tensor payload. The two grants were spent through a no-echo
SSH pipe and ordinary HTTPS, with redirects refused; no credentials or signed URLs
were persisted or emitted in evidence.

Evidence is banked in the workspace at
`outputs/h3-resume-20260905/bf16-recovery/`, including the source restore/assembly
proof, action log, object census, upload results, canonical finalization, and exact
metadata readback. Release selection is a separate owner action; this harness does
not publish a release or assert video quality.
