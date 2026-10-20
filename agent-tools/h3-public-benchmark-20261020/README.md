# Public H3 benchmark wrapper — preparation only

This owned prefix wraps exact `h3_load.py` source `5ea7092ce12d51630c4114539c4137b8ae467826` using the existing public PEP723 author-main/Outputs Tree interface. It changes no benchmark or product source and performs no rental. Candidate fields in `candidate-template.json` are intentionally unresolved; no final payload has been rendered or run.

The fixed staging client is `c7accfc6af5e6775fa12d2d7f6a8977db3c64cd4`, portable binary SHA256 `b0712ca8c21ce695e52e412a135778c648f0dac8b8f4c9e991788a685a31ebe9`. Corrected metadata archive SHA256 is `ceafd89e2e5c61b73f2153663b3ea9de9c45d5b7c0121554758715eaba9da25a`; its unchanged D10 header is `0f03804af2280bac6c81c633cd058f4dfbb5fd71f5db0a3c5829907e4ef21f2e`. All 23 metadata files have fixed hashes in `metadata-files.json`.

The rendered public signature is `main(ctx, *, stage_client: FileAsset, metadata_bundle: FileAsset, native_wheel: FileAsset, deadline_unix: int, out: Outputs) -> Tree`, with fixed public `AssetBound` limits. It uses only FileAsset reads, bounded subprocesses, and `out.save_tree`. No new transport or credentials are introduced.

Before the only client `--download`, an owned child checks one sealed H100 UUID, at least **79,305,338,149 free GPU bytes**, **156,316,919,880 available host/cgroup bytes**, and **228,760,140,584 disk bytes + 8 GiB** on the fresh work filesystem. A two-GPU rental can run one assigned card; both GPUs' VRAM is never pooled. Actual Torch/NVML UUID and free capacity are recorded, not inferred from a catalog label. A failed check returns failure receipts before model data transfer.

The parent must pass an integer Unix deadline at most 1,200 seconds ahead. Each child timeout uses only the remaining budget, reserving the existing helper's 3-second TERM plus 10-second KILL/reap bound. The parent's independent rental stop/cleanup guard remains required; this wrapper does not enforce provider billing or own the lease. A failed, timed-out, unreaped, or surviving-child stage stops later work.

The sequence is local artifact/provenance checks, metadata-only client preflight, actual capacity check, explicit client download and complete closure verification, then the unmodified H3 benchmark. There are **60 measured rows**: two CPU readers and four GPU readers, each cold/warm, for five rotated repetitions. The original six initialization reads and 30 complete warmup reads remain intact. All returned data is checked in full; no window, decode substitution, offload, or CPU fallback is added. Cold page-cache eviction remains advisory. Slower complete results remain valid data, with `parity_pass=false`.

`render.py` requires exact Runtime/native wheel sources and hashes, a supplied review-receipt hash, and exact dependency pins. It verifies actual wheel metadata, Runtime COMMIT/native kernels, TensorFS extension identity, and all package file hashes without importing either package. The wheel paths remain public `[tool.uv.sources]`; the exact `+cu130` Torch build uses the explicit named PyTorch index. No package or model is installed by the renderer. The supplied review receipt is provenance to inspect, not a fabricated hardware test.

After the parent has reviewed a coherent candidate, the preparation command is:

```sh
python3 agent-tools/h3-public-benchmark-20261020/render.py \
  --candidate reviewed-candidate.json --pair-receipt actual-pair-receipt.json \
  --runtime-wheel /exact/runtime.whl --native-wheel /exact/tensorfs.whl \
  --requirements exact-requirements.txt --creator-repo /path/to/cozy-creator \
  --output /new/proof/h3.py
```

The candidate JSON has `runtime` and `native` objects containing full `source` and `sha256` pins, and `pair_receipt_sha256`. The renderer derives the remaining metadata from the supplied bytes. It refuses unresolved fields, dirty/uncommitted tool source, existing output paths, or a drifted frozen harness. Keep referenced wheels at their paths until the ordinary CLI stages the package.

Only after separate capacity/budget/lease authorization, the existing public launch shape is:

```sh
cozy run /new/proof/h3.py deadline_unix=ACTUAL_PARENT_DEADLINE \
  --asset stage_client=/verified/h3_chunk_stage \
  --asset metadata_bundle=/verified/h3-int8-d10-metadata.tar.gz \
  --asset native_wheel=/exact/tensorfs.whl --gpus 1 --on MACHINE --out RECEIPTS --json
```

Record that exact argv and deadline in `launch.json`; after the ordinary public run completes, supply its actual run/machine JSON to:

```sh
python3 agent-tools/h3-public-benchmark-20261020/validate.py RECEIPTS \
  --manifest /new/proof/h3.manifest.json --launch launch.json \
  --run-show run.json --machine-show machine.json
```

The validator checks all full rows, source/package identity, the best control per cell, parent deadline, completed owned children, selected UUID and positive public GPU execution, and the exact returned public Tree digest. It does not accept process exit alone. R4 link/lifecycle/performance acceptance remains a separate review.

The returned Tree contains only receipt JSON, frozen source and full logs. Chunks, normalized fixtures, wheels and temporary environments remain outside it. Stdlib-only checks are `python3 -m unittest discover -s agent-tools/h3-public-benchmark-20261020 -p 'test_*.py' -v`; their synthetic GPU-shaped fixtures are not hardware evidence.
