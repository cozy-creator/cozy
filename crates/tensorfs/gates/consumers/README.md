# Frozen public-cozy consumer proof

This is the existing PEP 723 author-main/Outputs Tree transport with fresh B7-style child
venvs. It runs the frozen tiny Diffusers and upstream Comfy model/library forward scripts.
It qualifies neither a ComfyUI graph/GUI/visual workflow nor Runtime's complete GPU cohort.
No script rents, updates or ends a machine. Parents own the shared allocation and artifact freeze.

The generator requires explicit final wheel files, SHA256s and full source commits; it reads
versions/Requires-Python from METADATA and verifies Runtime's stamped COMMIT. Known pre-GPU
f060/pre-fork88a sources refuse generation. No final artifact is selected by default.
The native wheel travels as a public FileAsset, avoiding the existing 32MiB script bound.
Both wheel paths also use the CLI's existing carried `[tool.uv.sources]` mechanism.

After parents supply the fixed coherent artifacts, create a NEW durable proof directory:

```bash
python3 crates/tensorfs/gates/consumers/make_payload.py \
  --creator-repo "$CREATOR_REPO" \
  --native-wheel "$NATIVE_WHEEL" --native-source "$NATIVE_SOURCE" --native-sha256 "$NATIVE_SHA256" \
  --runtime-wheel "$RUNTIME_WHEEL" --runtime-source "$RUNTIME_SOURCE" --runtime-sha256 "$RUNTIME_SHA256" \
  --output "$PROOF/consumer.py"
```

Keep the input wheels at their referenced paths until the public CLI has staged the package.
The payload checks the live Runtime source/version and exact native FileAsset. Each fresh
consumer child sets/checks deterministic cuBLAS environment before importing TensorFS/Torch
or initializing CUDA; optimized Python and interpreter fallback refuse execution. It preserves
sealed CUDA visibility, requires one GPU, and matches Torch UUID to nvidia-smi physical identity.
Child native extension bytes must match the frozen wheel, including a separately requested kernel child.

Every owned subprocess starts in its own new session with file-backed complete stdout/stderr.
Git/setup commands have a 60-second bound, each consumer 180 seconds, and an explicitly requested
kernel child 300 seconds. Timeout or surviving descendants trigger SIGTERM to that exact new
child group, at most 3 seconds grace, then SIGKILL and at most 10 seconds direct-child reap.
A timeout, residual descendants or failed reap fails the proof even if success JSON was printed.
The worker stops further consumers on child failure. It never signals the author parent/group.
The child's nvidia-smi call also has a 15-second bound inside its owned consumer group.

The existing diagnostic preparation path, performed by the parent before freezing installed files:

```bash
"$DEV_COZY" machine update "$M" --tensorhub "$H" --tensord "$DEV_COZY" \
  --runtime-wheel "$RUNTIME_WHEEL" --tensorfs-wheel "$NATIVE_WHEEL" --json
cozy machine show "$M" --tensorhub "$H" --once --json > "$PROOF/machine.json"
```

Record the actual public launch argv; run through the ordinary default home, with exactly one GPU:

```bash
run=(cozy run "$PROOF/consumer.py" --asset "native_wheel=$NATIVE_WHEEL" \
  --on "$M" --tensorhub "$H" --gpus 1 --out "$PROOF/out" --json)
python3 - "$PROOF/launch.json" "$PROOF/consumer.manifest.json" "${run[@]}" <<'PY'
import json,sys
from pathlib import Path
freeze=json.loads(Path(sys.argv[2]).read_text())
Path(sys.argv[1]).write_text(json.dumps({"payload_sha256":freeze["payload_sha256"],"argv":sys.argv[3:]},indent=2)+"\n")
PY
"${run[@]}" > "$PROOF/run.ndjson"
cozy run show "$RUN" --tensorhub "$H" --once --json > "$PROOF/run.json"
python3 crates/tensorfs/gates/consumers/validate.py "$DOWNLOADED_TREE" \
  --manifest "$PROOF/consumer.manifest.json" --launch "$PROOF/launch.json" \
  --run-show "$PROOF/run.json" --machine-show "$PROOF/machine.json"
```

`RUN` is the actual returned run number; `DOWNLOADED_TREE` is its landed Tree directory.
A CPU parent Context or `gpu=-1` row contributes no physical evidence. The validator requires
the public one-GPU launch and matching run/machine, sealed child CUDA census, Torch/NVML
UUID/physical-index/name/driver, and a positive existing public execution GPU row with matching
UUID/index and real PID/time/architecture. Missing or only negative execution rows fail.
Keep Torch installed in the **outer** wrapper (included in its dependencies) so the existing
Runtime identity path produces that row; the separately isolated kernel child remains Torch-free.
Decimal CUDA visibility entries are local ordinals; equality to an NVML physical index is not
assumed. The real child Torch UUID joined to NVML and the public execution UUID is authoritative.
Exit zero or a completed transport alone never passes.

The validator also reconstructs the exact Tree document from **every** downloaded file, including
`files.json`, and compares its SHA256/length to the completed Tree output on that supplied public
run. Its output ID must be `<run.number>/<output.name>`. This binds the retained numeric
`Context.request_id` and all receipt bytes to that run; the numeric TensorD execution ID is not
equal to Calcifer's public `run-…` ID or public run number. Swapping a same-machine receipt fails.
The wire source is Creator `aec6c548` (unchanged in this proof branch):
`crates/tensord/src/device_executor.rs::tree_binding`, `trees.rs::document`, and
`boundary_json.rs::exact`: entries sorted by path, keys sorted, compact UTF-8 JSON, no newline.
Calcifer `store.rs::record` supplies the public output ID. Runtime `Outputs.save_tree` snapshots
regular files; `internal.executor::_serialize_result` emits the typed Tree row for native retention.

The Tree contains only source, manifests, reports and complete fresh stdout/stderr. Native wheels,
venvs and temporary checkpoints remain outside it. The exact public Comfy commit is fetched
without pretrained weights; its Git source archive and setup log are retained.

Optional `make_payload.py --include-kernel` packages the existing owner-frozen
`cuda_kernel_conform.py` from `f8245484c34e45ece257715569468ced526c94fd`, SHA256
`7660d1cd97c95ca460e8f4bc884c01ee1f68b8adf8db8d755adda8fe8e053475`, plus its intact sibling
12-file cases corpus (SHA256 `92a0f4f01f4490c2f7f8c391e5eb600f5e5b33d647eed48d91ff7c21e7af57d4`).
It executes `--device 0` in a separate TensorFS-only venv with neither Torch nor Triton installed
or loaded. The exact `tensorfs.cuda-kernel-conformance/1` JSON must report all 209 full and
209 odd-byte-window bit-exact cases, all 39 schemes, the frozen metadata for every case,
zero host fallback, and actual CUDA ordinal 0. The wrapper binds its exact native extension,
sealed visibility, source and corpus bytes to the same public output Tree and GPU execution
identity as the consumers; it adds no duplicate kernel implementation.
The kernel routing oracle at the same frozen source was separately CPU-proven in run39334400067;
that is plan evidence, not device evidence. Without `--include-kernel`, `kernel` is unrequested.
`validate.py --require-kernel` refuses an unrequested or incomplete kernel proof.

Stdlib checks (synthetic GPU-shaped fixtures are validator tests, never hardware evidence):

```bash
python3 -m unittest discover -s crates/tensorfs/gates/consumers -p 'test_*.py' -v
```

The hosted no-Torch installed-Runtime contract check additionally discovers the generated public
script's actual FileAsset/Outputs signature and default accelerator, snapshots a real output Tree,
serializes its result through the installed Runtime, and checks its canonical document bytes:

```bash
"$TENSORFS_TEST_PYTHON" crates/tensorfs/gates/consumers/runtime_contract.py
```

This check requires the exact candidate Runtime wheel installed into the hosted native-wheel venv.
It does not execute either consumer, the payload body, a native build or CUDA work.

The retained real CPU negatives are Diffusers run39327753360 and Comfy run39330107744.
Their zero numerical errors cannot satisfy the GPU contract. No local heavyweight installs,
native builds, GPU execution or rental are needed to check this tooling.
