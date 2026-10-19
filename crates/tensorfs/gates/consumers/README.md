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
UUID/physical-index/name/driver. Positive public GPU measurement rows, when present, must
agree; absent rows are reported as absent. Exit zero or a completed transport alone never passes.

The Tree contains only source, manifests, reports and complete fresh stdout/stderr. Native wheels,
venvs and temporary checkpoints remain outside it. The exact public Comfy commit is fetched
without pretrained weights; its Git source archive and setup log are retained.

An optional frozen kernel script uses `--kernel-script`, `--kernel-source`, `--kernel-sha256`
and `--kernel-manifest` together. The manifest declares `proof`, unique `cases`, and explicit
`argv` (supports `{native_source}`/`{wheel}` placeholders). This requests a separate clean
TensorFS-only child without Torch/Triton installed or loaded. Its JSON contract currently
requires artifact/source/UUID/census fields plus one executed, non-skipped native CUDA row
per declared case. Reconcile that contract with the kernel owner before using the hook;
no kernels are implemented here. Without this input, `kernel` is explicitly unrequested.
`validate.py --require-kernel` refuses an unrequested or incomplete kernel proof.

Stdlib checks (synthetic GPU-shaped fixtures are validator tests, never hardware evidence):

```bash
python3 -m unittest discover -s crates/tensorfs/gates/consumers -p 'test_*.py' -v
```

The retained real CPU negatives are Diffusers run39327753360 and Comfy run39330107744.
Their zero numerical errors cannot satisfy the GPU contract. No local heavyweight installs,
native builds, GPU execution or rental are needed to check this tooling.
