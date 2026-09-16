# Ordinary remote attention qualification

This is a real tiny WanAttention model (132,608 bytes of BF16 checkpoint tensors),
not H3 inference or a video benchmark. Runtime prepares it through the normal Model
loader and installs attention processors through the normal request override.
Its authored policy chooses SDPA. The result reports the executing processor,
installed distribution versions, output SHA256, finite values, comparison against
an independent SDPA processor with the same weights, and synchronized CUDA timing.
Runtime request metrics additionally report exact installed artifact provenance.

Copy this directory into an owned workspace before generating locks or wheels.
Use the Runtime candidate wheel being qualified and the prebuilt FA3 dependency:

```sh
uv add --no-sync /absolute/path/to/cozy_runtime-<candidate>.whl
uv add --no-sync ./vendor/cozy_kernel_flash_attn3-<candidate>.whl
uv lock --upgrade-package 'diffusers==0.40.0' --upgrade-package 'torch==2.13.0+cu130'
uv sync --locked
uv run --locked python seed.py /absolute/path/to/native-store --org YOUR_ORG
```

The Torch CUDA index must already be configured in the project (`uv` otherwise
cannot resolve a `+cu130` release). Keep Torch as a locked public registry wheel;
it need not be vendored into this source tree. The seed command writes only to
the explicitly supplied native store and prints the manifest, release and lane.
Use the same native store configured for the ordinary Cozy CLI. With the printed
manifest and length, create a fresh local alias and upload/publish it:

```sh
tfs local replace /absolute/path/to/native-store attention-kernel-benchmark \
  MANIFEST MANIFEST LENGTH --observed absent
cozy model upload local/attention-kernel-benchmark \
  YOUR_ORG/attention-kernel-benchmark --await --json
cozy model publish YOUR_ORG/attention-kernel-benchmark \
  --release=1 --lane=bf16=CHECKPOINT
```

Here both `MANIFEST` arguments are the printed `sha256:...` value (the first is
source identity), `LENGTH` is the printed manifest length, and `CHECKPOINT` is the
upload result. Seeding and alias creation do not upload or buy anything.

```sh
cozy package install . --editable --no-model-download
cozy run local/attention-kernel-benchmark/generate \
  model.model=YOUR_ORG/attention-kernel-benchmark@1/bf16 \
  seed=91 tokens=512 iterations=10 --rental=YOUR_RENTAL \
  kernel.attention=model/dit=sdpa --await --json
# Repeat with model/dit=flash-attn3, then model/dit=sdpa.
```

Use fresh request identities for all three runs. The two SDPA output hashes must
match; inspect the FA3 result's SDPA errors and Runtime's actual artifact/backend
provenance. A CPU execution deliberately refuses. Local setup or CPU tests do not
qualify CUDA execution, optional wheel compatibility, or remote A/B/A restoration.
