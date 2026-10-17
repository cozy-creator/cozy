# Real H3 style adapter headers

These are only the safetensors length prefix and original JSON header, range-read
from the immutable revisions in MANIFEST.json. They contain no tensor payloads.
`source_sha256` is Hugging Face's declared whole-file LFS digest; `header_sha256`
is locally measured over the retained bytes. Neither is full-body verification.

| Member | Keys | Content |
| --- | ---: | --- |
| `wushu_spatial_physics_*` | 416 | FP16 rank16 A/B, attention + MLP, 50 main and 2 refiner blocks; AdaLN removed upstream |
| `wushu_action_v7_*` | 600 | FP32 rank32 Kohya down/up + alpha, attention + MLP, 50 main blocks |
| `h3-realism-people-t2v-i2v-r2v` (fal) | 208 | BF16 rank32 A/B, attention only (fal `minimax-h3-trainer`) |
| `studio1939-strong` (lovis93) | 208 | BF16 rank64 A/B, attention only (fal trainer) |
| `vh5tape-comfyui` (KennethFal) | 312 | the fal layout plus F32 alpha per module |
| `minimax_h3_turbo_v4_step600_ema` (larryvrh) | 518 | bare names, attention + MLP + 51 AdaLN pairs |

The `h3/native-lora-a-b/2` (`diffusion_model.` keys) and `h3/native-lora-kohya/2`
(`lora_unet_` keys) profiles match by the `h3.lora/2` grammar: any complete A/B
pairs, optional scalar alpha, on the reviewed attention and MLP projections. Each
manifest row names its profile and the constructs it plans, or the refusal it
must raise; `session` pins plans that predate grammar matching. AdaLN factors
refuse `ADAPTER_TARGET_UNSUPPORTED` when declared and store as-is otherwise.

Every target shape fits both native H3 DiTs, but structural fit does not establish
REF2VA behavior or quality transfer. The canonical conversion splits contiguous
QKV B thirds, swaps native SwiGLU B halves, preserves A and alpha, and changes no
scalar values. Runtime validates the chosen base component and applies request
scales.

```sh
cargo test -p tensorfs-core --test h3_lora -j 2
```
