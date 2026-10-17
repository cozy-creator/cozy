# Civitai Anima base-v1.0 (2945208) header evidence

The exact 8-byte length plus JSON header of each SafeTensors file of Civitai model 2458426
version 2945208, read by byte range; no tensor payload is included. MANIFEST.json records the
Civitai file ids, the declared SHA-256 and the probed lengths.

- `2824391` (primary, `anima_baseV10.safetensors`): the ComfyUI `net.` DiT (567 keys) with its
  `net.llm_adapter.` text conditioner (118).
- `2824387` (`anima_baseV10_txt.safetensors`): the Qwen3 0.6B text encoder under `model.` (310).
- `2004692` (`qwen_image_vae_2004692.safetensors`): the Qwen-Image VAE in its original keys (194).

`crates/tensorfs-core/src/ingest/routes/anima.spec.json` generates the `anima.single_file`
route table from these. The transformer routes come from diffusers 0.40.0's
`convert_cosmos_transformer_checkpoint_to_diffusers` and the VAE's from
`convert_wan_vae_to_diffusers`. The text encoder and conditioner have no single-file
converter; upstream (`scripts/convert_anima_to_diffusers.py` at diffusers v0.40.0) strips
their prefix, which is what the spec's `null` conversion does. The whole table equals that
upstream script route for route. On the real bytes, every route's dtype, shape and leading
16 KiB equal the tensor in `circlestone-labs/Anima-Base-v1.0-Diffusers@073c3a9d`
(1,189 routes, 409 of them compared whole).

Original model materials use the included CircleStone Labs Non-Commercial License.
