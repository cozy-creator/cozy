# The first-party packages' cells, sourced by suites runtime and packages: one function per package, its default call
# with a scenario prompt and its own seed. A cell that takes another's output (SeedVR2 H3's clip, BiRefNet and the
# describer Klein's image, Qwen-Image's edits its render) falls back to the gate's reference when that one did not run.
PRODUCTS=(minimax-h3 seedvr2 sdxl flux-3-klein qwen-image-2 anima birefnet qwen3.5 fibo-2 hidream-o1 moss-image qwen-image)
# --first runs these packages before the rest, in its order: a clean standalone number before models stack.
[ -z "$FIRST" ] || mapfile -t PRODUCTS < <(tr , '\n' <<< "$FIRST"; printf '%s\n' "${PRODUCTS[@]}" | grep -vxF -f <(tr , '\n' <<< "$FIRST"))
product() {
  local f=product_${1//[-.]/_}
  if declare -F "$f" >/dev/null; then "$f"; else log "$1: no product cells"; fi
}

product_minimax_h3() {
  # H3's references: BiRefNet cleans the character image and Qwen3.5 describes it (greedy), as for a user's character.
  # While cr-215 is open (lead 10-23) the reference is the image only; the 3 s video reference returns with its fix.
  cell h3 "$(pkg minimax-h3 ref2va_turbo)" references:="[{\"name\": \"$NAME\", \"kind\": \"character\", \"image\": \"$REF\"}]" \
    seed:=$((R + 1)) duration_s:=15 resolution=768p prompt="<$NAME> $MOTION"
  CLIP=$(output h3 mp4)
}
product_seedvr2() {
  local src=${CLIP:-$(ls "$B/ref-$SKU-clip.mp4" 2>/dev/null)}
  if [ -n "$src" ]; then nice -n 19 ffmpeg -v error -y -i "$src" -t 1 -an -c:v libx264 -pix_fmt yuv420p clip-1s.mp4
  else nice -n 19 ffmpeg -v error -y -loop 1 -i "$REF" -t 1 -r 24 -vf scale=-2:768 -c:v libx264 -pix_fmt yuv420p clip-1s.mp4; fi
  cell seedvr2 "$(pkg seedvr2 upscale)" --asset video="$OUT/clip-1s.mp4" resolution=1080p seed:=$((R + 2))
}
product_sdxl() { cell sdxl "$(pkg sdxl generate)" prompt="$SCENE, photo" seed:=$((R + 3)); }
product_flux_3_klein() {
  cell klein "$(pkg flux-3-klein generate_turbo)" prompt="$CHAR" prompt_upsampling:=true seed:=$((R + 4))
  K=$(output klein 'png|webp|jpg')
}
product_qwen_image_2() {
  cell qwen "$(pkg qwen-image-2 generate)" seed:=$((R + 5)) prompt="This is an RGBA image with transparency. $OBJECT, semi-transparent with soft reflections. The image has alpha channel and the background is transparent."
}
product_anima() { cell anima "$(pkg anima generate)" prompt="masterpiece, best quality, anime illustration of $CHAR" seed:=$((R + 6)); }
product_birefnet() { cell birefnet "$(pkg birefnet remove_background)" --asset image="${K:-$REF}"; }
product_qwen3_5() {
  cell describer "$(pkg qwen3.5 ask)" prompt="Describe this character: appearance, clothing and pose." --asset images="${K:-$REF}"
}
# Fibo's prompt upsampler (a greedy VLM) runs, as by default.
product_fibo_2() { cell fibo "$(pkg fibo-2 generate)" prompt="$SCENE" prompt_upsampling:=true seed:=$((R + 8)); }
product_hidream_o1() { cell hidream "$(pkg hidream-o1 generate)" prompt="$CHAR, cinematic portrait" seed:=$((R + 9)); }
product_moss_image() { cell moss "$(pkg moss-image generate)" prompt="$OBJECT, studio shot" seed:=$((R + 10)); }
product_qwen_image() {
  cell qwen-image "$(pkg qwen-image generate)" prompt="$SCENE at dusk, wide shot" seed:=$((R + 11))
  # Two checkpoints of one release that share a component, in turn on one executor with no restart between them
  # (swap.py; TFS 0.15.0 keyed varena's weight sets by release/component, run 15744): Qwen-Image-2512 for text,
  # Qwen-Image-Edit-2511 for an edit, text again; then the Lightning LoRAs, turbo_lora for text and edit_turbo_lora for
  # an edit. No first-party SDXL LoRA exists to swap.
  local qi; qi=$(output qwen-image 'png|webp|jpg')
  cell qwen-image-edit "$(pkg qwen-image generate)" prompt="$SCENE at dusk, covered in fresh snow, wide shot" \
    --asset reference_images.0="${qi:-$REF}" seed:=$((R + 12))
  cell qwen-image-t2i "$(pkg qwen-image generate)" prompt="$OBJECT on a beach at noon, photo" seed:=$((R + 13))
  cell qwen-image-turbo "$(pkg qwen-image generate_turbo)" prompt="$CHAR on a rainy street at night, photo" seed:=$((R + 14))
  cell qwen-image-turbo-edit "$(pkg qwen-image generate_turbo)" prompt="$SCENE in bright morning sun, wide shot" \
    --asset reference_images.0="${qi:-$REF}" seed:=$((R + 15))
}
