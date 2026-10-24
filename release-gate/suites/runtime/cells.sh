# Suite `runtime`: every first-party package's default call on this SKU (products.sh), then a restart and SDXL in a
# second executor. It proves every model still installs, loads, runs, serves its kernels and holds its numbers.
for p in "${PRODUCTS[@]}"; do product "$p"; done
# Second executor process: a GPU memory limit write restarts the machine, so every executor is new.
if [ -z "$ONLY" ] || [[ $ONLY == *",sdxl-p2,"* || $ONLY == *",sdxl,"* ]]; then
  "$CLI" machine update "$M" --gpu-memory-limit off -y >restart.log 2>&1; log "restarted $M: $(grep -E '^status' restart.log)"
fi
cell sdxl-p2 "$(pkg sdxl generate)" prompt="$OBJECT on a wooden table, product photo" seed:=$((R + 7))
