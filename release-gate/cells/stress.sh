# Suite `stress` (gate.sh --cells): memory stress cells, sourced after every product cell on the consumer pod, whose
# `sdxl`, `anima` and `h3` are the uncapped references. Uses gate.sh's `cell`, `pkg`, `log`, $CLI, $M, $R, $REF and
# the scenario's $CHAR $MOTION $OBJECT $SCENE. Every request has its own prompt and seed.
#
# A cap is varena's and torch's software limit (`cozy machine update --gpu-memory-limit`); writing it restarts the
# machine, so every block starts with new executors. Every model must complete under every cap: 8 GB is every model's
# floor, 6 GB the fleet, 3 GB the low end (owner, 10-05/10-06/10-14). The 6 GB block alternates SDXL and Anima three
# times, so each run evicts the other's weights. stress.py judges the caps; quality.py, every gate's renders.
#
# Parked in known-failing.json until fixed: Klein under any cap (runs 15168, 15236, 15239, and 15143 on a real 3070:
# CUDA out of memory in the weight bind's `to_blocked`, raised past the memory manager), and H3 at 15 s under 8 GiB
# (run 15176: step 1 407 s, step 2 still running at 25 min; uncapped 346 s in all), so the H3 cell here is 5 s.
#
# The release-gate workflow also runs the product image cells on a real 8 GB card with no cap, where the driver's own
# out-of-memory paths are (a capped card never physically fills). Qwen is parked there (run 15150: its VAE's
# repeat_interleave needs 4.8 GB more and cannot be cut). rtx-3070 offers hold at most 270 GB of disk.
STRESS_SKU=rtx-5090 STRESS_REAL_SKU=rtx-3070 STRESS_REAL_CELLS=sdxl,anima STRESS_REAL_DISK=150

cap() {
  "$CLI" machine update "$M" --gpu-memory-limit "$1" -y >"cap-$1.log" 2>&1
  local i
  for i in $(seq 12); do  # the restarted machine writes its ledger line once it is up
    timeout 60 "$CLI" machine logs "$M" memory -n 50 >"cap-$1.memory.log" 2>&1
    [ "$1" = off ] || [ "$(grep -o '"capacity":[0-9]*[^}]*"event":"ledger"' "cap-$1.memory.log" | tail -1 | grep -o '[0-9]*' | head -1)" -le $(( ${1%GiB} << 30 )) ] 2>/dev/null && break
    sleep 5
  done
  log "cap $1 on $M: $(grep -E '^status' "cap-$1.log")"
}

ANIME='masterpiece, best quality, anime illustration of'
cap 8GiB
cell s8-sdxl "$(pkg sdxl generate)" prompt="$SCENE at dusk, wide shot, photo" seed:=$((R + 101))
cell s8-anima "$(pkg anima generate)" prompt="$ANIME $OBJECT on a desk" seed:=$((R + 103))
cell s8-h3 "$(pkg minimax-h3 ref2va_turbo)" --asset ref="$REF" seed:=$((R + 104)) duration_s:=5 resolution=768p prompt="$MOTION The camera slowly pushes in."
cap 6GiB
cell s6-sdxl-1 "$(pkg sdxl generate)" prompt="$OBJECT in a museum display case, photo" seed:=$((R + 111))
cell s6-anima-1 "$(pkg anima generate)" prompt="$ANIME $SCENE" seed:=$((R + 112))
cell s6-sdxl-2 "$(pkg sdxl generate)" prompt="$CHAR walking through $SCENE, film still" seed:=$((R + 113))
cell s6-anima-2 "$(pkg anima generate)" prompt="$ANIME $CHAR at night" seed:=$((R + 114))
cell s6-sdxl-3 "$(pkg sdxl generate)" prompt="an aerial photo of $SCENE" seed:=$((R + 115))
cell s6-anima-3 "$(pkg anima generate)" prompt="$ANIME $OBJECT in the rain" seed:=$((R + 116))
cap 3GiB
cell s3-sdxl "$(pkg sdxl generate)" prompt="$CHAR, oil painting" seed:=$((R + 121))
cell s3-anima "$(pkg anima generate)" prompt="$ANIME $CHAR and $OBJECT" seed:=$((R + 123))
cap off
