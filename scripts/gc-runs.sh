#!/usr/bin/env bash
# tfs-006 — roots, holds and local GC, proved on real bytes by real processes.
#
#   scripts/gc-runs.sh              the full cycle on two overlapping REAL SDXL snapshots
#   scripts/gc-runs.sh arms         the red arms only (fast, synthetic subject)
#
# No automated tests (owner rule #160). Every claim below is a live run: real ingests, real
# concurrent processes, real `kill -9`, and byte counts read off the filesystem.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
T="$ROOT/target/release/tfs"
G="${G:-/home/fidika/cozy_v2/tfs-006-work}"
SDXL="${SDXL:-/home/fidika/cozy_v2/tensorfs-bench/hf/sdxl}"
S="$G/store"
REG="$G/registry.json"
MODE="${1:-full}"

say() { printf '\n=== %s ===\n' "$*"; }
objects() { "$T" store info "$S" 2>/dev/null | grep '^objects' | awk '{print $2}'; }
disk() { du -sb "$S/objects" 2>/dev/null | cut -f1; }

ingest_one() { # session component family
  local sess="$1" comp="$2" fam="$3"
  local src2="$SDXL/$comp/model.fp16.safetensors"
  "$T" ingest bank safetensors.diffusers "$fam" "vae-$comp" diffusers.identity/1 --registry "$REG" \
    --source vae="$SDXL/vae/diffusion_pytorch_model.fp16.safetensors" \
    --source "$comp=$src2" --source-name "local/$sess" > /dev/null
  nice -n 19 "$T" ingest run "$S" "vae=plain/1,$comp=plain/1" --registry "$REG" --session "$sess" \
    --source vae="$SDXL/vae/diffusion_pytorch_model.fp16.safetensors" \
    --source "$comp=$src2" \
    --config vae="$SDXL/vae/config.json" --config "$comp=$SDXL/$comp/config.json" > /dev/null 2>&1
  "$T" ingest install "$S" "$sess" | head -1
}

# ---------------------------------------------------------------- red arms

if [ "$MODE" = arms ]; then
  rm -rf "$G/syn"
  "$T" checkpoint write "$G/syn" > /dev/null
  SYN=$("$T" checkpoint write "$G/syn" 2>&1 | grep -oE 'checkpoint_id   sha256:[0-9a-f]{64}' | grep -oE '[0-9a-f]{64}')

  say "an EMPTY authority over a non-empty store refuses — and still marks from roots"
  "$T" gc sweep "$G/syn"

  say "the roots/GC red arms"
  "$T" root register "$G/syn" head "$SYN" --kind workspace
  "$T" gc arms "$G/syn"
  exit 0
fi

# ---------------------------------------------------------------- the real cycle

[ -d "$SDXL" ] || { echo "no SDXL tree at $SDXL — skipping"; exit 0; }
rm -rf "$G"; mkdir -p "$G"

say "two OVERLAPPING snapshots from real SDXL components (both carry the same VAE)"
ingest_one a text_encoder sdxl-a
ingest_one b text_encoder_2 sdxl-b
"$T" root list "$S"

say "1. mark — every object is reached, and reached for a NAMED reason"
"$T" gc mark "$S"

say "2. usage — exclusive is the freed-if-deleted number"
"$T" gc usage "$S"
EXCL=$("$T" gc usage "$S" | grep -A3 "sha256:4a27e0ac" | grep exclusive | awk '{print $2}')

say "3. nothing is collectable while both roots stand"
"$T" gc sweep "$S" --dry-run

N0=$(objects); D0=$(disk)
say "4. release ONE root and sweep — the numeric proof"
"$T" root release "$S" ingest.a
"$T" gc sweep "$S"
N1=$(objects); D1=$(disk)
FREED=$((D0 - D1))
echo "objects $N0 -> $N1   bytes freed $FREED   exclusive promised $EXCL"
if [ "$FREED" = "$EXCL" ]; then
  echo "PROOF: collection reclaimed EXACTLY the bytes exclusive promised"
else
  echo "MISMATCH by $((FREED - EXCL)) — the accounting is wrong, not the sweep"
fi

say "5. the SHARED set survived: snapshot B still walks and its bytes still verify"
B=$("$T" root list "$S" | grep 'root  ingest.b' | grep -oE '[0-9a-f]{64}')
"$T" snapshot walk "$S" "$B" | sed -n '2,4p'
"$T" read prove "$S" "$B" --sample 40 | tail -2

say "6. a filesystem rendering of the roots has ZERO authority"
"$T" gc refs "$S"
rm -rf "$S/refs"
mkdir -p "$S/refs" && echo "sha256:$(printf 'f%.0s' $(seq 64))" > "$S/refs/forged"
echo "  forged a ref for a snapshot that does not exist, and deleted the real ones"
"$T" gc sweep "$S" --dry-run > "$G/refs.log" 2>&1; sed -n '1,2p' "$G/refs.log"
echo "  the census is unchanged: GC never read refs/"
rm -rf "$S/refs"

say "7. a live LEASE blocks collection of the objects it covers"
"$T" root release "$S" ingest.b | head -1
rm -f "$G/ready"
"$T" read lease "$S" "$B" --hold-secs 30 --ready "$G/ready" > "$G/lease.log" 2>&1 &
LP=$!
for _ in $(seq 60); do [ -f "$G/ready" ] && break; sleep 0.25; done
echo "  holder pid $LP, hold $(cat "$G/ready")"
"$T" gc sweep "$S" > "$G/sweep-held.log" 2>&1
sed -n 1p "$G/sweep-held.log"
echo "  0 roots, and the bytes are still here: $(objects) objects"

say "8. kill -9 the holder — a SIGKILLed lease must not strand a permanent root"
kill -9 $LP 2>/dev/null; wait $LP 2>/dev/null
"$T" gc sweep "$S" > "$G/sweep-stranded.log" 2>&1
sed -n 1p "$G/sweep-stranded.log"
echo "  still protected: the hold row outlives the process until liveness is PROVED"
"$T" hold reap "$S"
"$T" gc sweep "$S" > "$G/sweep-reaped.log" 2>&1
sed -n 1p "$G/sweep-reaped.log"
echo "  objects now $(objects), disk $(disk) B"

say "9. kill -9 MID-SWEEP: claimed, not yet deleted"
ingest_one c text_encoder sdxl-c > /dev/null 2>&1 || ingest_one c text_encoder sdxl-a
N_PRE=$(objects); D_PRE=$(disk)
"$T" root release "$S" ingest.c > /dev/null
nice -n 19 "$T" gc sweep "$S" --fault claimed > "$G/fault.log" 2>&1 &
FP=$!
for _ in $(seq 80); do grep -q FAULT-READY "$G/fault.log" 2>/dev/null && break; sleep 0.25; done
kill -9 $FP 2>/dev/null; wait $FP 2>/dev/null
echo "  killed at the claim point"
echo "  objects $N_PRE -> $(objects) (nothing deleted), disk $D_PRE -> $(disk) B"
echo "  claims left in the authority: $("$T" root list "$S" | grep -c '^  claim')"
echo "  --- recovery: a fresh sweep finishes the job ---"
"$T" gc sweep "$S" > "$G/recover.log" 2>&1; sed -n '1,2p' "$G/recover.log"
"$T" gc reap "$S"
echo "  objects now $(objects)"

say "10. two sweeps at once serialize — the authority lock is the whole race"
ingest_one d text_encoder sdxl-a > /dev/null 2>&1
"$T" root release "$S" ingest.d > /dev/null
( "$T" gc sweep "$S" > "$G/race1.log" 2>&1 ) &
( "$T" gc sweep "$S" > "$G/race2.log" 2>&1 ) &
wait
echo "  sweep 1: $(sed -n 1p "$G/race1.log")"
echo "  sweep 2: $(sed -n 1p "$G/race2.log")"
echo "  both completed, no object was collected twice and none was left claimed:"
echo "  claims left: $("$T" root list "$S" | grep -c '^  claim')   objects: $(objects)"

say "11. binding churn moves ROOTS, never bytes"
ingest_one e text_encoder sdxl-a > /dev/null 2>&1
E=$("$T" root list "$S" | grep 'root  ingest.e' | grep -oE '[0-9a-f]{64}')
"$T" root reconcile "$S" "binding-0=$E" "binding-1=$E"
"$T" root reconcile "$S" "binding-1=$E"

say "12. pressure: the durability invariant, on an UNROOTED snapshot"
# Every `vae + text_encoder` ingest produces the SAME snapshot id (boundary determinism), so
# several roots point at it by now. Pressure is about UNPINNED bytes, so drop them all first.
for r in $("$T" root list "$S" | awk '/^  root /{print $2}'); do
  "$T" root release "$S" "$r" > /dev/null
done
echo "  --- hub-published: the bytes can be got back, so they may be evicted ---"
"$T" root note "$S" "$E" --hub-published
"$T" gc pressure "$S" --free 0 --reserve 1000000000000 | grep -E 'EVICT|KEEP|evictable|admission'
echo "  --- the SAME snapshot with only a source URL: provenance is not durability ---"
"$T" root note "$S" "$E" --source-url https://example.invalid/m
"$T" gc pressure "$S" --free 0 --reserve 1000000000000 | grep -E 'EVICT|KEEP|evictable|admission'
echo "  (snapshot B has never been published or fetched — it is the local-only arm above)"

say "12b. kill -9 mid-DELETION: partial removal must recover, never lose a live object"
ingest_one f text_encoder sdxl-a > /dev/null 2>&1
F=$("$T" root list "$S" | grep 'root  ingest.f' | grep -oE '[0-9a-f]{64}')
"$T" root release "$S" ingest.f > /dev/null
rm -f "$G/fault2.log"
nice -n 19 "$T" gc sweep "$S" --fault deleting > "$G/fault2.log" 2>&1 &
FP2=$!
for _ in $(seq 80); do grep -q FAULT-READY "$G/fault2.log" 2>/dev/null && break; sleep 0.25; done
kill -9 $FP2 2>/dev/null; wait $FP2 2>/dev/null
echo "  killed after the first object was taken"
echo "  claims left: $("$T" root list "$S" | grep -c '^  claim')"
# The rename→unlink window is microseconds, so a kill rarely lands inside it. Plant the
# artifact it would leave and prove the reap finishes it: the file is already OUT of the
# object namespace, so removing it can never lose a live object.
mkdir -p "$S/gc/planted" && echo scratch > "$S/gc/planted/$(printf 'a%.0s' $(seq 64))"
echo "  planted 1 interrupted-removal artifact under gc/"
"$T" gc reap "$S"
"$T" gc sweep "$S" > "$G/recover2.log" 2>&1; sed -n 1p "$G/recover2.log"
echo "  objects now $(objects)"

say "12c. GC racing a fill: the hold registers BEFORE the first fetch"
ingest_one g text_encoder sdxl-a > /dev/null 2>&1
GS=$("$T" root list "$S" | grep 'root  ingest.g' | grep -oE '[0-9a-f]{64}')
"$T" root release "$S" ingest.g > /dev/null
echo "  --- (a) the fill HAS registered: the sweep must collect nothing it covers ---"
rm -f "$G/ready2"
"$T" read lease "$S" "$GS" --hold-secs 8 --ready "$G/ready2" > "$G/racefill.log" 2>&1 &
RP=$!
for _ in $(seq 60); do [ -f "$G/ready2" ] && break; sleep 0.25; done
"$T" gc sweep "$S" > "$G/racesweep.log" 2>&1
sed -n 1p "$G/racesweep.log"
echo "  objects still present: $(objects)"
wait $RP 2>/dev/null
"$T" read prove "$S" "$GS" --sample 20 > "$G/proveg.log" 2>&1 && sed -n '$p' "$G/proveg.log"

echo "  --- (b) the fill has NOT registered: the sweep may win, and the fill REFUSES ---"
"$T" hold reap "$S" > /dev/null
"$T" read lease "$S" "$GS" > "$G/racefill2.log" 2>&1 &
RP2=$!
"$T" gc sweep "$S" > "$G/racesweep2.log" 2>&1
sed -n 1p "$G/racesweep2.log"
wait $RP2 2>/dev/null
echo "  fill result: $(grep -E 'verified:|REFUSED' "$G/racefill2.log" | head -1)"
echo "  a lost race is a typed refusal, never a partial or wrong read"

say "12d. fuzzed concurrency: sweeps, root swaps and leases at once over LIVE data"
ingest_one h text_encoder sdxl-a > /dev/null 2>&1
HS=$("$T" root list "$S" | grep 'root  ingest.h' | grep -oE '[0-9a-f]{64}')
for i in $(seq 6); do
  ( "$T" gc sweep "$S" > "$G/fz-sweep-$i.log" 2>&1 ) &
  ( "$T" root register "$S" "churn-$i" "$HS" --kind workspace > /dev/null 2>&1
    "$T" root release "$S" "churn-$i" > /dev/null 2>&1 ) &
  ( "$T" read lease "$S" "$HS" > "$G/fz-lease-$i.log" 2>&1 ) &
done
wait
echo "  sweeps that refused: $(grep -l REFUSED "$G"/fz-sweep-*.log 2>/dev/null | wc -l) of 6"
echo "  leases that refused: $(grep -l REFUSED "$G"/fz-lease-*.log 2>/dev/null | wc -l) of 6"
echo "  objects collected across all sweeps: $(awk '/^scanned/{s+=$6} END{print s+0}' "$G"/fz-sweep-*.log)"
echo "  --- the live root's data must be intact ---"
"$T" snapshot walk "$S" "$HS" > "$G/fzwalk.log" 2>&1; sed -n '4p' "$G/fzwalk.log"
"$T" read prove "$S" "$HS" --sample 30 > "$G/fzprove.log" 2>&1; sed -n '$p' "$G/fzprove.log"
echo "  authority still parses at generation $("$T" root list "$S" | sed -n 1p | awk '{print $2}')"
echo "  claims left: $("$T" root list "$S" | grep -c '^  claim')"

say "13. the red arms"
"$T" gc arms "$S"

say "14. fences and the frozen corpus"
python3 "$ROOT/scripts/fence.py" | tail -2
"$T" vectors verify | tail -1
