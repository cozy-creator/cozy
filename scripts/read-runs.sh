#!/usr/bin/env bash
# tfs-005 — the verified read path, the projection ladder, and their red arms, run live.
#
#   scripts/read-runs.sh              arms + projection over a fresh synthetic store
#   scripts/read-runs.sh real         the real-SDXL arms (needs BENCH, read-only)
#   scripts/read-runs.sh bench        the reader-pool / window / ring sweep on real bytes
#
# There are NO automated tests here (owner rule #160): every line below is a real process
# against a real store, and every red arm is planted into real bytes.
#
# BENCH points at the tfs-018 store, which is a QUARRY: this script only ever makes a
# hardlinked working copy of it, and the destructive arms refuse to run on a store whose
# objects share inodes with another tree.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TFS="$ROOT/target/release/tfs"
WORK="${WORK:-/home/fidika/cozy_v2/tfs-005-work}"
BENCH="${BENCH:-/home/fidika/cozy_v2/tensorfs-bench/store}"
BENCH_SNAP="${BENCH_SNAP:-108c585af313e4db4410ed6c20a313b072fd84610ef786246e8c846847d340e6}"
MODE="${1:-arms}"

say() { printf '\n=== %s ===\n' "$*"; }

evict() { # posix_fadvise(DONTNEED) — there is no passwordless sudo on this box for drop_caches
  python3 - "$1" <<'PY'
import os, sys, pathlib
root = pathlib.Path(sys.argv[1])
n = b = 0
for p in (root / "objects/sha256").rglob("*"):
    if not p.is_file():
        continue
    fd = os.open(p, os.O_RDONLY)
    try:
        st = os.fstat(fd)
        os.posix_fadvise(fd, 0, st.st_size, os.POSIX_FADV_DONTNEED)
        n += 1; b += st.st_size
    finally:
        os.close(fd)
print(f"evicted {n} objects, {b} B")
PY
}

work_copy() { # hardlinked objects (records stay valid: same dev/ino), everything else real
  rm -rf "$WORK/bench"
  mkdir -p "$WORK/bench"
  cp -al "$BENCH/objects" "$WORK/bench/objects"
  cp -a "$BENCH/store.json" "$BENCH/verified" "$WORK/bench/"
  mkdir -p "$WORK/bench/tmp" "$WORK/bench/quarantine"
}

# ---------------------------------------------------------------- a fresh subject

if [ "$MODE" = "arms" ]; then
  rm -rf "$WORK/syn"
  say "a real checkpoint through the real CAS"
  "$TFS" checkpoint write "$WORK/syn" | tail -5
  SYN=$("$TFS" checkpoint write "$WORK/syn" 2>&1 | grep -oE 'checkpoint_id   sha256:[0-9a-f]{64}' | head -1 | grep -oE '[0-9a-f]{64}')
  echo "snapshot $SYN"

  say "1. the lease: acquire, verify, release"
  "$TFS" read lease "$WORK/syn" "$SYN"

  say "2. a lease DROPPED instead of released — the row is kept for the reaper"
  "$TFS" read lease "$WORK/syn" "$SYN" --drop-unreleased

  say "3. the plan: destination order is the caller's, never the header's"
  "$TFS" read plan "$WORK/syn" "$SYN" --window 1048576

  say "4. the pooled fill into a caller-owned ring"
  "$TFS" read fill "$WORK/syn" "$SYN" --workers 4 --window 1048576 --ring 67108864

  say "5. the same bytes through the SEQUENTIAL document path — digests must agree"
  "$TFS" read verify-fill "$WORK/syn" "$SYN"

  say "6. delivered bytes vs the digests the header declares"
  "$TFS" read prove "$WORK/syn" "$SYN" --sample 40

  say "7. O_DIRECT asked for, buffered delivered — the receipt says so"
  "$TFS" read fill "$WORK/syn" "$SYN" --workers 2 --ring 33554432 --direct --no-digest | grep -E 'RECEIPT|throughput'

  say "8. torn storage UNDER a live lease"
  "$TFS" read tear "$WORK/syn" "$SYN"

  say "9. the read/projection red arms"
  "$TFS" read arms "$WORK/syn" "$SYN"

  say "10. projection tier 2 — relative symlinks, no weight paths"
  rm -rf "$WORK/proj"; "$TFS" project "$WORK/syn" "$SYN" "$WORK/proj"
  find "$WORK/proj" -type l | head -1 | xargs -I{} sh -c 'echo "  {} -> $(readlink {})"'

  say "11. a filesystem without symlinks falls back to copies, and confesses"
  rm -rf "$WORK/proj2"; "$TFS" project "$WORK/syn" "$SYN" "$WORK/proj2" --no-symlink | head -4

  say "12. tier 3 — a RETAINED carrier blob, streamed"
  head -c 5000000 /dev/urandom > "$WORK/carrier.bin"
  "$TFS" snapshot blob put "$WORK/syn" "$WORK/carrier.bin" | head -3
  "$TFS" materialize "$WORK/syn" "$WORK/carrier.bin" "$WORK/carrier.out"
  cmp "$WORK/carrier.bin" "$WORK/carrier.out" && echo "  byte-identical to the source"

  say "13. the ONE fill transform: arithmetic decomposition + the block-granularity rule"
  "$TFS" read transform

  say "14. fences (three of them armed by this issue)"
  python3 "$ROOT/scripts/fence.py"

  say "15. the frozen corpus"
  "$TFS" vectors verify | tail -1
  exit 0
fi

# ---------------------------------------------------------------- the real artifact

[ -d "$BENCH" ] || { echo "no store at $BENCH — skipping the real arms"; exit 0; }
work_copy

if [ "$MODE" = "real" ]; then
  say "the real 4.78 GiB SDXL UNet checkpoint"
  "$TFS" read plan "$WORK/bench" "$BENCH_SNAP" | head -6
  "$TFS" read lease "$WORK/bench" "$BENCH_SNAP"
  "$TFS" read prove "$WORK/bench" "$BENCH_SNAP" --sample 120
  "$TFS" read arms "$WORK/bench" "$BENCH_SNAP"
  say "the destructive arm refuses on a hardlinked working copy"
  "$TFS" read tear "$WORK/bench" "$BENCH_SNAP"
  exit 0
fi

# ---------------------------------------------------------------- the sweep

say "reader-pool depth (cold = fadvise-evicted, warm = the run straight after)"
printf '%-8s %-10s %-9s %-9s\n' workers window cold warm
for w in 1 2 4 8 16; do
  evict "$WORK/bench" > /dev/null
  cold=$(nice -n 19 "$TFS" read fill "$WORK/bench" "$BENCH_SNAP" --workers "$w" --window 16777216 --ring 1073741824 --no-digest | grep throughput | awk '{print $2}')
  warm=$(nice -n 19 "$TFS" read fill "$WORK/bench" "$BENCH_SNAP" --workers "$w" --window 16777216 --ring 1073741824 --no-digest | grep throughput | awk '{print $2}')
  printf '%-8s %-10s %-9s %-9s GB/s\n' "$w" 16MiB "$cold" "$warm"
done

say "transfer window at depth 8 (cold)"
for win in 1048576 4194304 16777216 67108864; do
  evict "$WORK/bench" > /dev/null
  v=$(nice -n 19 "$TFS" read fill "$WORK/bench" "$BENCH_SNAP" --workers 8 --window "$win" --ring 1073741824 --no-digest | grep throughput | awk '{print $2}')
  printf '  window %-10s %s GB/s\n' "$win" "$v"
done

say "caller ring size at depth 8 (cold) — every batch boundary is a pool drain"
for ring in 67108864 268435456 1073741824; do
  evict "$WORK/bench" > /dev/null
  v=$(nice -n 19 "$TFS" read fill "$WORK/bench" "$BENCH_SNAP" --workers 8 --window 4194304 --ring "$ring" --no-digest | grep -E 'throughput|batches' | tr '\n' ' ')
  printf '  ring %-11s %s\n' "$ring" "$v"
done

say "the denominator: the SAME object files, at the SAME depth, no tensorfs"
# Cold-vs-cold or it is not a comparison: the raw arm gets its own eviction.
cat > "$WORK/raw.py" <<'RAWPY'
import os, sys, time, pathlib, concurrent.futures as cf
root, workers = pathlib.Path(sys.argv[1]), int(sys.argv[2])
files = sorted(p for p in (root / "objects/sha256").rglob("*") if p.is_file())
def rd(p):
    n = 0
    with open(p, "rb", buffering=0) as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                return n
            n += len(b)
t0 = time.time()
with cf.ThreadPoolExecutor(workers) as ex:
    total = sum(ex.map(rd, files))
d = time.time() - t0
print(f"{total/1e9/d:.3f}")
RAWPY
printf '%-6s %-10s %-10s %-10s %-10s\n' depth tfs-cold raw-cold tfs-warm raw-warm
for d in 1 4 16; do
  evict "$WORK/bench" > /dev/null
  tc=$(nice -n 19 "$TFS" read fill "$WORK/bench" "$BENCH_SNAP" --workers "$d" --window 4194304 --ring 1073741824 --no-digest | grep throughput | awk '{print $2}')
  tw=$(nice -n 19 "$TFS" read fill "$WORK/bench" "$BENCH_SNAP" --workers "$d" --window 4194304 --ring 1073741824 --no-digest | grep throughput | awk '{print $2}')
  evict "$WORK/bench" > /dev/null
  rc=$(nice -n 19 python3 "$WORK/raw.py" "$WORK/bench" "$d")
  rw=$(nice -n 19 python3 "$WORK/raw.py" "$WORK/bench" "$d")
  printf '%-6s %-10s %-10s %-10s %-10s GB/s\n' "qd$d" "$tc" "$rc" "$tw" "$rw"
done
echo
echo "COLD is the storage result. Warm figures above ~6 GB/s are host memory bandwidth,"
echo "not a device measurement, and both arms converge there at depth."
