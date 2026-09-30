#!/usr/bin/env bash
# Run tests/product as N concurrent processes, each with its own short TMPDIR (test roots
# are named under os.TempDir(), and Unix socket paths must stay under 108 bytes).
# Extra arguments go to every test binary.
#
#   scripts/product-tests.sh [-j N | --shard I/N] [--log-dir DIR] [test binary flags...]
# --shard executes one zero-based partition of the same round-robin split as -j N.
set -euo pipefail

jobs=1
shard=""
logs=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -j) jobs="$2"; shift 2 ;;
    --shard) shard="$2"; shift 2 ;;
    --log-dir) logs="$2"; shift 2 ;;
    *) break ;;
  esac
done
[[ $jobs =~ ^[1-9][0-9]*$ ]] || { echo '-j requires a positive process count' >&2; exit 2; }
if [ -n "$shard" ]; then
  [[ $shard =~ ^(0|[1-9][0-9]*)/([1-9][0-9]*)$ ]] || { echo '--shard requires a zero-based I/N' >&2; exit 2; }
  index=${BASH_REMATCH[1]}
  count=${BASH_REMATCH[2]}
  [ "$jobs" = 1 ] && [ "$index" -lt "$count" ] || { echo '--shard selects one valid process partition' >&2; exit 2; }
else
  count=$jobs
fi
if [ -n "$logs" ]; then
  mkdir -p "$logs"
  logs="$(cd "$logs" && pwd)"
fi
started=$SECONDS
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

go test -c -o "$work/product.test" "$root/tests/product"
cd "$root/tests/product"
"$work/product.test" -test.list '^Test' | grep '^Test' > "$work/all"
width=${#count}
[ "$width" -ge 2 ] || width=2
split -n "r/$count" -d -a "$width" "$work/all" "$work/part."
parts=("$work"/part.*)
if [ -n "$shard" ]; then
  printf -v suffix "%0${width}d" "$index"
  parts=("$work/part.$suffix")
fi
cat "${parts[@]}" > "$work/selected"
[ -s "$work/selected" ] || { echo 'selected partition has no tests' >&2; exit 2; }

pids=()
tmps=()
trap 'rm -rf "$work" "${tmps[@]}"' EXIT
# Background jobs of a non-interactive shell start with SIGINT ignored, and Go keeps an
# inherited ignore; restore the default so the CLIs under test stay interruptible.
for part in "${parts[@]}"; do
  tmp="$(mktemp -d /tmp/cpXXXX)"
  tmps+=("$tmp")
  TMPDIR="$tmp" env --default-signal=INT "$work/product.test" -test.run "^($(paste -sd'|' "$part"))\$" -test.v -test.timeout 30m "$@" \
    > "$part.log" 2>&1 &
  pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || status=1
done
echo "slowest:"
sed -n 's/^--- \(PASS\|FAIL\|SKIP\): \(Test[^ ]*\) (\([0-9.]*\)s)$/\3\t\1\t\2/p' "$work"/part.*.log > "$work/timings.tsv"
sort -rn "$work/timings.tsv" | sed -n '1,15p'
awk -F '\t' -v selected="$(wc -l < "$work/selected")" '{counts[$2]++} END {printf "selected=%d completed=%d pass=%d fail=%d skip=%d\n", selected, NR, counts["PASS"], counts["FAIL"], counts["SKIP"]}' "$work/timings.tsv"
echo "partition=${shard:-all/$count} elapsed=$((SECONDS-started))s"
grep -h '^--- FAIL' "$work"/part.*.log || true
if [ -n "$logs" ]; then
  cp "$work/all" "$logs/all-tests.txt"
  cp "$work/selected" "$logs/selected-tests.txt"
  cp "$work/timings.tsv" "$logs/timings.tsv"
  cp "$work"/part.*.log "$logs/"
fi
if [ "$status" != 0 ]; then
  for part in "${parts[@]}"; do
    grep -q '^--- FAIL\|^FAIL\|^panic' "$part.log" && { echo "::group::${part##*/} log"; cat "$part.log"; echo "::endgroup::"; }
  done
fi
exit "$status"
