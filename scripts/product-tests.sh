#!/usr/bin/env bash
# Run tests/product as N concurrent processes, each with its own short TMPDIR (test roots
# are named under os.TempDir(), and Unix socket paths must stay under 108 bytes).
# Extra arguments go to every test binary.
#
#   scripts/product-tests.sh [-j N] [test binary flags...]
set -euo pipefail

jobs=1
if [ "${1:-}" = -j ]; then jobs="$2"; shift 2; fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

go test -c -o "$work/product.test" "$root/tests/product"
cd "$root/tests/product"
"$work/product.test" -test.list '^Test' | grep '^Test' > "$work/all"
split -n "r/$jobs" -d "$work/all" "$work/part."

pids=()
tmps=()
trap 'rm -rf "$work" "${tmps[@]}"' EXIT
# Background jobs of a non-interactive shell start with SIGINT ignored, and Go keeps an
# inherited ignore; restore the default so the CLIs under test stay interruptible.
for part in "$work"/part.*; do
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
cat "$work"/part.??.log | sed -n 's/^--- \(PASS\|FAIL\|SKIP\): \(Test[^ ]*\) (\([0-9.]*\)s)$/\3 \1 \2/p' | sort -rn | head -15 || true
grep -h '^--- FAIL' "$work"/part.??.log || true
if [ "$status" != 0 ]; then
  for part in "$work"/part.??; do
    grep -q '^--- FAIL\|^FAIL\|^panic' "$part.log" && { echo "::group::${part##*/} log"; cat "$part.log"; echo "::endgroup::"; }
  done
fi
exit "$status"
