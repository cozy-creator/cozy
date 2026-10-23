#!/usr/bin/env bash
# The `cli` suite: cozy as a person uses it on a computer with no GPU, Linux, macOS or Windows. Install, up,
# status, the local machine, the candidates through `cozy machine update local`, a real BiRefNet removal on
# the CPU, run list/show, a second executor (the machine restarted) with a new input, down, uninstall.
# A broken step is recorded and the suite goes on, so one run reports all of them.
#   e2e/cli/run.sh --out DIR [--cozy-bin F] [--tensorfs-wheel F] [--runtime-wheel F] [--packages DIR]
. "$(dirname "$0")/../lib.sh"
parse "$@"

step install; install_cozy
user_services
step up; cozy --json up
step status; cozy --json status | tee "$OUT/status.json"
step "the local machine"; cozy --json machine enable; machine_ready
step candidates; candidates

step "BiRefNet on the CPU"
dog=$(picture dog1.jpg)
if cell first "$(birefnet)" --asset "image=$dog"; then white_background "$(output first webp)" "$dog" || check "first: not a background removal"; fi
first=$(cat "$OUT/first.number")
if [ "$first" != none ]; then
  grep -q '"number": *'"$first" "$OUT/first.json" || check "cozy run show $first does not show it"
  cozy --json run list > "$OUT/runs.json" || check "cozy run list failed"
  grep -Eq "\"number\": *$first([^0-9]|\$)" "$OUT/runs.json" || check "cozy run list does not list run $first"
fi

step "a second executor"
cozy --json machine disable || check "cozy machine disable failed"
cozy --json machine enable || check "cozy machine enable failed"
machine_ready
dog=$(picture dog2.jpg)
if cell second "$(birefnet)" --asset "image=$dog"; then white_background "$(output second webp)" "$dog" || check "second: not a background removal"; fi

step down; cozy --json down || check "cozy down failed"
step uninstall; cozy uninstall --yes || check "cozy uninstall failed"
[ ! -e "$COZY" ] || check "cozy uninstall left $COZY"
finish "cli: runs $first and $(cat "$OUT/second.number")"
