#!/usr/bin/env bash
# The `cli` suite: cozy as a person uses it on a computer with no GPU, Linux, macOS or Windows. Install, up,
# status, the local machine, the candidates through `cozy machine update local`, a weightless request through
# them, a real BiRefNet removal on the CPU, run list/show, a second executor (the machine restarted) with a new
# input, down, uninstall.
# A broken step is recorded and the suite goes on, so one run reports all of them.
#   e2e/cli/run.sh --out DIR [--cozy-bin F] [--tensorfs-wheel F] [--runtime-wheel F] [--packages DIR]
. "$(dirname "$0")/../lib.sh"
parse "$@"

step install; install_cozy
user_services
step up; cozy --json up
step status; cozy --json status | tee "$OUT/status.json"
step "the local machine"; machine_ready
step candidates; candidates

# The candidate set end to end: a request through TensorD, the Runtime and TensorFS to the reply, no model.
step "a weightless request"
text=e2e-$RANDOM$RANDOM back=''
for ((i = ${#text} - 1; i >= 0; i--)); do back+=${text:i:1}; done
if (cd "$(dirname "$0")" && cell weightless ./weightless/reverse "text=$text"); then
  grep -q "$back" "$OUT/weightless.ndjson" || check "weightless: the reply is not $back"
fi

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
machine_stop || check "the local machine did not stop"
machine_ready
dog=$(picture dog2.jpg)
if cell second "$(birefnet)" --asset "image=$dog"; then white_background "$(output second webp)" "$dog" || check "second: not a background removal"; fi

step "Calcifer reaps the machine it started"; machine_reaped
step "a restarted Calcifer adopts the machine"; calcifer_restart_adopts

step down; cozy --json down || check "cozy down failed"
step uninstall; cozy uninstall --yes || check "cozy uninstall failed"
[ ! -e "$COZY" ] || check "cozy uninstall left $COZY"
finish "cli: runs $first and $(cat "$OUT/second.number")"
