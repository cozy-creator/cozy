#!/usr/bin/env bash
# The `cli` suite: cozy as a person uses it, on a computer with no GPU, Linux, macOS or Windows. Install,
# status, the local machine, the candidates through `cozy machine update local` (cozy doctor restarts the
# installer's Calcifer on them and must find everything healthy), a weightless request through them, a real
# BiRefNet removal on the CPU from a YAML request, run list/show, a second executor (the machine restarted)
# with an --asset input, a stuck Calcifer that cozy doctor restarts, and removal as the README says.
# A broken step is recorded and the suite goes on, so one run reports all of them.
#   e2e/cli/run.sh --out DIR [--cozy-bin F] [--tensorfs-wheel F] [--runtime-wheel F] [--packages DIR]
. "$(dirname "$0")/../lib.sh"
parse "$@"

step install; install_cozy
user_services
step status; cozy --json status | tee "$OUT/status.json"
step "the local machine"; machine_ready
step candidates; candidates
step doctor; cozy --json doctor > "$OUT/doctor.json" || check "cozy doctor is not healthy: $(grep -o '"state": *"PROBLEM[^"]*"' "$OUT/doctor.json" | head -3)"

# The candidate set end to end: a request through TensorD, the Runtime and TensorFS to the reply, no model.
step "a weightless request"
text=e2e-$RANDOM$RANDOM back=''
for ((i = ${#text} - 1; i >= 0; i--)); do back+=${text:i:1}; done
if (cd "$(dirname "$0")" && cell weightless ./weightless/reverse "text=$text"); then
  grep -q "$back" "$OUT/weightless.ndjson" || check "weightless: the reply is not $back"
fi

# An idle executor killed is reaped at once (gone from ps, no zombie), and the next request works.
step "a killed idle executor"
if executor_reaped; then
  if (cd "$(dirname "$0")" && cell reaped ./weightless/reverse "text=$text"); then
    grep -q "$back" "$OUT/reaped.ndjson" || check "reaped: the reply is not $back"
  fi
fi

step "BiRefNet on the CPU, from a YAML request"
dog=$(picture dog1.jpg)
# The request file names its image relative to itself, as a person writes one beside their files.
cp "$dog" "$OUT/request-dog.jpg"
printf 'image: request-dog.jpg\n' > "$OUT/request.yaml"
if cell first "$(birefnet)" --input "$OUT/request.yaml"; then white_background "$(output first webp)" "$dog" || check "first: not a background removal"; fi
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

step "Calcifer stopped under the machine it started"
if calcifer_orphaned; then
  (cd "$(dirname "$0")" && cell after-orphan ./weightless/reverse "text=orphan$RANDOM") || check "no run went after cozy doctor recovered Calcifer"
fi

step "a stuck Calcifer"
if calcifer_stuck; then
  (cd "$(dirname "$0")" && cell after-stuck ./weightless/reverse "text=after$RANDOM") || check "no run went after cozy doctor restarted Calcifer"
fi

step "remove cozy"; remove_cozy
finish "cli: runs $first and $(cat "$OUT/second.number")"
