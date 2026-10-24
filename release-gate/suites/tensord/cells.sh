# Suite `tensord`: the machine agent through `cozy`, on one rental (2 GPUs for placement). before.sh (cl-650) moved the
# fresh pod through TensorD 0.11.15 to the candidate and met a release's Runtime floor. Each check below is an `expect`
# row, PASS or FAIL with what happened, which analyze.py fails the gate on. Every request has its own prompt and seed.
T=$OUT/tensord; mkdir -p "$T"
show() { timeout 120 "$CLI" run show "$1" --json 2>/dev/null > "$T/$1.json"; }
# get <run> <expr>: a field of the run's record (status, number, attempt, executor, gpus, length, ms).
get() {
  show "$1"; python3 - "$T/$1.json" "$2" <<'PY'
import json, sys
try: r = json.load(open(sys.argv[1]))
except Exception: sys.exit()
m = r.get('measurements') or {}
def walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values(): yield from walk(v)
    elif isinstance(o, list):
        for v in o: yield from walk(v)
f = {'status': lambda: r.get('status'), 'reason': lambda: (r.get('reason') or {}).get('code', ''),
     'attempt': lambda: (m.get('software') or {}).get('attempt_ordinal'),
     'executor': lambda: ((m.get('execution') or {}).get('executor') or {}).get('started_unix_ms'),
     'gpus': lambda: ','.join(sorted({str(d['gpu']) for d in walk(m) if isinstance(d.get('gpu'), int) and d['gpu'] >= 0})),
     'window': lambda: f"{r.get('accepted_at_ms') or 0} {r.get('settled_at_ms') or 0}",
     'ms': lambda: r.get('execution_ms') or ''}[sys.argv[2]]
print(f() if f() is not None else '')
PY
}
# submit <target> [inputs]: a detached run's number.
submit() { "$CLI" run "$@" --on "$M" --detach --json 2>>"$T/submit.err" | grep -o '"number":[0-9]*' | head -1 | cut -d: -f2; }
until_status() {  # until_status <run> <status|terminal>: waits while the run moves (cozy run watch for the end)
  if [ "$2" = terminal ]; then timeout 3600 "$CLI" run watch "$1" --json > "$T/$1.watch" 2>&1; return; fi
  local i; for i in $(seq 300); do [ "$(get "$1" status)" = "$2" ] && return 0; case $(get "$1" status) in completed|failed|canceled) return 1;; esac; sleep 2; done; return 1
}
envs() { timeout 60 "$CLI" --json machine show "$M" 2>/dev/null | python3 -c 'import json,sys
print(" ".join(sorted(e["package"].split("/")[-1] + "@" + e["release"] for e in json.load(sys.stdin)["status"]["environments"])))'; }
released() {  # released <package>: its two newest releases users get (not yanked), newest first
  timeout 60 "$CLI" package info "$1" --json 2>/dev/null | python3 -c 'import json,sys
print(" ".join([r["release"] for r in json.load(sys.stdin)["releases"] if not r.get("yanked")][:2]))'; }
NGPU=$(timeout 60 "$CLI" --json machine show "$M" 2>/dev/null | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["status"]["gpus"]))')

# Generations: install, update and roll back a package; then the Runtime back a release and forward to the candidate.
read -r first second <<< "$(released sdxl)"
NEW=$(tr , '\n' <<< "$PINS" | sed -n 's/^sdxl@//p' | head -1); NEW=${NEW:-$first}
OLD=$first; [ "$OLD" != "$NEW" ] || OLD=$second
"$CLI" package install "sdxl@$OLD" --on "$M" -y > "$T/install.log" 2>&1; a=$(envs)
"$CLI" package update "sdxl@$NEW" --on "$M" -y > "$T/update.log" 2>&1; b=$(envs)
n1=$(submit "sdxl/generate@$NEW" prompt="$SCENE, generation $NEW" seed:=$((R + 30))); until_status "$n1" terminal
"$CLI" package update "sdxl@$OLD" --on "$M" -y > "$T/rollback.log" 2>&1; c=$(envs)
n2=$(submit "sdxl/generate@$OLD" prompt="$OBJECT, generation $OLD" seed:=$((R + 31))); until_status "$n2" terminal
if [[ " $a " == *" sdxl@$OLD "* && " $b " == *" sdxl@$NEW "* && " $c " == *" sdxl@$OLD "* ]] \
   && [ "$(get "$n1" status)" = completed ] && [ "$(get "$n2" status)" = completed ]; then
  expect tensord-generations PASS "sdxl $OLD installed, updated to $NEW (run $n1), rolled back to $OLD (run $n2)"
else
  expect tensord-generations FAIL "installed [$a] updated [$b] rolled back [$c]; run $n1 $(get "$n1" status), run $n2 $(get "$n2" status)"
fi
if [ -n "$RWF$TWF" ]; then
  "$CLI" machine update "$M" --runtime 0.31.0 --tensorfs 0.14.0 -y > "$T/runtime-back.log" 2>&1; back=$?
  n=$(submit "sdxl/generate@$NEW" prompt="$CHAR, the Runtime back" seed:=$((R + 32))); until_status "$n" terminal; s1=$(get "$n" status)
  "$CLI" machine update "$M" ${RWF:+--runtime "$RWF"} ${TWF:+--tensorfs "$TWF"} -y > "$T/runtime-forward.log" 2>&1; fwd=$?
  m=$(submit "sdxl/generate@$NEW" prompt="$CHAR, the Runtime forward" seed:=$((R + 33))); until_status "$m" terminal; s2=$(get "$m" status)
  [ "$back$fwd" = 00 ] && [ "$s1$s2" = completedcompleted ] \
    && expect tensord-runtime-rollback PASS "Runtime back to 0.31.0 (run $n) and forward to the candidate (run $m)" \
    || expect tensord-runtime-rollback FAIL "back exit $back run $n $s1; forward exit $fwd run $m $s2"
fi

# Run lifecycle: a second run queues behind the first, a queued run cancels without running, a running one cancels,
# and a completed run's outputs land here whole.
a=$(submit "sdxl/generate@$NEW" prompt="$SCENE, the long run" steps:=60 seed:=$((R + 34)))
b=$(submit "sdxl/generate@$NEW" prompt="$OBJECT, the queued run" steps:=60 seed:=$((R + 35)))
until_status "$a" running; qb=$(get "$b" status)
"$CLI" run cancel "$b" -y > "$T/cancel-queued.log" 2>&1; until_status "$b" terminal
"$CLI" run cancel "$a" -y > "$T/cancel-running.log" 2>&1; until_status "$a" terminal
c=$(submit "sdxl/generate@$NEW" prompt="$CHAR, the landed run" seed:=$((R + 36))); until_status "$c" terminal
f=$(ls ~/.cozy/outputs/*/"$c"-* 2>/dev/null | grep -v '\.part$' | head -1)
want=$(grep -ho '"length":[0-9]*' "$T/$c.watch" | tail -1 | cut -d: -f2)
if [ "$qb" = queued ] && [ "$(get "$b" status)" = canceled ] && [ -z "$(get "$b" ms)" ] && [ "$(get "$a" status)" = canceled ] \
   && [ "$(get "$c" status)" = completed ] && [ -n "$f" ] && [ "$(stat -c %s "$f")" = "$want" ]; then
  expect tensord-run-lifecycle PASS "run $b queued behind $a and canceled unrun; $a canceled running; $c's $want bytes landed"
else
  ran=$(get "$b" ms)
  expect tensord-run-lifecycle FAIL "run $b was ${qb:-?} then $(get "$b" status) (ran ${ran:-0} ms); $a $(get "$a" status); $c $(get "$c" status), landed ${f:-nothing} ($(stat -c %s "$f" 2>/dev/null || echo 0) of ${want:-?} bytes)"
fi

# Executors and leases: models stack on the pod and idle executors end when the last one needs the card (cr-216's
# describer); the canceled run's executor gave its memory back; a restart gives every run a new executor.
s=$(submit "$(pkg anima generate)" prompt="masterpiece, anime illustration of $CHAR, stacked" seed:=$((R + 37))); until_status "$s" terminal
k=$(submit "$(pkg flux-3-klein generate_turbo)" prompt="$OBJECT, stacked" seed:=$((R + 38))); until_status "$k" terminal
d=$(submit "$(pkg qwen3.5 ask)" prompt="Describe this image in one sentence." --asset images="$REF"); until_status "$d" terminal
ended=$(timeout 60 "$CLI" machine logs "$M" -n 2000 2>/dev/null | grep -c 'ending idle')
[ "$(get "$s" status)$(get "$k" status)$(get "$d" status)" = completedcompletedcompleted ] \
  && expect tensord-leases PASS "SDXL, Anima and Klein stacked, then the describer fit (run $d); $ended idle executors ended" \
  || expect tensord-leases FAIL "anima $s $(get "$s" status), klein $k $(get "$k" status), describer $d $(get "$d" status) $(get "$d" reason)"
"$CLI" machine update "$M" --gpu-memory-limit off -y > "$T/restart.log" 2>&1; at=$(( $(date +%s) * 1000 ))
r=$(submit "sdxl/generate@$NEW" prompt="$SCENE, after the restart" seed:=$((R + 39))); until_status "$r" terminal
[ "$(get "$r" status)" = completed ] && [ "$(get "$r" executor)" -gt "$at" ] 2>/dev/null \
  && expect tensord-executor-restart PASS "after the machine restart, run $r on a new executor" \
  || expect tensord-executor-restart FAIL "run $r $(get "$r" status), executor started $(get "$r" executor) (restart at $at)"

# No re-execution: the machine restarts while a run is in flight; the run ends once, as completed or failed with a
# reason, and is never attempted again.
x=$(submit "sdxl/generate@$NEW" prompt="$OBJECT, in flight" steps:=60 seed:=$((R + 40))); until_status "$x" running
"$CLI" machine update "$M" --gpu-memory-limit off -y > "$T/restart-inflight.log" 2>&1; until_status "$x" terminal
case "$(get "$x" status):$(get "$x" attempt)" in
  completed:1) expect tensord-no-reexecution PASS "run $x completed once (the restart waited for it: a mid-run TensorD kill is not reachable through cozy)";;
  failed:1) expect tensord-no-reexecution PASS "run $x failed once, $(get "$x" reason), not executed again";;
  *) expect tensord-no-reexecution FAIL "run $x $(get "$x" status), attempt $(get "$x" attempt)";;
esac

# Multi-GPU placement: two runs at once go to two GPUs.
if [ "${NGPU:-1}" -ge 2 ]; then
  p=$(submit "sdxl/generate@$NEW" prompt="$SCENE, GPU one" steps:=40 seed:=$((R + 41)))
  q=$(submit "$(pkg anima generate)" prompt="masterpiece, anime illustration of $OBJECT, GPU two" seed:=$((R + 42)))
  until_status "$p" terminal; until_status "$q" terminal
  read -r p0 p1 <<< "$(get "$p" window)"; read -r q0 q1 <<< "$(get "$q" window)"
  [ "$(get "$p" status)$(get "$q" status)" = completedcompleted ] && [ "$(get "$p" gpus)" != "$(get "$q" gpus)" ] && [ "$p0" -lt "$q1" ] && [ "$q0" -lt "$p1" ] \
    && expect tensord-placement PASS "runs $p and $q at once on GPUs $(get "$p" gpus) and $(get "$q" gpus)" \
    || expect tensord-placement FAIL "run $p $(get "$p" status) GPUs $(get "$p" gpus) [$p0,$p1]; run $q $(get "$q" status) GPUs $(get "$q" gpus) [$q0,$q1]"
else
  log "tensord: placement not run, $M has ${NGPU:-1} GPU"
fi

# Output delivery: an output's landing against this computer's own download speed (5.9 MB took 4 min on 10-23).
link=$(curl -sfL -o /dev/null -w '%{speed_download}' "https://github.com/cozy-creator/cozy-creator/releases/download/v0.11.15/cozy-x86_64-unknown-linux-gnu.tar.gz")
"$CLI" run "sdxl/generate@$NEW" --on "$M" --json prompt="$SCENE, a large render" megapixels:=2 seed:=$((R + 43)) 2>/dev/null \
  | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done > "$T/delivery.ndjson"; end=$(date +%s.%N)
read -r done_ bytes <<< "$(grep '"type":"output_item.done"' "$T/delivery.ndjson" | tail -1 | python3 -c 'import json,sys
t, j = sys.stdin.read().split(" ", 1); print(t, json.loads(j)["item"]["length"])' 2>/dev/null)"
python3 - "${done_:-0}" "$end" "${bytes:-0}" "${link:-0}" <<'PY' > "$T/delivery.txt"
import sys
done, end, size, link = map(float, sys.argv[1:])
took = end - done; expect = size / link if link else 0
print('PASS' if size and took <= 10 * expect + 5 else 'FAIL', f'{size / 1e6:.1f} MB landed in {took:.1f} s; this computer downloads {link / 1e6:.1f} MB/s, so {expect:.1f} s (bar: 10x + 5 s)')
PY
read -r v rest < "$T/delivery.txt"; expect tensord-output-delivery "$v" "$rest"
