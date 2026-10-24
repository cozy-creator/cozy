# e2e/lib.sh: what the e2e suites share. A suite drives `cozy` the way a person does, in the default ~/.cozy,
# on a fresh GitHub runner: the real installer, real packages and models from tensorhub.com, real outputs.
#
#   source e2e/lib.sh; parse "$@"
#   --out DIR             where records, outputs and logs go
#   --cozy-bin FILE       a candidate `cozy` (dev build): it becomes this computer's cozy and its machine
#   --tensorfs-wheel FILE a candidate TensorFS wheel for the local machine
#   --runtime-wheel FILE  a candidate cozy-runtime wheel for the local machine
#   --packages DIR        a packages checkout: its package directories run instead of the published releases
#   --pins N@R,...        package releases to take instead of the latest unyanked ones (gated candidates)
#   --published           the imports suite loads published releases, not the packages checkout

set -euo pipefail
OUT='' CAND_COZY='' CAND_TFS='' CAND_RT='' PACKAGES='' PINS='' PUBLISHED=''
COZY=$HOME/.cozy/bin/cozy
[ "${RUNNER_OS:-}" != Windows ] || COZY=$HOME/.cozy/bin/cozy.exe

parse() {
  while [ $# -gt 0 ]; do
    case $1 in
      --out) OUT=$2 ;;
      --cozy-bin) CAND_COZY=$2 ;;
      --tensorfs-wheel) CAND_TFS=$2 ;;
      --runtime-wheel) CAND_RT=$2 ;;
      --packages) PACKAGES=$2 ;;
      --pins) PINS=$2 ;;
      --published) PUBLISHED=1; shift; continue ;;
      *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift 2
  done
  [ -n "$OUT" ] || { echo "--out is required" >&2; exit 2; }
  mkdir -p "$OUT"
}

step() { echo "::endgroup::"; echo "::group::$*"; }
# fail: the suite cannot go on. check: a failure the suite records and goes past, so one run reports
# every broken step; `finish` exits non-zero when any was recorded.
fail() { echo "::error::$*"; echo "$*" >> "$OUT/failures.txt"; exit 1; }
check() { echo "::error::$*"; echo "$*" >> "$OUT/failures.txt"; }
finish() {
  echo "::endgroup::"
  [ -s "$OUT/failures.txt" ] || { echo "$1: passed"; return 0; }
  echo "$1: $(wc -l < "$OUT/failures.txt") failure(s):"; cat "$OUT/failures.txt"; exit 1
}
cozy() { "$COZY" "$@"; }

# The install line from the README, as a person pastes it.
install_cozy() {
  if [ "$RUNNER_OS" = Windows ]; then
    powershell -NoProfile -ExecutionPolicy Bypass -c 'irm https://tensorhub.com/install.ps1 | iex' || fail "the installer failed"
  else
    curl -LsSf --retry 3 --retry-all-errors https://tensorhub.com/install.sh | sh || fail "the installer failed"
    found=$(env -i HOME="$HOME" PATH=/usr/bin:/bin sh -lc 'command -v cozy' || true)
    [ "$found" = "$COZY" ] || fail "a login shell finds '$found', not $COZY"
  fi
  cozy --version
}

# A user service manager, as a desktop login has: the runner's user lingers, with its bus.
user_services() {
  [ "$RUNNER_OS" = Linux ] || return 0
  sudo loginctl enable-linger "$(id -un)"
  export XDG_RUNTIME_DIR=/run/user/$(id -u)
  until [ -S "$XDG_RUNTIME_DIR/bus" ]; do sleep 1; done
  export DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus
}

# Candidates go on through `cozy machine update local`, the path a person takes for a dev build, in the
# spelling of the installed cozy (0.11.16 renamed --runtime-wheel/--tensorfs-wheel to --runtime/--tensorfs).
candidates() {
  local args=() rt=--runtime tfs=--tensorfs
  if cozy machine update --help | grep -- --runtime-wheel > /dev/null; then rt=--runtime-wheel tfs=--tensorfs-wheel; fi
  [ -z "$CAND_COZY" ] || args+=(--tensord "$CAND_COZY")
  [ -z "$CAND_TFS" ] || args+=("$tfs" "$CAND_TFS")
  [ -z "$CAND_RT" ] || args+=("$rt" "$CAND_RT")
  [ ${#args[@]} -gt 0 ] || return 0
  cozy --json machine update local "${args[@]}" -y | tee "$OUT/update.json" || fail "cozy machine update local refused the candidates"
  if [ -n "$CAND_COZY" ]; then
    want=$("$CAND_COZY" --version)
    [ "$(cozy --version)" = "$want" ] || fail "cozy is $(cozy --version), not the candidate $want"
    # The installer's Calcifer keeps running until restarted (`cozy update` does that for a person): the
    # candidate's Calcifer is the one that starts this computer's machine from here on.
    cozy --json down > "$OUT/down-candidate.json" || check "cozy down failed"
    cozy --json up > "$OUT/up-candidate.json" || fail "the candidate's cozy up failed"
    grep -q "\"version\": *\"${want#cozy }\"" "$OUT/up-candidate.json" || check "Calcifer is not the candidate ${want#cozy }"
  fi
  machine_ready
}

# Whether this computer's cozy has `cozy machine <verb>`.
has_verb() { cozy machine --help | grep -E "^ +$1 " > /dev/null; }

# The local machine ends, as its idle exit ends it: the holder of its root's lock. The next use starts it.
# A cozy before cl-646 disables it instead (machine_ready enables it again).
machine_stop() {
  if has_verb disable; then cozy --json machine disable > "$OUT/disable.json"; return; fi
  local holder pid
  holder=$(find "$HOME/.cozy/machine" -name agent.lock.holder 2> /dev/null | head -1)
  find "$HOME/.cozy/machine" -name 'agent.lock*' > "$OUT/locks.txt" 2>&1 || true
  [ -n "$holder" ] || return 1
  cp "$holder" "$OUT/agent.lock.holder"
  pid=$(grep -o '"pid": *[0-9]*' "$holder" | grep -o '[0-9]*$') || return 1
  if [ "$RUNNER_OS" = Windows ]; then taskkill //F //T //PID "$pid"; else kill "$pid"; fi
  # Gone, or a zombie its parent has not reaped yet (kill -0 still finds one).
  for _ in $(seq 60); do
    if [ "$RUNNER_OS" = Windows ]; then kill -0 "$pid" 2> /dev/null || return 0
    else case $(ps -o stat= -p "$pid" 2> /dev/null || true) in '' | Z*) return 0 ;; esac; fi
    sleep 1
  done
  ps -o pid,stat,args -p "$pid" > "$OUT/stop.txt" 2>&1 || true
  return 1
}

# The process holding the local machine's root (on Windows its service, whose parent is TensorD's own).
machine_pid() { grep -o '"pid": *[0-9]*' "$HOME/.cozy/machine/var/lib/cozy/machine/agent.lock.holder" | grep -o '[0-9]*$'; }
parent_of() {
  if [ "$RUNNER_OS" = Windows ]; then
    powershell -NoProfile -c "(Get-CimInstance Win32_Process -Filter 'ProcessId=$1').ParentProcessId" | tr -d '\r '
  else
    ps -o ppid= -p "$1" | tr -d ' '
  fi
}
# Whether process $1 is gone, a zombie not: one answers kill -0 until its parent reaps it.
gone() {
  if [ "$RUNNER_OS" = Windows ]; then ! tasklist //FI "PID eq $1" | grep -q " $1 "; else ! kill -0 "$1" 2>/dev/null; fi
}

# cl-646: the local machine Calcifer started, ended, is reaped by Calcifer (gone, no zombie), and the next
# `machine show local` starts another. One the installer's `cozy-machine` service still runs is ended
# first, until the machine is the one Calcifer started.
machine_reaped() {
  local calcifer pid top='' i
  calcifer=$(cozy --json status | grep -o '"pid": *[0-9]*' | head -1 | grep -o '[0-9]*$')
  for i in 1 2 3; do
    machine_ready
    pid=$(machine_pid) || { check "no process holds the local machine's root"; return; }
    top=$pid
    [ "$RUNNER_OS" != Windows ] || top=$(parent_of "$pid")
    [ "$(parent_of "$top")" = "$calcifer" ] && break
    echo "the local machine (pid $top) is $(parent_of "$top")'s, not Calcifer's ($calcifer): ending it"
    machine_stop || true
  done
  [ "$(parent_of "$top")" = "$calcifer" ] || { check "Calcifer (pid $calcifer) never started the local machine"; return; }
  if [ "$RUNNER_OS" = Windows ]; then taskkill //F //T //PID "$top"; else kill "$top"; fi
  for i in $(seq 30); do gone "$top" && break; sleep 1; done
  gone "$top" || check "Calcifer left the local machine it started (pid $top) behind: state $(ps -o stat= -p "$top" 2>/dev/null)"
  machine_ready
  [ "$(machine_pid)" != "$pid" ] || check "the next machine show local did not start another machine"
  echo "the local machine Calcifer started (pid $top) was reaped; the next one is pid $(machine_pid)"
}

# cl-646: a Calcifer that stops (a service manager's SIGTERM, or killed) leaves the local machine it started
# running, and the next one adopts it at once: the next `cozy` command answers within seconds with the same
# machine.
calcifer_restart_adopts() {
  local pid how calcifer began took
  machine_ready
  pid=$(machine_pid) || { check "no process holds the local machine's root"; return; }
  for how in term kill; do
    [ $how = kill ] || [ "$RUNNER_OS" != Windows ] || continue
    calcifer=$(cozy --json status | grep -o '"pid": *[0-9]*' | head -1 | grep -o '[0-9]*$')
    case $how in
      term) kill -TERM "$calcifer" ;;
      kill) if [ "$RUNNER_OS" = Windows ]; then taskkill //F //PID "$calcifer"; else kill -9 "$calcifer"; fi ;;
    esac
    for _ in $(seq 30); do gone "$calcifer" && break; sleep 1; done
    began=$(date +%s)
    cozy --json machine show local > "$OUT/adopted-$how.json" 2>&1 || check "after $how: cozy machine show local failed"
    took=$(($(date +%s) - began))
    [ "$took" -le 30 ] || check "after $how: the next cozy command took ${took}s"
    [ "$(machine_pid)" = "$pid" ] || check "after $how: Calcifer started another machine, not adopting pid $pid"
    echo "after $how: the next Calcifer answered in ${took}s with the same machine (pid $pid)"
  done
}

# cl-646: an idle executor that is killed leaves TensorD's pool and is reaped at once: gone from ps within a
# second, never left a zombie until the next call. Linux and macOS (Windows has no zombies).
executor_reaped() {
  [ "$RUNNER_OS" != Windows ] || return 0
  local executor
  executor=$(pgrep -n -f "cozy_runtime.internal.executor")
  [ -n "$executor" ] || { check "no idle executor to end"; return 1; }
  kill -9 "$executor"
  sleep 1
  if ps -o stat= -p "$executor" > /dev/null 2>&1; then
    check "the killed idle executor (pid $executor) is still in ps a second later: $(ps -o stat= -p "$executor")"
    return 1
  fi
  echo "the killed idle executor (pid $executor) was gone from ps within a second"
}

# The local machine is up and reports itself (`machine show local` starts it); its record lands in
# $OUT/machine.json. A cozy before cl-646 (#1883; the installer's release until one carries it) makes its
# machine with `cozy machine enable`; a later one starts it on first use.
machine_ready() {
  if has_verb enable; then cozy --json machine enable > "$OUT/enable.json" || check "cozy machine enable failed"; fi
  for _ in $(seq 120); do
    cozy --json machine show local > "$OUT/machine.json" 2>/dev/null && grep -q '"state": *"ready"' "$OUT/machine.json" && return 0
    sleep 5
  done
  cat "$OUT/machine.json" || true
  fail "the local machine is not ready after 10 min"
}

# cell <name> <target> [inputs...]: one `cozy run` on local, followed to its end. Its events go to
# $OUT/<name>.ndjson, its record (`cozy run show`) to $OUT/<name>.json, its number to $OUT/<name>.number,
# its outputs under $OUT/<name>/. A run that does not complete is a recorded failure (`check`), unless
# ALLOW names a code its refusal carries: then $OUT/<name>.refused holds that code. Returns 1 when the
# run neither completed nor was refused as allowed.
cell() {
  local name=$1 target=$2 code=0 n
  shift 2
  cozy --json run "$target" "$@" --on local --out "$OUT/$name" > "$OUT/$name.ndjson" 2> "$OUT/$name.err" || code=$?
  n=$(grep -o '"number":[0-9]*' "$OUT/$name.ndjson" | head -1 | cut -d: -f2 || true)
  echo "${n:-none}" > "$OUT/$name.number"
  [ -z "$n" ] || cozy --json run show "$n" > "$OUT/$name.json" 2>> "$OUT/$name.err" || true
  if [ "$code" = 0 ] && grep -q '"type":"run.completed"' "$OUT/$name.ndjson"; then
    echo "$name: run $n completed"
    return 0
  fi
  tail -c 3000 "$OUT/$name.ndjson" || true
  tail -c 3000 "$OUT/$name.err" || true
  if [ -n "${ALLOW:-}" ] && grep -q "$ALLOW" "$OUT/$name.ndjson" "$OUT/$name.err"; then
    echo "$ALLOW" > "$OUT/$name.refused"
    echo "$name: run ${n:-none} refused with $ALLOW"
    return 0
  fi
  check "$name: cozy run $target did not complete (exit $code, run ${n:-none}): $(grep -o '"message":"[^"]*"' "$OUT/$name.ndjson" | head -1)"
  return 1
}

# The first output file of a cell with this extension.
output() { find "$OUT/$1" -type f -name "*.$2" | head -1; }

# A background removal: the input's geometry, its background white, and its subject kept (a small subject is
# mostly white; an all-white image lost it).
white_background() {
  uv run -q --no-project --with pillow python - "$1" "$2" <<'PY'
import sys
from PIL import Image
image, source = Image.open(sys.argv[1]).convert("RGB"), Image.open(sys.argv[2])
assert image.size == source.size, (image.size, source.size)
white = sum(1 for p in image.getdata() if min(p) > 245) / (image.width * image.height)
print(f"white background {white:.2f}")
assert 0.1 < white < 0.98, white
PY
}

# Every input image the suites send: real photos, each run its own (BiRefNet memoizes a repeated input).
picture() {
  local file=$OUT/inputs/$1
  mkdir -p "$OUT/inputs"
  [ -f "$file" ] || curl -fsSL -o "$file" "https://raw.githubusercontent.com/pytorch/vision/v0.24.0/gallery/assets/$1"
  echo "$file"
}

# The BiRefNet target: a pinned release, the packages checkout's directory, or the published release.
birefnet() {
  local pin
  pin=$(tr ',' '\n' <<< "$PINS" | sed -n 's/^birefnet@//p' | head -1)
  if [ -n "$pin" ]; then echo "birefnet/remove_background@$pin"
  elif [ -n "$PACKAGES" ]; then echo "./$PACKAGES/birefnet/remove_background"
  else echo birefnet/remove_background; fi
}

echo "::group::start"
