# e2e/lib.sh: what the e2e suites share. A suite drives `cozy` the way a person does, in the default ~/.cozy,
# on a fresh GitHub runner: the real installer, real packages and models from tensorhub.com, real outputs.
#
#   source e2e/lib.sh; parse "$@"
#   --out DIR             where records, outputs and logs go
#   --cozy-bin FILE       a candidate `cozy` (dev build): it becomes this computer's cozy and its machine
#   --tensorfs-wheel FILE a candidate TensorFS wheel for the local machine
#   --runtime-wheel FILE  a candidate cozy-runtime wheel for the local machine
#   --packages DIR        a packages checkout: its package directories run instead of the published releases

set -euo pipefail
OUT='' CAND_COZY='' CAND_TFS='' CAND_RT='' PACKAGES=''
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
      *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift 2
  done
  [ -n "$OUT" ] || { echo "--out is required" >&2; exit 2; }
  mkdir -p "$OUT"
}

step() { echo "::endgroup::"; echo "::group::$*"; }
fail() { echo "::error::$*"; echo "$*" >> "$OUT/failures.txt"; exit 1; }
cozy() { "$COZY" "$@"; }

# The install line from the README, as a person pastes it.
install_cozy() {
  if [ "$RUNNER_OS" = Windows ]; then
    powershell -NoProfile -ExecutionPolicy Bypass -c 'irm https://tensorhub.com/install.ps1 | iex' || fail "the installer failed"
  else
    curl -LsSf https://tensorhub.com/install.sh | sh || fail "the installer failed"
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

# Candidates go on through `cozy machine update local`, the path a person takes for a dev build.
candidates() {
  local args=()
  [ -z "$CAND_COZY" ] || args+=(--tensord "$CAND_COZY")
  [ -z "$CAND_TFS" ] || args+=(--tensorfs-wheel "$CAND_TFS")
  [ -z "$CAND_RT" ] || args+=(--runtime-wheel "$CAND_RT")
  [ ${#args[@]} -gt 0 ] || return 0
  cozy --json machine update local "${args[@]}" -y | tee "$OUT/update.json" || fail "cozy machine update local refused the candidates"
  if [ -n "$CAND_COZY" ]; then
    want=$("$CAND_COZY" --version)
    [ "$(cozy --version)" = "$want" ] || fail "cozy is $(cozy --version), not the candidate $want"
  fi
  machine_ready
}

# The local machine is up and reports itself; its record lands in $OUT/machine.json.
machine_ready() {
  for _ in $(seq 120); do
    cozy --json machine show local > "$OUT/machine.json" 2>/dev/null && grep -q '"state": *"ready"' "$OUT/machine.json" && return 0
    sleep 5
  done
  cat "$OUT/machine.json" || true
  fail "the local machine is not ready after 10 min"
}

# cell <name> <target> [inputs...]: one `cozy run` on local, followed to its end. Its events go to
# $OUT/<name>.ndjson, its record (`cozy run show`) to $OUT/<name>.json, its number to $OUT/<name>.number,
# its outputs under $OUT/<name>/. A run that does not complete fails the suite, unless ALLOW names a code
# its refusal carries: then $OUT/<name>.refused holds that code.
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
  fail "$name: cozy run $target did not complete (exit $code, run ${n:-none})"
}

# The first output file of a cell with this extension.
output() { find "$OUT/$1" -type f -name "*.$2" | head -1; }

# A background removal: the input's geometry, its subject kept and its background white, neither all
# of the image nor none of it.
white_background() {
  uv run -q --no-project --with pillow python - "$1" "$2" <<'PY'
import sys
from PIL import Image
image, source = Image.open(sys.argv[1]).convert("RGB"), Image.open(sys.argv[2])
assert image.size == source.size, (image.size, source.size)
white = sum(1 for p in image.getdata() if min(p) > 245) / (image.width * image.height)
print(f"white background {white:.2f}")
assert 0.1 < white < 0.9, white
PY
}

# Every input image the suites send: real photos, each run its own (BiRefNet memoizes a repeated input).
picture() {
  local file=$OUT/inputs/$1
  mkdir -p "$OUT/inputs"
  [ -f "$file" ] || curl -fsSL -o "$file" "https://raw.githubusercontent.com/pytorch/vision/v0.24.0/gallery/assets/$1"
  echo "$file"
}

# The BiRefNet target: the published release, or the packages checkout's directory.
birefnet() { if [ -n "$PACKAGES" ]; then echo "./$PACKAGES/birefnet/remove_background"; else echo birefnet/remove_background; fi; }

echo "::group::start"
