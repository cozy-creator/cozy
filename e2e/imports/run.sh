#!/usr/bin/env bash
# The `imports` suite: every first-party package still loads on the candidate set. The local machine takes
# the candidates (`cozy machine update local`), installs each root package's latest unyanked release, or the
# release --pins names (a gated candidate, published yanked; a yanked release looks the same as a retired
# one, so only a pin makes it a candidate), and a package with neither is skipped by name
# (`cozy package install`: its environment with the machine's own Runtime and TensorFS, no weights),
# and imports it there as an executor does: its entry points and every module under them. A Runtime that
# moves or drops what a published package imports fails here, before the Runtime reaches PyPI. A release
# whose Runtime floor the candidate does not meet is refused by name and skipped. With --packages, the
# checkout's package directories are installed and imported the same way.
#   e2e/imports/run.sh --out DIR [--cozy-bin F] [--tensorfs-wheel F] [--runtime-wheel F] [--packages DIR]
#                      [--pins name@release,...]
. "$(dirname "$0")/../lib.sh"
parse "$@"
PROBE=$(cd "$(dirname "$0")" && pwd)/probe.py

step install; install_cozy
user_services
step up; cozy --json up
step "the local machine"; machine_ready
step candidates; candidates

# import_release <label> <distribution> <version>: the probe in the environment holding that release.
import_release() {
  local norm env py
  norm=$(echo "$2" | tr '.-' '__')
  env=$(find "$HOME/.cozy" -type d -path '*/site-packages/*' -name "$norm-$3.dist-info" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2- | sed 's|/lib/python[^/]*/site-packages/.*||')
  py=$env/bin/python
  [ -x "$py" ] || { check "$1: no environment holds $2 $3"; return; }
  # TensorD's own bootstrap: -I -S, then only the environment's site-packages.
  "$py" -I -S -c "import pathlib,runpy,sys
root=pathlib.Path(sys.executable).absolute().parent.parent
sys.path[:0]=[str(p) for p in root.glob('lib/python*/site-packages') if p.is_dir()]
sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')" "$PROBE" "$2" > "$OUT/import-$1.log" 2>&1 \
    || check "$1: $(grep -m3 '^FAIL' "$OUT/import-$1.log" | tr '\n' ' ')"
  tail -2 "$OUT/import-$1.log"
  [ -z "$CAND_RT" ] || grep -q "^cozy-runtime $(basename "$CAND_RT" | cut -d- -f2)," "$OUT/import-$1.log" \
    || check "$1: imported with $(head -1 "$OUT/import-$1.log"), not the candidate Runtime"
}

# install <label> <target>: the environment. A release refused because it needs a newer Runtime or TensorFS
# than the candidates ships with a later set: skipped, named. One refused because the candidates are too new
# for it fails: the set cannot ship while a published package cannot run on it.
install() {
  cozy --json package install "$2" --on local -y > "$OUT/install-$1.ndjson" 2> "$OUT/install-$1.err" && return 0
  local msg want have
  msg=$(grep -ho '"message":"[^"]*"' "$OUT/install-$1.ndjson" "$OUT/install-$1.err" | head -1)
  if grep -q runtime_floor_unmet "$OUT/install-$1.ndjson" "$OUT/install-$1.err"; then
    # "… this machine runs Runtime A and TensorFS B: update it first (… --runtime C --tensorfs D)"
    have=$(sed -n 's/.*runs Runtime \([0-9.]*\) and TensorFS \([0-9.]*\).*/\1 \2/p' <<< "$msg")
    want=$(sed -n 's/.*--runtime \([0-9.]*\) --tensorfs \([0-9.]*\).*/\1 \2/p' <<< "$msg")
    if [ -n "$have" ] && [ -n "$want" ] && [ "$(printf '%s\n%s\n' "${have% *}" "${want% *}" | sort -V | head -1)" = "${have% *}" ] \
      && [ "$(printf '%s\n%s\n' "${have#* }" "${want#* }" | sort -V | head -1)" = "${have#* }" ]; then
      echo "$1: skipped, needs a newer set (Runtime/TensorFS $want) than the candidates ($have)"; return 1
    fi
  fi
  check "$1: cozy package install $2 failed: $(grep -o '"message":"[^"]*"' "$OUT/install-$1.ndjson" | head -1)$(tail -c 300 "$OUT/install-$1.err")"
  return 1
}

step "published packages"
cozy --json package search --limit 200 > "$OUT/search.json" || fail "cozy package search failed"
releases=''
for name in $(jq -r '.packages[] | select(.package | contains("/") | not) | .package' "$OUT/search.json"); do
  pin=$(tr ',' '\n' <<< "$PINS" | sed -n "s/^$name@//p" | head -1)
  latest=$(jq -r --arg n "$name" '.packages[] | select(.package == $n) | .latest_release // empty' "$OUT/search.json")
  if [ -n "$pin" ]; then releases+="$name $pin pinned"$'\n'
  elif [ -n "$latest" ]; then releases+="$name $latest release"$'\n'
  else echo "::notice::$name: no unyanked release and no pin: skipped (retired, or a candidate not pinned)"; fi
done
[ -n "$releases" ] || fail "the Hub lists no first-party package release"
while read -r name release kind; do
  [ -n "$name" ] || continue
  echo "::group::$name $release ($kind)"
  if install "$name" "$name@$release"; then import_release "$name" "$name" "$release"; fi
  echo "::endgroup::"
done <<< "$releases"

if [ -n "$PACKAGES" ]; then
  step "the checkout's packages"
  # A package is a directory whose pyproject declares a cozy.application entry point.
  for toml in $(grep -l 'cozy.application' "$PACKAGES"/*/pyproject.toml); do
    dir=$(dirname "$toml") name=$(sed -n 's/^name = "\(.*\)"/\1/p' "$toml" | head -1)
    version=$(sed -n 's/^version = "\(.*\)"/\1/p' "$toml" | head -1)
    if install "checkout-$name" "./$dir"; then import_release "checkout-$name" "$name" "$version"; fi
  done
fi

step down; cozy --json down || check "cozy down failed"
finish "imports: $(grep -c . <<< "$releases") packages"
