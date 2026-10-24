#!/usr/bin/env bash
# The `imports` suite: every first-party package still loads on the candidate set. The local machine takes
# the candidates (`cozy machine update local`) and installs each package (`cozy package install`: its
# environment with the machine's own Runtime and TensorFS, no weights): on a PR, every package directory of
# the packages checkout (packages master, the RC's package source, or the change's own); with --published or
# --pins, each root package's latest unyanked release or the pinned one (a gated candidate is published
# yanked and looks like a retired release, so only a pin makes it a candidate; one with neither is skipped),
# and imports it there as an executor does: its entry points and every module under them. A Runtime that
# moves or drops what a published package imports fails here, before the Runtime reaches PyPI. A release
# whose Runtime floor the candidate does not meet is refused by name and skipped. With --packages, the
# checkout's package directories are installed and imported the same way.
#   e2e/imports/run.sh --out DIR [--cozy-bin F] [--tensorfs-wheel F] [--runtime-wheel F] [--packages DIR]
#                      [--published] [--pins name@release,...]
. "$(dirname "$0")/../lib.sh"
parse "$@"
PROBE=$(cd "$(dirname "$0")" && pwd)/probe.py

step install; install_cozy
user_services
step doctor; cozy --json doctor > "$OUT/doctor.json" || check "cozy doctor is not healthy"
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
  # The environment's own interpreter, isolated (-I) but with its site: a checkout installs editable, through
  # a .pth that -S would skip.
  "$py" -I "$PROBE" "$2" > "$OUT/import-$1.log" 2>&1 \
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
  msg=$(grep -ho '"message": *"[^"]*"' "$OUT/install-$1.ndjson" "$OUT/install-$1.err" | head -1)
  # A checkout whose dependencies are Hub-hosted packages needs a signed-in computer for the Hub's index;
  # these runners are anonymous (the signed-in release gate covers it).
  if grep -q '"not_logged_in"' "$OUT/install-$1.ndjson" "$OUT/install-$1.err"; then
    echo "::notice::$1: skipped, its dependencies come from the Hub's index and this runner is not signed in"; return 1
  fi
  if grep -q runtime_floor_unmet "$OUT/install-$1.ndjson" "$OUT/install-$1.err"; then
    # "… this machine runs Runtime A and TensorFS B: update it first (… --runtime C --tensorfs D)"
    have=$(sed -n 's/.*runs Runtime \([0-9.]*\) and TensorFS \([0-9.]*\).*/\1 \2/p' <<< "$msg")
    want=$(sed -n 's/.*--runtime \([0-9.]*\) --tensorfs \([0-9.]*\).*/\1 \2/p' <<< "$msg")
    if [ -n "$have" ] && [ -n "$want" ] && [ "$(printf '%s\n%s\n' "${have% *}" "${want% *}" | sort -V | head -1)" = "${have% *}" ] \
      && [ "$(printf '%s\n%s\n' "${have#* }" "${want#* }" | sort -V | head -1)" = "${have#* }" ]; then
      echo "$1: skipped, needs a newer set (Runtime/TensorFS $want) than the candidates ($have)"; return 1
    fi
  fi
  check "$1: cozy package install $2 failed: $msg$(tail -c 300 "$OUT/install-$1.err")"
  return 1
}

count=0
if [ -n "$PUBLISHED" ] || [ -n "$PINS" ]; then
  # A release's question: the published releases (or the pinned candidates) on the candidate set.
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
    count=$((count + 1))
  done <<< "$releases"
else
  # A PR's question: the RC's own packages (packages master, or the change's checkout) on the candidate set.
  [ -n "$PACKAGES" ] || fail "no packages checkout and no --published/--pins: nothing to import"
  step "the checkout's packages"
  # A package is a directory whose pyproject declares a cozy.application entry point.
  for toml in $(grep -l 'cozy.application' "$PACKAGES"/*/pyproject.toml); do
    dir=$(dirname "$toml") name=$(sed -n 's/^name = "\(.*\)"/\1/p' "$toml" | head -1)
    version=$(sed -n 's/^version = "\(.*\)"/\1/p' "$toml" | head -1)
    echo "::group::$name $version (checkout)"
    if install "checkout-$name" "./$dir"; then import_release "checkout-$name" "$name" "$version"; fi
    echo "::endgroup::"
    count=$((count + 1))
  done
fi

finish "imports: $count packages"
