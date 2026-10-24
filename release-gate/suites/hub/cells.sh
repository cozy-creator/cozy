# Suite `hub` (gate.sh --suite hub): the Hub's API contract (tensorhub api/openapi.yaml at $HUB_REF) as cozy uses
# it, on the rented machine $M and a throwaway <account>/ci-hub-<R> package and model that the suite retires at the
# end (the package's releases yanked, the model deleted):
#   paths            contract.py: each path no cozy verb reaches, called anonymously, answers as its contract says
#   account          whoami, credits, credits history, the rental catalog, the rental $M itself
#   providers        an admin rental of the cheapest CPU SKU on each provider (RunPod, vast), ended and checked
#                    gone, so both acquisition paths are covered (renters never pick one: th-254)
#   package          lock and publish 0.1.0 and 0.1.1; the unversioned run takes 0.1.1; yank 0.1.1 → the simple index marks it
#                    data-yanked, a fresh install takes 0.1.0, its exact pin still installs; un-yank → 0.1.1 again
#   model            upload, publish 0.1.0 (listed), download; yank (listed yanked), the exact pin downloads; un-yank;
#                    delete → gone
# Each is an `expect` row: a FAIL fails the gate. The migration rehearsal on a prod dump is the release agent's
# step before a Hub deploy (release-gate README).
HB=$OUT/suite-hub HUB_URL=${HUB_URL:-https://tensorhub.com} HUB_REF=${HUB_REF:-master} HUB_PKG= HUB_MODEL= HUB_LIVE= HUB_RENTED=
mkdir -p "$HB"
# hub_do <check> ok|refused <cozy args...>: PASS (returns 0) when the command exits as wanted; output in <check>.out.
hub_do() {
  local check=$1 want=$2 code; shift 2
  "$CLI" --json "$@" >"$HB/$check.out" 2>&1; code=$?
  if { [ "$want" = ok ] && [ "$code" = 0 ]; } || { [ "$want" = refused ] && [ "$code" != 0 ]; }; then
    expect "hub $check" PASS "cozy $* → exit $code"
  else
    expect "hub $check" FAIL "cozy $* → exit $code (wanted $want): $(tail -c 300 "$HB/$check.out")"
    return 1
  fi
}
# hub_runs <check> <release> <package> [pin]: `cozy run <package>/echo[@pin]` on $M answers with that release.
hub_runs() {
  local check=$1 want=$2 n got
  "$CLI" --json run "$3/echo${4:+@$4}" text=hub --on "$M" >"$HB/$check.ndjson" 2>"$HB/$check.err"
  n=$(grep -o '"number":[0-9]*' "$HB/$check.ndjson" | head -1 | cut -d: -f2)
  got=$([ -n "$n" ] && timeout 120 "$CLI" --json run show "$n" 2>/dev/null | python3 -c 'import json,sys; print((json.load(sys.stdin).get("result") or {}).get("release", ""))')
  [ "$got" = "$want" ] && expect "hub $check" PASS "run $n of $3${4:+@$4} served $got" ||
    expect "hub $check" FAIL "run ${n:-none} of $3${4:+@$4} served '${got}', not $want: $(tail -c 300 "$HB/$check.err")"
}
# hub_release <version>: the throwaway package at that version, published from its own directory.
hub_release() {
  local dir=$HB/pkg-$1
  rm -rf "$dir" && cp -r "$HERE/suites/hub/pkg" "$dir"
  sed -i "s/^name = \"ci-hub-pkg\"/name = \"$HUB_PKG\"/; s/^version = \"0.1.0\"/version = \"$1\"/" "$dir/pyproject.toml"
  sed -i "s/^RELEASE = \"0.1.0\"/RELEASE = \"$1\"/" "$dir/ci_hub_pkg/__init__.py"
  (cd "$dir" && hub_do "package-lock-$1" ok package lock && hub_do "package-publish-$1" ok package publish -y)
}
# hub_listed <check> <yanked: true|false>: `cozy model info` lists the model's release 0.1.0, yanked as given.
hub_listed() {
  "$CLI" --json model info "$HUB_MODEL" >"$HB/$1.out" 2>&1
  python3 -I -c 'import json,sys; r=[x for x in json.load(open(sys.argv[1])).get("releases") or [] if x.get("release")=="0.1.0"]; sys.exit(0 if r and str(r[0].get("yanked")).lower()==sys.argv[2] else 1)' "$HB/$1.out" "$2" &&
    expect "hub $1" PASS "model info lists 0.1.0, yanked $2" ||
    expect "hub $1" FAIL "model info does not list 0.1.0 with yanked $2: $(tail -c 300 "$HB/$1.out")"
}
hub_cleanup() {
  local v
  for v in $HUB_LIVE; do "$CLI" package yank "$WHO/$HUB_PKG@$v" -y >>"$HB/cleanup.log" 2>&1; done
  [ -z "$HUB_MODEL" ] || "$CLI" model delete "$HUB_MODEL" -y >>"$HB/cleanup.log" 2>&1
  for v in $HUB_RENTED; do "$CLI" machine end "$v" -y >>"$HB/cleanup.log" 2>&1; done
  HUB_LIVE= HUB_MODEL= HUB_RENTED=
}
trap 'hub_cleanup; declare -F tfs_cleanup >/dev/null && tfs_cleanup; end_it' EXIT

# paths: the contract document of the Hub being released.
if gh api -H 'Accept: application/vnd.github.raw' "repos/cozy-creator/tensorhub-v2/contents/api/openapi.yaml?ref=$HUB_REF" >"$HB/openapi.yaml" 2>"$HB/openapi.err"; then
  while IFS=$'\t' read -r check verdict what; do expect "$check" "$verdict" "$what"; done < <(uv run -q --no-project \
    --with pyyaml --with jsonschema python -I "$HERE/suites/hub/contract.py" "$HUB_URL" "$HB/openapi.yaml" paths)
else
  expect "hub contract document" FAIL "tensorhub-v2 api/openapi.yaml@$HUB_REF unreadable: $(tail -c 200 "$HB/openapi.err")"
fi
# account
WHO=; for _ in 1 2 3; do  # a dropped connection is asked again
  WHO=$("$CLI" --json whoami 2>"$HB/whoami.err" | python3 -c 'import json,sys; print(json.load(sys.stdin)["account"])' 2>/dev/null) && [ -n "$WHO" ] && break
  sleep 5
done
[ -n "$WHO" ] && expect "hub whoami" PASS "signed in as $WHO" || expect "hub whoami" FAIL "$(tail -c 200 "$HB/whoami.err")"
hub_do credits ok credits
hub_do credits-history ok credits history
hub_do catalog ok machine rent
# The rental itself is gate.sh's (rent, then end and check it is gone).
P=$("$CLI" --json machine show "$M" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["rental"]["sku"])' 2>/dev/null)
[ -n "$P" ] && expect "hub rental" PASS "$M is a $P rental" || expect "hub rental" FAIL "machine show $M names no rental"
# hub_machines: the names `cozy machine list` shows, one per line.
hub_machines() {
  "$CLI" --json machine list 2>/dev/null | python3 -c 'import json,sys
def walk(o):
    if isinstance(o, dict):
        if isinstance(o.get("name"), str) and ("rental" in o or "kind" in o): print(o["name"])
        for v in o.values(): walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
walk(json.load(sys.stdin))' 2>/dev/null | sort -u
}
# hub_rent_on <provider>: an admin rental of `cpu` there (`machine rent --provider`, admin-only), ended and checked
# gone. A rental it cannot name from the reply is the one new in `machine list`; hub_cleanup ends any left.
hub_rent_on() {
  local m before
  before=$(hub_machines)
  "$CLI" --json machine rent cpu --provider "$1" -y >"$HB/rent-$1.out" 2>&1
  m=$(python3 -c 'import json,sys
t = open(sys.argv[1]).read(); dec = json.JSONDecoder(); i = 0
while (i := t.find("{", i)) >= 0:
    d, i = dec.raw_decode(t, i)
    if isinstance(d.get("machine"), dict) and d["machine"].get("name"): print(d["machine"]["name"]); break' "$HB/rent-$1.out" 2>/dev/null)
  [ -n "$m" ] || m=$(comm -13 <(echo "$before") <(hub_machines) | grep -vxF "$M" | head -1)
  [ -n "$m" ] || { expect "hub rent on $1" FAIL "$(tail -c 300 "$HB/rent-$1.out")"; return; }
  HUB_RENTED="$HUB_RENTED $m"
  grep -q '"machine"' "$HB/rent-$1.out" && expect "hub rent on $1" PASS "rented $m on $1" ||
    expect "hub rent on $1" FAIL "rented $m on $1, but the reply named no machine: $(tail -c 200 "$HB/rent-$1.out")"
  "$CLI" machine end "$m" -y >"$HB/end-$1.out" 2>&1
  for _ in $(seq 30); do
    hub_machines | grep -qxF "$m" || { expect "hub end on $1" PASS "$m ended and gone"; HUB_RENTED=${HUB_RENTED/ $m/}; return; }
    sleep 10
  done
  expect "hub end on $1" FAIL "$m still listed 5 min after cozy machine end: $(tail -c 200 "$HB/end-$1.out")"
}
if "$CLI" machine rent --help 2>/dev/null | grep -q -- --provider || "$CLI" machine rent cpu --provider runpod --help >/dev/null 2>&1; then
  for p in runpod vast; do hub_rent_on "$p"; done
else
  expect "hub providers" FAIL "this cozy has no admin \`machine rent --provider\`: it cannot rent on each provider"
fi
[ -n "$WHO" ] || return 0
HUB_PKG=ci-hub-$R
# package
hub_release 0.1.0 && HUB_LIVE="0.1.0"
hub_release 0.1.1 && HUB_LIVE="0.1.0 0.1.1"
hub_do package-info ok package info "$WHO/$HUB_PKG"
hub_runs package-newest 0.1.1 "$WHO/$HUB_PKG"
hub_do package-yank ok package yank "$WHO/$HUB_PKG@0.1.1"
while IFS=$'\t' read -r check verdict what; do expect "$check" "$verdict" "$what"; done < <(uv run -q --no-project \
  --with pyyaml --with jsonschema python -I "$HERE/suites/hub/contract.py" "$HUB_URL" "$HB/openapi.yaml" yanked "$WHO" "$HUB_PKG" 0.1.1)
hub_do package-remove ok package remove "$WHO/$HUB_PKG" --on "$M"
hub_runs package-skips-yanked 0.1.0 "$WHO/$HUB_PKG"
hub_runs package-pinned-yanked 0.1.1 "$WHO/$HUB_PKG" 0.1.1
if "$CLI" package yank --help 2>/dev/null | grep -q -- --undo; then
  hub_do package-unyank ok package yank --undo "$WHO/$HUB_PKG@0.1.1"
  hub_do package-remove-again ok package remove "$WHO/$HUB_PKG" --on "$M"
  hub_runs package-unyanked 0.1.1 "$WHO/$HUB_PKG"
else
  expect "hub package-unyank" FAIL "this cozy has no \`package yank --undo\`"
fi
# model: 8 MiB of new random weights as a CozyTensors folder.
uv run -q --no-project --with numpy --with safetensors python -I - "$HB/w.safetensors" <<'PY'
import sys
import numpy as np
from safetensors.numpy import save_file
save_file({f"w.{i}": np.random.default_rng().standard_normal((1024, 1024)).astype(np.float16) for i in range(4)}, sys.argv[1])
PY
TENSORFS_HOME=$HB/tfs-store uv tool run -q --from "${TWF:-tensorfs}" tfs import "$HB/w.safetensors" --name local/ci-hub >/dev/null &&
  TENSORFS_HOME=$HB/tfs-store uv tool run -q --from "${TWF:-tensorfs}" tfs export local/ci-hub "$HB/w" >/dev/null
HUB_MODEL=$WHO/$HUB_PKG
hub_do model-upload ok model upload "$HB/w" "$HUB_MODEL@a" --on "$M" -y
hub_do model-publish ok model publish "$HUB_MODEL" --release 0.1.0 --commit a
hub_listed model-info-published false
hub_do model-download ok model download "$HUB_MODEL" --on "$M" -y
hub_do model-yank ok model yank "$HUB_MODEL" --release 0.1.0
hub_listed model-info-yanked true
hub_do model-pinned-yanked ok model download "$HUB_MODEL@0.1.0" --on "$M" -y
hub_do model-unyank ok model yank "$HUB_MODEL" --release 0.1.0 --undo
hub_do model-unyanked ok model download "$HUB_MODEL" --on "$M" -y
hub_do model-delete ok model delete "$HUB_MODEL" -y && HUB_MODEL=
# Deleted: gone, as the Hub serves the model (GET /v1/models/{org}/{name}) and as `model info` shows it.
hub_do model-gone refused model info "$WHO/$HUB_PKG"
hub_cleanup
