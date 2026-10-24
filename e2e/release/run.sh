#!/usr/bin/env bash
# The `cli-release` suite: cozy's signed-in commands before a release (e2e/coverage.toml's `release` tier).
# Signed in with the CI key (or, by hand with --installed, as this computer is), it works a throwaway
# fidika/ci-e2e-<run> package and model through lock, publish, yank, un-yank and delete, reads the account's
# credit and makes (never pays) a checkout, and rents ONE machine of --sku (default cpu): keepalive, a run on
# it, and an end while that run is still going, checked gone. The throwaway and the rental are removed in an
# EXIT trap whatever fails.
#   e2e/release/run.sh --out DIR [--key FILE | --installed] [--sku SKU] [--cozy-bin F] [--tensorfs-wheel F] [--runtime-wheel F]
. "$(dirname "$0")/../lib.sh"
parse "$@"
here=$(cd "$(dirname "$0")" && pwd)
J() { jq -e "$@" > /dev/null; }
run=$(date -u +%m%d%H%M)$RANDOM
pkg=ci-e2e-$run model=fidika/ci-e2e-$run M=''

# The machine an answer of `machine rent` names (pretty or NDJSON).
rented() { jq -r '.. | objects | select(.kind? == "rental") | .name // empty' "$1" 2> /dev/null | tail -1; }
cleanup() {
  set +e
  [ -n "$M" ] || [ ! -s "$OUT/rent.ndjson" ] || M=$(rented "$OUT/rent.ndjson")
  if [ -n "$M" ]; then
    cozy --json machine end "$M" --force -y > "$OUT/end-trap.json" 2>&1
    for _ in $(seq 30); do cozy --json machine list | J --arg m "$M" '.machines | any(.name == $m)' || break; sleep 10; done
  fi
  for r in 0.1.0 0.2.0; do cozy --json package yank "fidika/$pkg@$r" > /dev/null 2>&1; done
  cozy --json model delete "$model" -y > /dev/null 2>&1
}
trap cleanup EXIT

if [ -z "$INSTALLED" ]; then step install; install_cozy; user_services; step candidates; candidates; fi

step sign-in
if [ -n "$KEY" ]; then cozy --json login --key "$KEY" > "$OUT/login.json" || fail "cozy login --key failed"; fi
cozy --json whoami > "$OUT/whoami.json"
J '.logged_in == true and .account == "fidika"' "$OUT/whoami.json" || fail "whoami: not signed in as fidika ($(head -c 300 "$OUT/whoami.json"))"

step credits
cozy --json credits > "$OUT/credits.json" && J '.billing | length > 0' "$OUT/credits.json" || check "credits: no billing"
cozy --json credits history > "$OUT/history.json" || check "credits history failed"
J 'type == "object"' "$OUT/history.json" || check "credits history: no answer"
cozy --json credits buy 10 --no-browser --no-wait > "$OUT/buy.json" || check "credits buy 10 (the least one purchase may be) failed"
checkout=$(jq -r '.. | strings | select(test("^https://"))' "$OUT/buy.json" | head -1) || true
[ -n "$checkout" ] && curl -fsS -o /dev/null "$checkout" || check "credits buy: no payment page ($(head -c 300 "$OUT/buy.json"))"

step "throwaway package"
dir=$OUT/$pkg
cp -r "$here/../cli/weightless" "$dir"
sed -i.orig -e "s/^name = \"e2e-weightless\"/name = \"$pkg\"/" "$dir/pyproject.toml" && rm -f "$dir/pyproject.toml.orig"
(cd "$dir" && cozy --json package lock > "$OUT/lock.json") && [ -s "$dir/uv.lock" ] || check "package lock: no uv.lock"
(cd "$dir" && cozy --json package lock --upgrade-package msgspec > "$OUT/lock-upgrade.json") && grep -q 'name = "msgspec"' "$dir/uv.lock" \
  || check "package lock --upgrade-package msgspec: msgspec not locked"
(cd "$dir" && cozy --json package publish > "$OUT/publish-1.json") || check "package publish 0.1.0 failed"
sed -i.orig 's/^version = "0.1.0"/version = "0.2.0"/' "$dir/pyproject.toml" && rm -f "$dir/pyproject.toml.orig"
(cd "$dir" && cozy --json package lock > /dev/null && cozy --json package publish > "$OUT/publish-2.json") || check "package publish 0.2.0 failed"
cozy --json package info "fidika/$pkg" > "$OUT/pkg-info.json" && J '[.releases[].release] | index("0.1.0") and index("0.2.0")' "$OUT/pkg-info.json" \
  || check "package info: not both releases"
resolves() {  # the release an unversioned run of the throwaway takes
  cozy --json run "fidika/$pkg/reverse" text=e2e --on local > "$OUT/resolve-$1.ndjson" 2>&1
  grep -oE "\"target\": *\"[^\"]*$pkg[^\"]*\"" "$OUT/resolve-$1.ndjson" | head -1 || true
}
[[ $(resolves before) == *@0.2.0* ]] || check "the throwaway's unversioned run is not 0.2.0"
cozy --json package yank "fidika/$pkg@0.2.0" > "$OUT/yank.json" || check "package yank failed"
[[ $(resolves yanked) == *@0.1.0* ]] || check "package yank: an unversioned run still takes 0.2.0"
cozy --json package yank "fidika/$pkg@0.2.0" --undo > "$OUT/unyank.json" || check "package yank --undo failed"
[[ $(resolves unyanked) == *@0.2.0* ]] || check "package yank --undo: an unversioned run does not take 0.2.0 again"

step "throwaway model"
uv run -q --no-project python -I - "$OUT/tiny.safetensors" <<'PY'
import json, struct, sys
header = json.dumps({"w": {"dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32]}}).encode()
header += b" " * (-len(header) % 8)
open(sys.argv[1], "wb").write(struct.pack("<Q", len(header)) + header + bytes(32))
PY
cozy --json model upload "$OUT/tiny.safetensors" "$model" --on local > "$OUT/upload.ndjson" || check "model upload failed"
commit=$(jq -r '.. | objects | .commit? // empty' "$OUT/upload.ndjson" 2> /dev/null | tail -1) || true
cozy --json model family e2e-family "$model" > /dev/null && cozy --json model info "$model" | J '.family == "e2e-family"' || check "model family: not set"
cozy --json model family --clear "$model" > /dev/null && cozy --json model info "$model" | J '.family == null' || check "model family --clear: not cleared"
cozy --json model publish "$model" --release 0.1.0 ${commit:+--commit "$commit"} > "$OUT/model-publish.json" || check "model publish failed"
cozy --json model info "$model" > "$OUT/model-info.json" && J '.releases | any(.release == "0.1.0" and .yanked == false)' "$OUT/model-info.json" \
  || check "model publish: no 0.1.0"
cozy --json model yank "$model" --release 0.1.0 > /dev/null && cozy --json model info "$model" | J '.releases | any(.release == "0.1.0" and .yanked)' \
  || check "model yank: 0.1.0 not yanked"
cozy --json model yank "$model" --release 0.1.0 --undo > /dev/null && cozy --json model info "$model" | J '.releases | any(.release == "0.1.0" and (.yanked | not))' \
  || check "model yank --undo: 0.1.0 still yanked"
cozy --json model delete "$model" -y > "$OUT/model-delete.json" || check "model delete failed"
! cozy --json model info "$model" > /dev/null 2>&1 || check "model delete: $model still answers"

step rental
before=$(cozy --json machine list | jq '[.machines[] | select(.kind == "rental")] | length') || true
! cozy --json machine rent "$SKU" --gpus 64 -y > "$OUT/rent-too-many.json" 2>&1 || check "machine rent --gpus 64 bought something"
[ "$(cozy --json machine list | jq '[.machines[] | select(.kind == "rental")] | length')" = "$before" ] || check "machine rent --gpus 64: a rental appeared"
cozy --json machine rent "$SKU" --disk-gb 60 -y > "$OUT/rent.ndjson" || fail "machine rent $SKU failed: $(tail -c 400 "$OUT/rent.ndjson")"
M=$(rented "$OUT/rent.ndjson") || true
[ -n "$M" ] || fail "machine rent: no machine named"
echo "rented $M ($SKU)"
cozy --json machine show "$M" > "$OUT/rental.json" && J '.rental.disk_gb >= 60' "$OUT/rental.json" || check "machine rent --disk-gb 60: $(jq .rental.disk_gb "$OUT/rental.json") GB"
deadline=$(jq '.idle_deadline_ms // 0' "$OUT/rental.json") || true
sleep 5
cozy --json machine keepalive "$M" > "$OUT/keepalive.json" || check "machine keepalive failed"
J --argjson d "${deadline:-0}" '.idle_deadline_ms > $d' "$OUT/keepalive.json" || check "machine keepalive: its idle deadline did not move"
(cd "$here/../cli" && cozy --json run ./weightless/reverse text=rented --on "$M" > "$OUT/on-rental.ndjson") && grep -q detner "$OUT/on-rental.ndjson" \
  || check "a run on $M: no reply"
(cd "$here/../cli" && cozy --json run ./weightless/wait seconds:=600 --on "$M" > "$OUT/rental-wait.ndjson" 2>&1) &
for _ in $(seq 60); do grep -qE '"status": *"in_progress"' "$OUT/rental-wait.ndjson" && break; sleep 5; done
cozy --json machine end "$M" --force -y > "$OUT/end.json" || check "machine end $M --force failed"
gone=''
for _ in $(seq 30); do cozy --json machine list | J --arg m "$M" '.machines | any(.name == $m)' || { gone=1; break; }; sleep 10; done
[ -n "$gone" ] && M='' || check "machine end --force: $M is still listed after 5 min"
wait || true

step coverage; coverage "$COZY" release
finish "cli-release: $pkg, $model, ${SKU}"
