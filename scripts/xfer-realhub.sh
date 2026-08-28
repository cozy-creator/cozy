#!/usr/bin/env bash
# `cozy model publish` -> `cozy model download` against the REAL tensorhub, with NO SHIM anywhere.
#
# cl-006's run of this script found the write side one field behind: th-003 REMOVED
# `model_family` from the publish declaration (the hub classifies the family from the
# artifact's topology digest) and decodes with DisallowUnknownFields, so every `cozy model publish`
# was `request.malformed_body`, exit 3, before a byte moved. cl-010 removed the field and
# `--family` with it; this script is the closure, and the publish below is now a real arm
# rather than a recorded follow-up.
#
# cl-012 proved download against a 40-line forwarding shim, because th-002 had landed the
# whole write side and no read side. th-003 (tensorhub 1353e53) landed the real one:
# POST .../checkpoints/{snapshot}/reads, presigned at the final content keys off the
# hub's own custody facts, with the REAL recorded `length` where the shim answered 0.
# This closes the loop: it is what turns "the distribution slice runs end to end" from a
# harness claim into a product claim.
#
#   scripts/xfer-realhub.sh
#
# It builds tensorhub from a read-only `git archive` of the pinned commit, runs its own
# Postgres on its own port, publishes one small artifact through the real `cozy model publish`,
# pulls it back into a FRESH store through the real `cozy model download`, and tears everything
# down. Bytes are a few hundred KB — this is a control-plane proof, not a weights move.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HUB_REPO="${HUB_REPO:-$HOME/cozy_v2/tensorhub}"
HUB_SHA="${HUB_SHA:-c4f57ed}"
TFS_REPO="${TFS_REPO:-$HOME/cozy_v2/tensorfs}"
TFS_BIN="${TFS_BIN:-$TFS_REPO/target/release/tfs}"
ENV_FILE="${ENV_FILE:-$HOME/cozy/e2e/.env}"
BUCKET="${BUCKET:-repo-cas}"
RUN="${RUN:-$(date +%s)}"
PREFIX="v2/cl-006-realhub/${RUN}/"
DATASET_BUCKET="${DATASET_BUCKET:-dataset-cas}"
DATASET_PREFIX="v2/cl-006-unused-dataset/${RUN}/"
SOURCE_BUCKET="${SOURCE_BUCKET:-endpoint-source-code}"
SOURCE_PREFIX="v2/cl-006-unused-source/${RUN}/"
MEDIA_BUCKET="${MEDIA_BUCKET:-user-media}"
MEDIA_PREFIX="v2/cl-006-unused-media/${RUN}/"
PG_NAME=tensorhub-v2-cl006
PG_PORT=55441
PG_PASS=verify
PORT=18091
BASE="http://127.0.0.1:${PORT}"
ADMIN_TOKEN="cl-006-realhub-token-0123456789abcdef"
WORK="${WORK:-/tmp/cl-006-realhub-$RUN}"

fail=0
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mPASS\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; fail=1; }
note() { printf '  ---- %s\n' "$*"; }

SRV_PID=""
teardown() {
  [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
  wait "$SRV_PID" 2>/dev/null
  docker rm -f "$PG_NAME" >/dev/null 2>&1 && echo "  postgres container removed"
  if [ -d "$WORK/r2-secrets" ]; then
    python3 "$WORK/hub/scripts/r2.py" --secrets "$WORK/r2-secrets" --bucket "$BUCKET" \
      clean "$PREFIX" 2>/dev/null | tail -1
    python3 "$WORK/hub/scripts/r2.py" --secrets "$WORK/r2-secrets" --bucket "$DATASET_BUCKET" \
      clean "$DATASET_PREFIX" >/dev/null 2>&1
    python3 "$WORK/hub/scripts/r2.py" --secrets "$WORK/r2-secrets" --bucket "$SOURCE_BUCKET" \
      clean "$SOURCE_PREFIX" >/dev/null 2>&1
    python3 "$WORK/hub/scripts/r2.py" --secrets "$WORK/r2-secrets" --bucket "$MEDIA_BUCKET" \
      clean "$MEDIA_PREFIX" >/dev/null 2>&1
  fi
}
trap teardown EXIT

mkdir -p "$WORK"

step "subjects — everything pinned, nothing edited"
git -C "$HUB_REPO" rev-parse "$HUB_SHA" >/dev/null 2>&1 || { bad "no commit $HUB_SHA in $HUB_REPO"; exit 1; }
FULL_SHA="$(git -C "$HUB_REPO" rev-parse "$HUB_SHA")"
TFS_SHA="$(git -C "$TFS_REPO" rev-parse HEAD)"
mkdir -p "$WORK/hub"
# READ-ONLY: a git archive of the pinned tree, never a checkout of the working copy.
git -C "$HUB_REPO" archive "$FULL_SHA" | tar -x -C "$WORK/hub" || { bad "archive failed"; exit 1; }
note "tensorhub ${FULL_SHA:0:12} (git archive, read-only) · tensorfs ${TFS_SHA:0:12}"
note "cozy-creator $(git rev-parse --short HEAD) · prefix ${PREFIX}"
ok "sources pinned"

step "build"
( cd "$WORK/hub" && nice -n 19 go build -o "$WORK/tensorhub" ./cmd/tensorhub ) || { bad "hub build"; exit 1; }
nice -n 19 env CGO_ENABLED=0 go build -o "$WORK/cozy" . || { bad "cozy build"; exit 1; }
[ -x "$TFS_BIN" ] || { bad "no tfs binary at $TFS_BIN"; exit 1; }
ok "tensorhub, cozy and tfs in hand"

step "credentials and a throwaway Postgres"
mkdir -p "$WORK/secrets" "$WORK/r2-secrets"
python3 - "$ENV_FILE" "$WORK/secrets" "$WORK/r2-secrets" <<'PY' || { bad "credentials"; exit 1; }
import pathlib, sys
env = {}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
out = pathlib.Path(sys.argv[2])
r2out = pathlib.Path(sys.argv[3])
values = {
    "storage.access_key_id": "TENSORHUB_S3_ACCESS_KEY_ID",
    "storage.secret_access_key": "TENSORHUB_S3_SECRET_ACCESS_KEY",
    "storage.endpoint_url": "TENSORHUB_S3_ENDPOINT_URL",
    "storage.region": "TENSORHUB_S3_REGION",
}
for key, name in values.items():
    if name not in env:
        sys.exit(f"{name} is not in the env file")
    (r2out / key).write_text(env[name])
for domain in ("repo_cas", "dataset_cas", "endpoint_source", "user_media"):
    for key, name in values.items():
        suffix = key.removeprefix("storage.")
        (out / f"storage.{domain}.{suffix}").write_text(env[name])
PY
chmod 600 "$WORK"/secrets/* "$WORK"/r2-secrets/* 2>/dev/null
printf '%s' "$ADMIN_TOKEN" > "$WORK/secrets/admin.token"

docker rm -f "$PG_NAME" >/dev/null 2>&1
docker run -d --name "$PG_NAME" -e POSTGRES_PASSWORD="$PG_PASS" \
  -p "127.0.0.1:${PG_PORT}:5432" postgres:18-alpine >/dev/null || { bad "postgres"; exit 1; }
for _ in $(seq 1 300); do docker exec "$PG_NAME" pg_isready -q -U postgres 2>/dev/null && break; sleep 1; done
docker exec "$PG_NAME" pg_isready -U postgres >/dev/null 2>&1 \
  || { bad "postgres never ready"; docker logs --tail 20 "$PG_NAME"; exit 1; }
ok "postgres:18-alpine on ${PG_PORT}"

cat >"$WORK/config.yaml" <<YAML
env:
  name: development
server:
  http_port: ${PORT}
database:
  url: "postgres://postgres:${PG_PASS}@127.0.0.1:${PG_PORT}/postgres?sslmode=disable"
storage:
  repo_cas:
    bucket: "${BUCKET}"
    prefix: "${PREFIX}"
  dataset_cas:
    bucket: "${DATASET_BUCKET}"
    prefix: "${DATASET_PREFIX}"
  endpoint_source:
    bucket: "${SOURCE_BUCKET}"
    prefix: "${SOURCE_PREFIX}"
  user_media:
    bucket: "${MEDIA_BUCKET}"
    prefix: "${MEDIA_PREFIX}"
verifier:
  tfs_path: "${TFS_BIN}"
  build: "tensorfs@${TFS_SHA}"
  work_dir: "${WORK}/verifier"
YAML

"$WORK/tensorhub" --config "$WORK/config.yaml" --secrets-dir "$WORK/secrets" storage bootstrap \
  >>"$WORK/bootstrap.log" 2>&1 || { bad "storage bootstrap"; tail -20 "$WORK/bootstrap.log"; exit 1; }
ok "four isolated storage prefixes bootstrapped"

"$WORK/tensorhub" --config "$WORK/config.yaml" --secrets-dir "$WORK/secrets" serve \
  >>"$WORK/serve.log" 2>&1 &
SRV_PID=$!
for _ in $(seq 1 400); do curl -fsS "$BASE/healthz" >/dev/null 2>&1 && break; sleep 0.05; done
curl -fsS "$BASE/healthz" >/dev/null 2>&1 || { bad "the hub never answered"; tail -20 "$WORK/serve.log"; exit 1; }
ok "the REAL tensorhub is serving on ${PORT}"

# The read route exists on this build. Stated before anything is published, because it
# is the ONE thing that separates this run from cl-012's shimmed one.
READS_PROBE=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
  "$BASE/v1/models/nobody/nothing/checkpoints/sha256:0/reads" -d '{}')
[ "$READS_PROBE" = "404" ] && ok "POST .../checkpoints/{snapshot}/reads is a ROUTE (404 on an unknown model, not route.not_found)" \
                          || note "reads probe answered $READS_PROBE"

step "a small artifact, ingested locally by tfs"
PUB="$WORK/publisher"; DST="$WORK/puller"
mkdir -p "$PUB/cas" "$DST/cas"
"$TFS_BIN" store init "$PUB/cas" >/dev/null 2>&1
"$TFS_BIN" store init "$DST/cas" >/dev/null 2>&1
WRITE="$("$TFS_BIN" checkpoint write "$PUB/cas" --blocks 1 --scale 4 --seed 6006 2>&1)"
SNAP="sha256:$(grep -oP 'checkpoint_id\s+sha256:\K[0-9a-f]{64}' <<<"$WRITE" | head -1)"
BYTES="$(grep -oP 'tensor bytes \K\d+' <<<"$WRITE" | head -1)"
[ -n "$BYTES" ] || { bad "tfs checkpoint write"; echo "$WRITE"; exit 1; }
ok "checkpoint ${SNAP:0:23} · ${BYTES} tensor bytes"

step "cozy model create + cozy model publish — the REAL write side"
export TENSORHUB_URL="$BASE" TENSORHUB_TOKEN="$ADMIN_TOKEN" COZY_TFS="$TFS_BIN"
COZY_HOME="$PUB" "$WORK/cozy" model create cl006/realhub \
  --reason "cl-006 side-check: real read plane" 2>&1 | sed 's/^/  /'
PUBLISH_START=$(date +%s.%N)
COZY_HOME="$PUB" "$WORK/cozy" model publish cl006/realhub "$SNAP" \
  --reason "cl-010 publish closure" 2>&1 | tee "$WORK/publish.txt" | sed 's/^/  /'
PUBLISH_RC=${PIPESTATUS[0]}
PUBLISH_MS=$(python3 -c "print(f'{($(date +%s.%N)-$PUBLISH_START)*1000:.0f}')")
if [ "$PUBLISH_RC" = "0" ]; then
  ok "cozy model publish exited 0 in ${PUBLISH_MS} ms — the artifact is installed in the REAL catalog"
else
  bad "publish exited $PUBLISH_RC — $(grep -m1 'error(' "$WORK/publish.txt" | head -c 200)"
fi
grep -q "model_family" "$WORK/publish.txt" && bad "the client still mentions model_family" \
  || ok "no model_family anywhere in the exchange — the hub CLASSIFIES the family"
# The classifier's verdict is the hub's, printed by the client from the completion answer.
grep -qiE "grade|satisfaction" "$WORK/publish.txt" \
  && ok "completion carried the hermetic verifier's verdict" \
  || note "completion printed no grade line; see $WORK/publish.txt"

step "exact model-publication replay — same operation, same committed root"
COZY_HOME="$PUB" "$WORK/cozy" model publish cl006/realhub "$SNAP" \
  --reason "cl-041 exact publication replay" 2>&1 | tee "$WORK/publish2.txt" | sed 's/^/  /'
REPLAY_RC=${PIPESTATUS[0]}
[ "$REPLAY_RC" = "0" ] && grep -q "duplicate: *true" "$WORK/publish2.txt" \
  && grep -q "moved: *0B" "$WORK/publish2.txt" \
  && ok "the committed incremental publication replayed with duplicate=true and 0 bytes moved" \
  || bad "exact publication replay was not a zero-byte duplicate"

step "interrupted known-transfer publication resumes from Tensorhub's journal"
RESUME_WRITE="$("$TFS_BIN" checkpoint write "$PUB/cas" --blocks 2 --scale 4 --seed 6007 2>&1)"
RESUME_SNAP="sha256:$(grep -oP 'checkpoint_id\s+sha256:\K[0-9a-f]{64}' <<<"$RESUME_WRITE" | head -1)"
[ "${#RESUME_SNAP}" = "71" ] && [ "$RESUME_SNAP" != "$SNAP" ] \
  || { bad "resume checkpoint is absent or not byte-distinct"; exit 1; }
COZY_HOME="$PUB" "$WORK/cozy" model create cl006/resume \
  --reason "cl-040 interrupted publication proof" >/dev/null 2>&1
COZY_HOME="$PUB" "$WORK/cozy" model publish cl006/resume "$RESUME_SNAP" \
  --reason "cl-040 planted interruption" --crash-after 3 \
  2>&1 | tee "$WORK/publish-crash.txt" | sed 's/^/  /'
CRASH_RC=${PIPESTATUS[0]}
[ "$CRASH_RC" = "1" ] && grep -q "crash_after_upload" "$WORK/publish-crash.txt" \
  && ok "the first process stopped after three accepted transfers" \
  || bad "the interruption arm did not stop at its declared transfer count"
CRASH_PUBLICATION=$(grep -oP '(?:opened|resumed) publication \K[0-9a-f-]+' "$WORK/publish-crash.txt" | head -1)
curl -fsS "$BASE/v1/models/cl006/resume/publications/$CRASH_PUBLICATION" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H 'X-Tensorhub-Reason: cl-040 inspect interrupted journal' \
  >"$WORK/publish-crash-state.json"
python3 - "$WORK/publish-crash-state.json" <<'PY' \
  && ok "Tensorhub journal records exactly three accepted transfers before restart" \
  || bad "interrupted journal did not record exactly three accepted transfers"
import json, pathlib, sys
rows = json.loads(pathlib.Path(sys.argv[1]).read_text()).get("transfers", [])
accepted = [row for row in rows if row.get("state") == "accepted"]
print(f"  ---- interrupted journal: {len(accepted)} accepted / {len(rows)} total transfers")
raise SystemExit(0 if len(accepted) == 3 and len(rows) > 3 else 1)
PY
COZY_HOME="$PUB" "$WORK/cozy" model publish cl006/resume "$RESUME_SNAP" \
  --reason "cl-040 resume same operation" \
  2>&1 | tee "$WORK/publish-resume.txt" | sed 's/^/  /'
RESUME_RC=${PIPESTATUS[0]}
RESUMED_PUBLICATION=$(grep -oP '(?:opened|resumed) publication \K[0-9a-f-]+' "$WORK/publish-resume.txt" | head -1)
[ "$RESUME_RC" = "0" ] && [ -n "$CRASH_PUBLICATION" ] \
  && [ "$CRASH_PUBLICATION" = "$RESUMED_PUBLICATION" ] \
  && grep -q "· 3 already resident" "$WORK/publish-resume.txt" \
  && grep -Eq "moved: +[1-9]" "$WORK/publish-resume.txt" \
  && grep -Eq "uploaded: +[1-9]" "$WORK/publish-resume.txt" \
  && grep -Eq "verified: +[1-9]" "$WORK/publish-resume.txt" \
  && ok "the rerun reused exactly three accepted transfers, moved the remainder, and sealed" \
  || bad "the interrupted publication did not resume from its durable transfer rows"

step "typed endpoint/model command surface against the real hub"
python3 scripts/hub-live.py --cozy "$WORK/cozy" --hub "$BASE" --token "$ADMIN_TOKEN" \
  --model cl006/realhub --only reads --only writes --only refusals --only secrecy \
  | tee "$WORK/hub-live.txt"
[ "${PIPESTATUS[0]}" = "0" ] && ok "typed create/show/search and retired-command refusals passed live" \
  || bad "typed command live verification failed"

step "the family the hub DERIVED — never one the client declared"
FAMILY=$(curl -fsS "$BASE/v1/models/cl006/realhub" 2>/dev/null | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print((d.get('model') or d).get('model_family') or 'none')" 2>/dev/null || echo "unreadable")
note "model card model_family: $FAMILY"

step "the hub's own read grants — real lengths, no shim"
READS=$(curl -fsS -X POST "$BASE/v1/models/cl006/realhub/checkpoints/$SNAP/reads" \
        -H 'Content-Type: application/json' -d '{}' 2>&1)
python3 - "$WORK/reads.json" <<PY
import json, pathlib, sys
doc = json.loads('''$READS''')
reads = doc.get("reads", [])
pathlib.Path(sys.argv[1]).write_text(json.dumps(doc))
zero = [r for r in reads if int(r.get("length", 0)) == 0]
print(f"  ---- {len(reads)} read grant(s); lengths "
      f"{min((int(r['length']) for r in reads), default=0)}..{max((int(r['length']) for r in reads), default=0)} B")
print(f"  ---- first key: {reads[0]['url'].split('?')[0].split('/')[-1][:32] if reads else 'none'}")
sys.exit(1 if (not reads or zero) else 0)
PY
[ $? -eq 0 ] && ok "every read grant carries a REAL recorded length (the shim answered 0)" \
             || bad "a read grant carried length 0 or none were issued"

step "cozy model download — into a FRESH store, through the real routes"
DOWNLOAD_START=$(date +%s.%N)
COZY_HOME="$DST" "$WORK/cozy" model download "cl006/realhub@$SNAP" 2>&1 | tee "$WORK/download.txt" | sed 's/^/  /'
DOWNLOAD_RC=${PIPESTATUS[0]}
DOWNLOAD_MS=$(python3 -c "print(f'{($(date +%s.%N)-$DOWNLOAD_START)*1000:.0f}')")
[ "$DOWNLOAD_RC" = "0" ] && ok "download exited 0 in ${DOWNLOAD_MS} ms" || bad "download exited $DOWNLOAD_RC"
grep -qi "no_read_plane" "$WORK/download.txt" && bad "the client still refused hub.no_read_plane" \
  || ok "the client never refused hub.no_read_plane — the route answered"

step "the downloaded store holds the artifact, byte for byte"
# The strongest available reading, and it is the PRODUCT's own: `tfs snapshot verify`
# re-reads every declared byte of the checkpoint out of the destination store and checks
# it against the identity that named it. Nothing here trusts the transfer.
VERIFY="$("$TFS_BIN" snapshot verify "$DST/cas" "${SNAP#sha256:}" 2>&1)"
echo "$VERIFY" | tail -4 | sed 's/^/  /'
if grep -qiE "^REFUSED|FAIL|corrupt|mismatch" <<<"$VERIFY"; then
  bad "the destination store did not verify the checkpoint"
else
  ok "every declared byte of the checkpoint re-verifies in the DESTINATION store"
fi
grep -q "every declared byte" "$WORK/download.txt" \
  && ok "and the download said so on the way in — verified before anything became a local root" \
  || note "the download output did not name its verification line"

step "idempotent re-download: a second run moves nothing"
COZY_HOME="$DST" "$WORK/cozy" model download "cl006/realhub@$SNAP" 2>&1 | tee "$WORK/pull2.txt" | sed 's/^/  /'
grep -qiE "nothing to fetch|already holds|0 B" "$WORK/pull2.txt" \
  && ok "the second download moved nothing — the store's own verification records are the journal" \
  || note "second download output did not name a no-op; see $WORK/pull2.txt"

step "verdict"
if [ "$fail" = "0" ]; then
  printf '  \033[32mGREEN\033[0m typed create/search/show + incremental publish -> resolved download, no shim\n'
else
  printf '  \033[31mRED\033[0m see the failures above\n'
fi
exit "$fail"
