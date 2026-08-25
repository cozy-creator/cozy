#!/usr/bin/env bash
# `cozy push` -> `cozy pull` against the REAL tensorhub, with NO SHIM anywhere.
#
# cl-006's run of this script found the write side one field behind: th-003 REMOVED
# `model_family` from the publish declaration (the hub classifies the family from the
# artifact's topology digest) and decodes with DisallowUnknownFields, so every `cozy push`
# was `request.malformed_body`, exit 3, before a byte moved. cl-010 removed the field and
# `--family` with it; this script is the closure, and the push below is now a real arm
# rather than a recorded follow-up.
#
# cl-012 proved pull against a 40-line forwarding shim, because th-002 had landed the
# whole write side and no read side. th-003 (tensorhub 1353e53) landed the real one:
# POST .../checkpoints/{snapshot}/reads, presigned at the final content keys off the
# hub's own custody facts, with the REAL recorded `length` where the shim answered 0.
# This closes the loop: it is what turns "the distribution slice runs end to end" from a
# harness claim into a product claim.
#
#   scripts/xfer-realhub.sh
#
# It builds tensorhub from a read-only `git archive` of the pinned commit, runs its own
# Postgres on its own port, publishes one small artifact through the real `cozy push`,
# pulls it back into a FRESH store through the real `cozy pull`, and tears everything
# down. Bytes are a few hundred KB — this is a control-plane proof, not a weights move.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HUB_REPO="${HUB_REPO:-$HOME/cozy_v2/tensorhub}"
HUB_SHA="${HUB_SHA:-1353e53}"
TFS_REPO="${TFS_REPO:-$HOME/cozy_v2/tensorfs}"
TFS_BIN="${TFS_BIN:-$TFS_REPO/target/release/tfs}"
ENV_FILE="${ENV_FILE:-$HOME/cozy/e2e/.env}"
BUCKET="${BUCKET:-dataset-cas}"
RUN="${RUN:-$(date +%s)}"
PREFIX="v2/cl-006-realhub/${RUN}/"
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
  if [ -d "$WORK/secrets" ]; then
    python3 "$WORK/hub/scripts/r2.py" --secrets "$WORK/secrets" --bucket "$BUCKET" \
      clean "$PREFIX" 2>/dev/null | tail -1
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
nice -n 19 env CGO_ENABLED=0 go build -o "$WORK/cozy" ./cmd/cozy || { bad "cozy build"; exit 1; }
[ -x "$TFS_BIN" ] || { bad "no tfs binary at $TFS_BIN"; exit 1; }
ok "tensorhub, cozy and tfs in hand"

step "credentials and a throwaway Postgres"
mkdir -p "$WORK/secrets"
python3 - "$ENV_FILE" "$WORK/secrets" <<'PY' || { bad "credentials"; exit 1; }
import pathlib, sys
env = {}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
out = pathlib.Path(sys.argv[2])
for key, name in {
    "storage.access_key_id": "TENSORHUB_S3_ACCESS_KEY_ID",
    "storage.secret_access_key": "TENSORHUB_S3_SECRET_ACCESS_KEY",
    "storage.endpoint_url": "TENSORHUB_S3_ENDPOINT_URL",
    "storage.region": "TENSORHUB_S3_REGION",
}.items():
    if name not in env:
        sys.exit(f"{name} is not in the env file")
    (out / key).write_text(env[name])
PY
chmod 600 "$WORK"/secrets/* 2>/dev/null
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
  bucket: "${BUCKET}"
  object_prefix: "${PREFIX}"
verifier:
  tfs_path: "${TFS_BIN}"
  build: "tensorfs@${TFS_SHA}"
  work_dir: "${WORK}/verifier"
YAML

"$WORK/tensorhub" --config "$WORK/config.yaml" --secrets-dir "$WORK/secrets" serve \
  >>"$WORK/serve.log" 2>&1 &
SRV_PID=$!
for _ in $(seq 1 400); do curl -fsS "$BASE/healthz" >/dev/null 2>&1 && break; sleep 0.05; done
curl -fsS "$BASE/healthz" >/dev/null 2>&1 || { bad "the hub never answered"; tail -20 "$WORK/serve.log"; exit 1; }
ok "the REAL tensorhub is serving on ${PORT}"

# The read route exists on this build. Stated before anything is published, because it
# is the ONE thing that separates this run from cl-012's shimmed one.
READS_PROBE=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
  "$BASE/v1/repos/nobody/nothing/checkpoints/sha256:0/reads" -d '{}')
[ "$READS_PROBE" = "404" ] && ok "POST .../checkpoints/{snapshot}/reads is a ROUTE (404 on an unknown repo, not route.not_found)" \
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

step "cozy repo create + cozy push — the REAL write side"
export TENSORHUB_URL="$BASE" TENSORHUB_TOKEN="$ADMIN_TOKEN" COZY_TFS="$TFS_BIN"
COZY_HOME="$PUB" "$WORK/cozy" repo create cl006/realhub --kind model \
  --reason "cl-006 side-check: real read plane" 2>&1 | sed 's/^/  /'
PUSH_START=$(date +%s.%N)
COZY_HOME="$PUB" "$WORK/cozy" push cl006/realhub "$SNAP" \
  --reason "cl-010 push closure" 2>&1 | tee "$WORK/push.txt" | sed 's/^/  /'
PUSH_RC=${PIPESTATUS[0]}
PUSH_MS=$(python3 -c "print(f'{($(date +%s.%N)-$PUSH_START)*1000:.0f}')")
if [ "$PUSH_RC" = "0" ]; then
  ok "cozy push exited 0 in ${PUSH_MS} ms — the artifact is installed in the REAL catalog"
else
  bad "push exited $PUSH_RC — $(grep -m1 'error(' "$WORK/push.txt" | head -c 200)"
fi
grep -q "model_family" "$WORK/push.txt" && bad "the client still mentions model_family" \
  || ok "no model_family anywhere in the exchange — the hub CLASSIFIES the family"
# The classifier's verdict is the hub's, printed by the client from the completion answer.
grep -qiE "grade|satisfaction" "$WORK/push.txt" \
  && ok "completion carried the hermetic verifier's verdict" \
  || note "completion printed no grade line; see $WORK/push.txt"

step "the family the hub DERIVED — never one the client declared"
FAMILY=$(curl -fsS "$BASE/v1/repos/cl006/realhub" 2>/dev/null | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print((d.get('repo') or d).get('model_family') or 'none')" 2>/dev/null || echo "unreadable")
note "repo card model_family: $FAMILY"

step "the hub's own read grants — real lengths, no shim"
READS=$(curl -fsS -X POST "$BASE/v1/repos/cl006/realhub/checkpoints/$SNAP/reads" \
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

step "cozy pull — into a FRESH store, through the real routes"
PULL_START=$(date +%s.%N)
COZY_HOME="$DST" "$WORK/cozy" pull "cl006/realhub@$SNAP" 2>&1 | tee "$WORK/pull.txt" | sed 's/^/  /'
PULL_RC=${PIPESTATUS[0]}
PULL_MS=$(python3 -c "print(f'{($(date +%s.%N)-$PULL_START)*1000:.0f}')")
[ "$PULL_RC" = "0" ] && ok "pull exited 0 in ${PULL_MS} ms" || bad "pull exited $PULL_RC"
grep -qi "no_read_plane" "$WORK/pull.txt" && bad "the client still refused hub.no_read_plane" \
  || ok "the client never refused hub.no_read_plane — the route answered"

step "the pulled store holds the artifact, byte for byte"
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
grep -q "every declared byte" "$WORK/pull.txt" \
  && ok "and the pull said so on the way in — verified before anything became a local root" \
  || note "the pull output did not name its verification line"

step "idempotent re-pull: a second run moves nothing"
COZY_HOME="$DST" "$WORK/cozy" pull "cl006/realhub@$SNAP" 2>&1 | tee "$WORK/pull2.txt" | sed 's/^/  /'
grep -qiE "nothing to fetch|already holds|0 B" "$WORK/pull2.txt" \
  && ok "the second pull moved nothing — the store's own verification records are the journal" \
  || note "second pull output did not name a no-op; see $WORK/pull2.txt"

step "verdict"
if [ "$fail" = "0" ]; then
  printf '  \033[32mGREEN\033[0m push -> pull end to end against the REAL hub read plane, no shim\n'
else
  printf '  \033[31mRED\033[0m see the failures above\n'
fi
exit "$fail"
