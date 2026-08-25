#!/usr/bin/env bash
# cl-012 live verification: a real tensorhub against a real Postgres and real R2, a
# real tensorfs store, real artifacts, and the real `cozy` binary moving real bytes.
# There are no tests in v2 (decisions.md #160) — this runs the system and observes it.
#
#   scripts/verify-cl012.sh              everything
#   scripts/verify-cl012.sh push pull    selected phases
#   scripts/verify-cl012.sh clean        sweep this run's R2 objects and exit
#
# Phases: push dedup resume pull pullresume refusals bench
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PG_NAME=cozy-creator-v2-cl012
PG_PORT=55471
PG_PASS=verify
DB_URL="postgres://postgres:${PG_PASS}@127.0.0.1:${PG_PORT}/postgres?sslmode=disable"
ADMIN_TOKEN="cl-012-verification-token-0123456789abcdef"
HUB_PORT=18471
SHIM_PORT=18472
HUB="http://127.0.0.1:${HUB_PORT}"

ENV_FILE="${CL012_ENV_FILE:-$HOME/cozy/e2e/.env}"
HUB_REPO="${CL012_HUB_REPO:-$HOME/cozy_v2/tensorhub}"
TFS_REPO="${CL012_TFS_REPO:-$HOME/cozy_v2/tensorfs}"
# Both peers are PINNED: they have concurrent writers, and a counterparty that
# changes underneath a verification run makes every number in it unreadable.
TFS_BIN="${CL012_TFS_BIN:-$TFS_REPO/target/release/tfs}"
TFS_SHA="${CL012_TFS_SHA:-$(git -C "$TFS_REPO" rev-parse HEAD)}"
HUB_SHA="${CL012_HUB_SHA:-$(git -C "$HUB_REPO" rev-parse HEAD)}"
BUCKET="${CL012_BUCKET:-dataset-cas}"
RUN="${CL012_RUN:-$(date +%s)}"
PREFIX="v2/cl-012/${RUN}/"
WORK="${CL012_WORK:-/tmp/cl-012-$RUN}"

HUB_BIN="$WORK/tensorhub"
COZY_BIN="$WORK/cozy"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '  ---- %s\n' "$*"; }

HUB_PID=""
teardown() {
  [ -n "$HUB_PID" ] && kill "$HUB_PID" 2>/dev/null; wait "$HUB_PID" 2>/dev/null
  if [ "${CL012_KEEP:-0}" != "1" ]; then
    docker rm -f "$PG_NAME" >/dev/null 2>&1 && echo "  postgres container removed"
  fi
}
trap teardown EXIT

mkdir -p "$WORK"

# ---------------------------------------------------------------- credentials
# The storage keys are lifted out of the workspace env file into a secrets directory
# named by dotted key. They never enter this process's environment and never reach a
# child through one.
mkdir -p "$WORK/secrets"
python3 - "$ENV_FILE" "$WORK/secrets" <<'PY' || exit 1
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
    p = out / key
    p.write_text(env[name])
    p.chmod(0o600)
(out / "admin.token").write_text("")
PY
printf '%s' "$ADMIN_TOKEN" > "$WORK/secrets/admin.token"
chmod 600 "$WORK/secrets/admin.token"

if [ "${1:-}" = "clean" ]; then
  python3 "$HUB_REPO/scripts/r2.py" --secrets "$WORK/secrets" --bucket "$BUCKET" clean "$PREFIX"
  exit 0
fi

# ---------------------------------------------------------------- the peers
step "peers (pinned)"
note "tensorhub $HUB_SHA · tensorfs $TFS_SHA · bucket $BUCKET · prefix $PREFIX"
[ -x "$TFS_BIN" ] || { echo "no tfs binary at $TFS_BIN"; exit 1; }
# The hub is built from the PINNED COMMIT through a read-only archive, never from the
# peer's working tree: that repo has a concurrent writer (th-003) and its tree does not
# necessarily compile. `git archive` writes nothing there, not even worktree metadata.
rm -rf "$WORK/hubsrc"; mkdir -p "$WORK/hubsrc"
git -C "$HUB_REPO" archive "$HUB_SHA" | tar -x -C "$WORK/hubsrc" || exit 1
( cd "$WORK/hubsrc" && nice -n 19 go build -o "$HUB_BIN" ./cmd/tensorhub ) || exit 1
nice -n 19 go build -o "$COZY_BIN" ./cmd/cozy || exit 1
note "tensorhub $(stat -c%s "$HUB_BIN") B · cozy $(stat -c%s "$COZY_BIN") B"

step "postgres (dedicated container, torn down on exit)"
docker rm -f "$PG_NAME" >/dev/null 2>&1
docker run -d --name "$PG_NAME" -e POSTGRES_PASSWORD="$PG_PASS" \
  -p "${PG_PORT}:5432" postgres:18-alpine >/dev/null || exit 1
# The init sequence starts postgres, runs initdb, and RESTARTS it. `pg_isready` goes
# green on the first of those, so waiting on it races the restart and the hub gets a
# reset connection. Wait for the second "ready to accept connections" instead.
for _ in $(seq 120); do
  [ "$(docker logs "$PG_NAME" 2>&1 | grep -c 'ready to accept connections')" -ge 2 ] && break
  sleep 0.5
done
docker exec "$PG_NAME" pg_isready -U postgres >/dev/null 2>&1 || { echo "postgres never came up"; exit 1; }
note "postgres up on ${PG_PORT}"

cat > "$WORK/hub.yaml" <<YAML
env:
  name: development
server:
  http_port: ${HUB_PORT}
database:
  url: "${DB_URL}"
storage:
  bucket: "${BUCKET}"
  object_prefix: "${PREFIX}"
verifier:
  tfs_path: ${TFS_BIN}
  build: "tensorfs@${TFS_SHA:0:12}"
  work_dir: ${WORK}/verify
YAML
mkdir -p "$WORK/verify"

step "tensorhub serve"
"$HUB_BIN" --config "$WORK/hub.yaml" --secrets-dir "$WORK/secrets" serve > "$WORK/hub.log" 2>&1 &
HUB_PID=$!
for _ in $(seq 60); do
  curl -fsS "${HUB}/healthz" >/dev/null 2>&1 && break
  sleep 0.25
done
curl -fsS "${HUB}/healthz" >/dev/null || { echo "hub did not come up"; tail -20 "$WORK/hub.log"; exit 1; }
note "$(curl -fsS "${HUB}/healthz")"

# ---------------------------------------------------------------- the run
step "cl-012 live driver"
CL012_TOKEN="$ADMIN_TOKEN" nice -n 19 python3 scripts/xfer-live.py \
  --hub "$HUB" --shim-port "$SHIM_PORT" \
  --cozy "$COZY_BIN" --tfs "$TFS_BIN" \
  --work "$WORK" --secrets "$WORK/secrets" --bucket "$BUCKET" --prefix "$PREFIX" \
  --hub-sha "$HUB_SHA" --tfs-sha "$TFS_SHA" \
  ${@:+--phases "$*"}
rc=$?

step "R2 hygiene"
python3 "$HUB_REPO/scripts/r2.py" --secrets "$WORK/secrets" --bucket "$BUCKET" clean "$PREFIX"
python3 "$HUB_REPO/scripts/r2.py" --secrets "$WORK/secrets" --bucket "$BUCKET" count "$PREFIX"
python3 "$HUB_REPO/scripts/r2.py" --secrets "$WORK/secrets" --bucket "$BUCKET" uploads "$PREFIX"

exit $rc
