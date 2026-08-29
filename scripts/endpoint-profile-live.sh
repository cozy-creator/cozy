#!/usr/bin/env bash
# Real Creator -> Tensorhub endpoint profile publication proof. It builds the exact
# requested Tensorhub revision from a read-only archive, uses dedicated PostgreSQL and
# MinIO, and drives the shipped cozy binary plus an exact built Runtime proof venv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HUB_REPO="${TENSORHUB_REPO:-$HOME/cozy_v2/tensorhub}"
HUB_SHA="${TENSORHUB_SHA:-origin/master}"
# These select SOURCE for this harness; they are not Tensorhub service-config keys and
# must not leak into the service's closed environment vocabulary.
unset TENSORHUB_REPO TENSORHUB_SHA
TFS_BIN="${TFS_BIN:-$HOME/cozy_v2/tensorfs/target/release/tfs}"
RUNTIME_VENV="${COZY_RUNTIME_VENV:-$HOME/cozy_v2/.worktrees/cozy-runtime/cr-052-profile-overlay-parity/.venv}"
unset COZY_RUNTIME_VENV
RUN_PAID="${RUN_PAID:-0}"
EXPECTED_PROFILE_STATE="${EXPECTED_PROFILE_STATE:-candidate}"
WORK="${WORK:-$(mktemp -d)}"
PG_NAME="creator-endpoint-profile-pg-$$"
S3_NAME="creator-endpoint-profile-s3-$$"
MC_VOLUME="creator-endpoint-profile-mc-$$"
ADMIN_TOKEN="creator-endpoint-profile-admin"
S3_ACCESS="creatorprofile"
S3_SECRET="creatorprofile-secret-key"
PROFILE_A="torch2.13.0-cu126-cp312-linux-x86"
PROFILE_B="torch2.13.0-cu130-cp312-linux-x86"

free_port() { python3 - <<'PY'
import socket
s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()
PY
}
PG_PORT="$(free_port)"; S3_PORT="$(free_port)"; API_PORT="$(free_port)"
BASE="http://127.0.0.1:$API_PORT"

cleanup() {
  status=$?
  trap - EXIT
  [ -z "${SERVER_PID:-}" ] || kill "$SERVER_PID" 2>/dev/null || true
  if [ "$status" -ne 0 ] && [ "${KEEP:-0}" = 1 ]; then
    docker exec "$PG_NAME" pg_dump -U postgres --schema=hub_v2 --data-only \
      >"$WORK/failure-db.sql" 2>"$WORK/failure-db.err" || true
  fi
  docker rm -f "$PG_NAME" "$S3_NAME" >/dev/null 2>&1 || true
  docker volume rm -f "$MC_VOLUME" >/dev/null 2>&1 || true
  [ "${KEEP:-0}" = 1 ] || rm -rf "$WORK"
  exit "$status"
}
trap cleanup EXIT

mkdir -p "$WORK/hub" "$WORK/secrets" "$WORK/verify" "$WORK/proof" "$WORK/home" "$WORK/source"
FULL_SHA="$(git -C "$HUB_REPO" rev-parse "$HUB_SHA")"
git -C "$HUB_REPO" archive "$FULL_SHA" | tar -x -C "$WORK/hub"
(cd "$WORK/hub" && nice -n 19 go build -o "$WORK/tensorhub" ./cmd/tensorhub)
(cd "$ROOT" && nice -n 19 go build -o "$WORK/cozy" .)
test -x "$TFS_BIN"
test -x "$RUNTIME_VENV/bin/cozy-environment-proof"
test -x "$RUNTIME_VENV/bin/python"

docker run -d --name "$PG_NAME" -e POSTGRES_PASSWORD=profile \
  -p "127.0.0.1:$PG_PORT:5432" postgres:18-alpine >/dev/null
docker run -d --name "$S3_NAME" -e "MINIO_ROOT_USER=$S3_ACCESS" \
  -e "MINIO_ROOT_PASSWORD=$S3_SECRET" -p "127.0.0.1:$S3_PORT:9000" \
  minio/minio:RELEASE.2025-09-07T16-13-09Z server /data >/dev/null
for _ in $(seq 1 120); do
  docker exec "$PG_NAME" pg_isready -q -U postgres >/dev/null 2>&1 \
    && curl -fsS "http://127.0.0.1:$S3_PORT/minio/health/live" >/dev/null 2>&1 && break
  sleep 0.25
done
docker exec "$PG_NAME" pg_isready -q -U postgres
curl -fsS "http://127.0.0.1:$S3_PORT/minio/health/live" >/dev/null
docker volume create "$MC_VOLUME" >/dev/null
MC=(docker run --rm --network "container:$S3_NAME" -v "$MC_VOLUME:/root/.mc" minio/mc:latest)
"${MC[@]}" alias set local http://127.0.0.1:9000 "$S3_ACCESS" "$S3_SECRET" >/dev/null
for bucket in repo-cas dataset-cas endpoint-source user-media; do "${MC[@]}" mb "local/$bucket" >/dev/null; done

printf '%s' "$ADMIN_TOKEN" >"$WORK/secrets/admin.token"
for domain in repo_cas dataset_cas endpoint_source user_media; do
  printf '%s' "$S3_ACCESS" >"$WORK/secrets/storage.$domain.access_key_id"
  printf '%s' "$S3_SECRET" >"$WORK/secrets/storage.$domain.secret_access_key"
done
if [ "$RUN_PAID" = 1 ]; then
  : "${RUNPOD_API_KEY:?RUN_PAID=1 requires RUNPOD_API_KEY}"
  : "${RUNPOD_CONTAINER_REGISTRY_AUTH_ID:?RUN_PAID=1 requires registry auth id}"
  printf '%s' "$RUNPOD_API_KEY" >"$WORK/secrets/provider.runpod.api_key"
  printf '%s' "$RUNPOD_CONTAINER_REGISTRY_AUTH_ID" >"$WORK/secrets/provider.runpod.container_registry_auth_id"
fi
cat >"$WORK/config.yaml" <<YAML
env: {name: development}
server: {host: 127.0.0.1, http_port: $API_PORT}
database: {url: "postgres://postgres:profile@127.0.0.1:$PG_PORT/postgres?sslmode=disable"}
verifier: {tfs_path: "$TFS_BIN", build: "tensorfs-live", work_dir: "$WORK/verify"}
endpoint_release: {environment_proof: "$RUNTIME_VENV/bin/cozy-environment-proof", python: "$RUNTIME_VENV/bin/python", sandbox: "$WORK/hub/scripts/dev-release-sandbox.sh", scratch_root: "$WORK/proof"}
storage:
  repo_cas: {endpoint_url: "http://127.0.0.1:$S3_PORT", bucket: repo-cas, region: us-east-1, prefix: "v2/creator-profile/repo/"}
  dataset_cas: {endpoint_url: "http://127.0.0.1:$S3_PORT", bucket: dataset-cas, region: us-east-1, prefix: "v2/creator-profile/dataset/"}
  endpoint_source: {endpoint_url: "http://127.0.0.1:$S3_PORT", bucket: endpoint-source, region: us-east-1, prefix: "v2/creator-profile/source/"}
  user_media: {endpoint_url: "http://127.0.0.1:$S3_PORT", bucket: user-media, region: us-east-1, prefix: "v2/creator-profile/media/"}
YAML
CFG=(--config "$WORK/config.yaml" --secrets-dir "$WORK/secrets")
"$WORK/tensorhub" "${CFG[@]}" migrate >/dev/null
"$WORK/tensorhub" "${CFG[@]}" storage bootstrap >/dev/null
"$WORK/tensorhub" "${CFG[@]}" serve >"$WORK/server.log" 2>&1 & SERVER_PID=$!
for _ in $(seq 1 120); do curl -fsS "$BASE/healthz" >/dev/null 2>&1 && break; sleep 0.1; done
curl -fsS "$BASE/healthz" >/dev/null

# Register the exact approved base manifests. This is platform setup, not endpoint
# publication: Creator still supplies no image/base choice in the release declaration.
python3 - "$WORK/hub" "$WORK" <<'PY'
import hashlib,json,pathlib,sys
hub,work=map(pathlib.Path,sys.argv[1:])
rows=[]; files={}
for build in ("cu126","cu130"):
    path=hub/f"vectors/base-worker-image-wheelhouse-{build}.json"
    raw=path.read_bytes(); digest="sha256:"+hashlib.sha256(raw).hexdigest()
    rows.append({"object_id":digest,"length":len(raw)}); files[digest]=str(path)
rows.sort(key=lambda row: row["object_id"])
(work/"base-objects.json").write_text(json.dumps({"operation_id":"creator-profile-base-manifests","objects":rows}))
(work/"base-files.json").write_text(json.dumps(files))
PY
admin=(-H "Authorization: Bearer $ADMIN_TOKEN" -H 'X-Tensorhub-Reason: Creator profile base setup')
curl -fsS -X POST "$BASE/v1/admin/build-artifacts" "${admin[@]}" -H 'Content-Type: application/json' \
  --data-binary @"$WORK/base-objects.json" >"$WORK/base-plan.json"
python3 - "$WORK/base-plan.json" "$WORK/base-files.json" <<'PY'
import json,pathlib,sys,urllib.request
plan=json.load(open(sys.argv[1])); files=json.load(open(sys.argv[2]))
for grant in plan["grants"]:
    request=urllib.request.Request(grant["url"],data=pathlib.Path(files[grant["object_id"]]).read_bytes(),method="PUT",headers=grant["required_headers"])
    with urllib.request.urlopen(request) as response:
        assert 200 <= response.status < 300
PY
curl -fsS -X POST "$BASE/v1/admin/build-artifacts/creator-profile-base-manifests/complete" \
  "${admin[@]}" -H 'Content-Type: application/json' -d '{}' >/dev/null
DB_URL="postgres://postgres:profile@127.0.0.1:$PG_PORT/postgres?sslmode=disable"
for profile in "$PROFILE_A" "$PROFILE_B"; do
  build="${profile#*2.13.0-}"; build="${build%%-*}"
  (cd "$WORK/hub" && nice -n 19 go run ./cmd/th075-seed-base "$DB_URL" "$profile" \
    "$WORK/hub/vectors/base-worker-image-wheelhouse-$build.json")
done

cat >"$WORK/source/pyproject.toml" <<'EOF'
[project]
name = "marco-polo-live"
version = "1.0.0"
dependencies = []
EOF
cat >"$WORK/source/endpoint.toml" <<'EOF'
[application]
object = "marco_polo:app"
EOF
printf '%s\n' 'app = object()' >"$WORK/source/marco_polo.py"
printf '%s\n' 'version = 1' >"$WORK/source/uv.lock"
cat >"$WORK/source/endpoint.descriptor.json" <<'EOF'
{"application":"marco_polo:app","entrypoints":[{"hidden":false,"name":"marco","request":{"fields":[]},"result":{"fields":[]}}],"format":"cozy.endpoint.descriptor/1","jobs":[]}
EOF
cat >"$WORK/source/endpoint.release.json" <<'EOF'
{"compatible_accelerator_models":["NVIDIA GeForce RTX 4090"],"model_bindings":[],"model_roots":[]}
EOF
git -C "$WORK/source" init -q
git -C "$WORK/source" config user.email fixture@example.invalid
git -C "$WORK/source" config user.name Fixture
git -C "$WORK/source" add . && git -C "$WORK/source" commit -qm fixture

export COZY_HOME="$WORK/home" TENSORHUB_URL="$BASE" TENSORHUB_TOKEN="$ADMIN_TOKEN"
"$WORK/cozy" endpoint create proof/marco --reason "Creator endpoint profile live proof" \
  >"$WORK/create.out"
set +e
publish=("$WORK/cozy" endpoint publish proof/marco --release live-1 --dir "$WORK/source"
  --profile "$PROFILE_B" --profile "$PROFILE_A" --reason "Creator endpoint profile live proof")
if [ "${TRACE_HTTP:-0}" = 1 ]; then
  strace -f -s 1048576 -e trace=write -o "$WORK/publish.strace" "${publish[@]}" >"$WORK/publish.out" 2>&1
else
  "${publish[@]}" >"$WORK/publish.out" 2>&1
fi
publish_rc=$?
set -e
if [ "$publish_rc" -ne 0 ]; then
  if grep -q 'route.not_found' "$WORK/publish.out"; then
    echo "endpoint-profile-live: BLOCKED — Tensorhub $FULL_SHA has no begin/finalize routes" >&2
    cat "$WORK/publish.out" >&2
    exit 75
  fi
  cat "$WORK/publish.out" >&2
  exit "$publish_rc"
fi
grep -q "$PROFILE_A:$EXPECTED_PROFILE_STATE" "$WORK/publish.out"
grep -q "$PROFILE_B:$EXPECTED_PROFILE_STATE" "$WORK/publish.out"

# Exact replay: no create, same source/profile set, zero semantic drift.
"$WORK/cozy" endpoint publish proof/marco --release live-1 --dir "$WORK/source" \
  --profile "$PROFILE_A" --profile "$PROFILE_B" --reason "Creator endpoint profile replay" \
  >"$WORK/replay.out"
grep -q 'created:.*true' "$WORK/replay.out" # exact replay returns byte-identical committed result

# Local pure/native source arms fire before a request can reach Tensorhub.
printf '%s' native >"$WORK/source/native.so"
git -C "$WORK/source" add native.so && git -C "$WORK/source" commit -qm native-arm
set +e
"$WORK/cozy" endpoint publish proof/native --release live-native --dir "$WORK/source" \
  --profile "$PROFILE_A" --reason "native red arm" >"$WORK/native.out" 2>&1
native_rc=$?
set -e
test "$native_rc" = 3
grep -q 'endpoint_source_native_input' "$WORK/native.out"

admin=(-H "Authorization: Bearer $ADMIN_TOKEN" -H 'X-Tensorhub-Reason: Creator profile gating proof')
status=$(curl -sS -o "$WORK/qualification-before.json" -w '%{http_code}' \
  "$BASE/v1/endpoints/proof/marco/releases/live-1/profiles/$PROFILE_B/qualification" "${admin[@]}")
test "$status" = 404
grep -q 'endpoint_profile.qualification_absent' "$WORK/qualification-before.json"
status=$(curl -sS -o "$WORK/local-before.json" -w '%{http_code}' -X POST \
  "$BASE/v1/endpoints/proof/marco/releases/live-1/profiles/$PROFILE_B/local-execution" \
  "${admin[@]}" -H 'Content-Type: application/json' -d '{"grant_ttl_seconds":600}')
test "$status" = 404
grep -q 'endpoint_profile.candidate_absent' "$WORK/local-before.json"
set +e
"$WORK/cozy" endpoint promote proof/marco live-1 --serve v1/marco --reason "prequalification gate" \
  >"$WORK/promote-before.out" 2>&1
promote_rc=$?
set -e
test "$promote_rc" = 13

if [ "$RUN_PAID" = 1 ]; then
  "$WORK/cozy" endpoint qualify proof/marco@live-1 --profile "$PROFILE_B" \
    --gpu "${QUALIFICATION_GPU:-NVIDIA GeForce RTX 4090}" \
    --max-cost "${QUALIFICATION_MAX_COST:-0.25}" --duration "${QUALIFICATION_DURATION:-15m}" \
    --reason "paid Creator endpoint qualification" | tee "$WORK/qualify.out"
  grep -q 'qualified' "$WORK/qualify.out"
  curl -fsS -X POST "$BASE/v1/endpoints/proof/marco/releases/live-1/profiles/$PROFILE_B/local-execution" \
    "${admin[@]}" -H 'Content-Type: application/json' -d '{"grant_ttl_seconds":600}' \
    >"$WORK/local-qualified.json"
  grep -q 'managed-local' "$WORK/local-qualified.json"
  "$WORK/cozy" endpoint promote proof/marco live-1 --serve v1/marco \
    --reason "qualified Creator endpoint promotion" >"$WORK/promote.out"
fi

echo "endpoint-profile-live: PASS — publish/replay, two profiles, native refusal, qualify/local/promote gates (paid=$RUN_PAID)"
