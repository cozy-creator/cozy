#!/usr/bin/env bash
# Hosted dev-build test command. Its existing runtime_ref/tensorfs_run inputs provide the
# exact Runtime/native wheels and tensorhub_ref checks out its Hub source. This script adds
# the real ordinary Hub/MinIO fixture inside dev-build's delegated cgroup. It never uses a
# rental or receives the checkout action's credential.
set -euo pipefail
test "${GITHUB_ACTIONS:-}" = true
: "${AUTHKIT_TEST_DATABASE_URL:?dev-build supplies a fresh PostgreSQL fixture}"
: "${COZY_TEST_RUNTIME_WHEEL:?set runtime_ref to the exact qualified Runtime}"
: "${COZY_TEST_TENSORFS_WHEEL:?set tensorfs_run to the qualified native wheel run}"
cozy_sha=$(git rev-parse HEAD)
hub_sha=da8effea817f2ec90fac19a4f553616a5ef73a15
hubdir=$(cd "${1:?pass the tensorhub_ref checkout path}" && pwd)
test "$(git -C "$hubdir" rev-parse HEAD)" = "$hub_sha"
python3 - <<'PY'
import hashlib
import os
import zipfile

native = os.environ["COZY_TEST_TENSORFS_WHEEL"]
with open(native, "rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == "569f8df74781f9c46ca6dc831c0f39bd3954c73e9134ca3c7f57c9287b9054cf"
with zipfile.ZipFile(os.environ["COZY_TEST_RUNTIME_WHEEL"]) as wheel:
    provenance = wheel.read("cozy_runtime/_build_provenance.py").decode()
    assert 'COMMIT = "9ef3bd8b9a337b602b8932d6c9a7b8563dc97859"' in provenance
print("Verified Runtime9ef and native TensorFS f060 wheels")
PY
printf 'Ordinary CLI proof Creator: %s; Hub: %s\n' "$cozy_sha" "$hub_sha"
cd "$hubdir"
# API fixture only: app.FS explicitly permits a binary without built browser pages.
go build -o dist/tensorhub .
export PROOF_DATABASE_URL="$AUTHKIT_TEST_DATABASE_URL"
scripts/live-hub-client.sh "$cozy_sha" -p cozy --test run_local \
  runtime_checkpoint_uploads -- --test-threads=1
