#!/usr/bin/env bash
# The pod supervisor's hardening proof, against the REAL binary (no test suite:
# decisions.md #160). Verification here is running the real thing.
#
# Verifies:
#   1. Ambient RunPod provider identity (RUNPOD_*) never enters bootstrap config.
#   2. Unknown COZY_* variables (provider-identity aliases, and the six retired
#      COZY_PROVISION_* grants) are refused by name.
#   3. The child-process environment allowlist is exactly the reviewed set, so
#      AWS/HF/RunPod/Cozy credentials never cross the child boundary.
#   4. The receipt ceiling is IMPORTED from internal/mediawire, not restated — the two
#      silent copies cl-031 found were in different repos and agreed by luck.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
nice -n 19 go build -o "$work/cozy-bootstrap" "$root/cmd/cozy-bootstrap"

pass=0
fail=0
ok() { printf '  PASS %s\n' "$*"; pass=$((pass + 1)); }
bad() { printf '  FAIL %s\n' "$*"; fail=$((fail + 1)); }

base_env=(
  "PATH=/usr/bin:/bin" "HOME=$work"
  "COZY_ACQUISITION_ATTEMPT_ID=ra-hardening-proof"
  "COZY_ACQUISITION_ATTEMPT_ORDINAL=1"
  "COZY_RENTAL_ID=rental-hardening-proof"
  "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL=$(printf 'A%.0s' $(seq 43))"
  "COZY_BOOTSTRAP_RECEIPT_DEADLINE_UNIX=4102444800"
  "COZY_WORKER_INTERNAL_PORT=43100"
  "COZY_MEDIA_INTERNAL_PORT=43101"
  "COZY_RENTER_TOKEN_SHA256_JSON=[\"$(printf 'a%.0s' $(seq 64))\"]"
)

run() { env -i "$@" timeout 15 "$work/cozy-bootstrap" 2>&1 || true; }

out=$(run "${base_env[@]}" \
  "RUNPOD_POD_ID=provider-native-id-must-not-enter-guest-truth" \
  "RUNPOD_PUBLIC_IP=203.0.113.10")
if grep -q 'unknown Cozy environment variable' <<<"$out"; then
  bad "ambient RUNPOD_* facts changed bootstrap config: $out"
else
  ok "ambient RunPod provider identity is ignored by the env gate"
fi

out=$(run "${base_env[@]}" "COZY_PROVIDER_POD_ID=compatibility-alias-forbidden")
if grep -q 'unknown Cozy environment variable COZY_PROVIDER_POD_ID' <<<"$out"; then
  ok "provider identity alias COZY_PROVIDER_POD_ID is refused"
else
  bad "provider identity alias was admitted: $out"
fi

# cl-036: the retired boot-closure grants are refused BY NAME, not ignored. A hub still
# emitting them is a boot failure the operator can read, never a pod that silently fetches
# nothing and reports ready.
for retired in COZY_PROVISION_SPEC_DIGEST COZY_PROVISION_SPEC_URL COZY_PROVISION_SPEC_LENGTH \
  COZY_PROVISION_BUNDLE_DIGEST COZY_PROVISION_BUNDLE_URL COZY_PROVISION_BUNDLE_LENGTH; do
  out=$(run "${base_env[@]}" "$retired=whatever")
  if grep -q "unknown Cozy environment variable $retired" <<<"$out"; then
    ok "retired boot-closure grant $retired is refused"
  else
    bad "retired boot-closure grant $retired was admitted: $out"
  fi
done

got=$(awk '/^func childEnvironment/,/^}$/' "$root/cmd/cozy-bootstrap/config.go" |
  grep -oE '"[A-Z_]+"' | tr -d '"' | sort | paste -sd, -)
# Class-D base inherit list — tracker-v2/spawn-allowlists.md (#616.d).
want="LANG,LC_ALL,SSL_CERT_DIR,SSL_CERT_FILE,TZ"
if [ "$got" = "$want" ]; then
  ok "child environment allowlist is exactly the reviewed set ($want)"
else
  bad "child environment allowlist drifted: got [$got], want [$want]"
fi

# Whole-line comments are prose (the same convention scripts/fence.py uses); the hazard
# is the number being DECLARED here, not the story of it having been.
if grep -hvE '^[[:space:]]*//' "$root"/cmd/cozy-bootstrap/*.go |
    grep -qE '(64[[:space:]]*<<[[:space:]]*10|maxReceiptEnvelopeSize)'; then
  bad "the receipt ceiling is restated in cozy-bootstrap instead of imported from internal/mediawire"
elif grep -q 'mediawire.MaxReceiptBytes' "$root/cmd/cozy-bootstrap/run.go"; then
  ok "the receipt ceiling is the media plane's own published constant, imported"
else
  bad "cozy-bootstrap bounds the readiness envelope by neither the published constant nor a literal"
fi

printf 'hardening proof: %d passed, %d failed\n' "$pass" "$fail"
exit "$fail"
