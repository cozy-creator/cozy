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
#   5. The pod writes NO token-hash file: the renter token digests reach cozy-media as a
#      launch grant, and cozy-media is FAIL-CLOSED on that grant — a valid set admits only
#      the matching bearer, and an absent, empty, or malformed set refuses to serve at all.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
nice -n 19 go build -o "$work/cozy-bootstrap" "$root/cmd/cozy-bootstrap"
nice -n 19 go build -o "$work/cozy-media" "$root/cmd/cozy-media"

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

# The pod holds DIGESTS, never a token, and it holds them in memory. Nothing in the
# supervisor may write a credential-shaped file for cozy-media to poll: that file had one
# writer and no updater, so its reload/epoch machinery could never fire.
if grep -qE 'atomicWrite\((token|hash)[A-Za-z]*Path' "$root"/cmd/cozy-bootstrap/*.go; then
  bad "cozy-bootstrap writes a token-hash file again — the digests are a launch grant"
elif grep -q '"--token-sha256", mediaTokenGrant' "$root/cmd/cozy-bootstrap/run.go"; then
  ok "renter token digests reach cozy-media as a launch grant, not a file"
else
  bad "cozy-bootstrap hands cozy-media no token digests at all"
fi

# cozy-media, run for real. The token is minted HERE and never leaves this script: the
# server is given only its sha256, which is the whole point of the grant's shape.
token=$(printf 'renter-token-hardening-proof')
digest=$(printf '%s' "$token" | sha256sum | cut -d' ' -f1)

media() { # media <name> <token-sha256 args...>
  local name=$1; shift
  local dir="$work/media-$name"
  mkdir -p "$dir"
  "$work/cozy-media" --listen 127.0.0.1:0 --root "$dir/root" --out "$dir" "$@" \
    >"$dir/log" 2>&1 &
  media_pid=$!
  media_addr=""
  for _ in $(seq 1 100); do
    if [ -s "$dir/media.addr" ]; then media_addr=$(cat "$dir/media.addr"); break; fi
    kill -0 "$media_pid" 2>/dev/null || break
    sleep 0.05
  done
  media_log="$dir/log"
}

media_stop() { kill "$media_pid" 2>/dev/null || true; wait "$media_pid" 2>/dev/null || true; }

code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

media valid --token-sha256 "sha256:$digest"
if [ -z "$media_addr" ]; then
  bad "cozy-media refused a valid digest set: $(cat "$media_log")"
else
  got=$(code -H "Authorization: Bearer $token" "http://$media_addr/v1/health")
  [ "$got" = "200" ] && ok "a valid digest set admits the matching bearer (200)" ||
    bad "the matching bearer was refused: HTTP $got"
  got=$(code -H "Authorization: Bearer not-the-renters-token" "http://$media_addr/v1/health")
  [ "$got" = "401" ] && ok "a non-matching bearer is refused (401)" ||
    bad "a non-matching bearer was admitted: HTTP $got"
  got=$(code "http://$media_addr/v1/health")
  [ "$got" = "401" ] && ok "no bearer is refused (401)" ||
    bad "an unauthenticated request was admitted: HTTP $got"
fi
media_stop

# FAIL CLOSED, three ways. Each of these must REFUSE TO SERVE — never bind and admit
# everybody, which is the failure a missing credential set invites.
closed() { # closed <name> <label> [args...]
  local name=$1 label=$2; shift 2
  media "$name" "$@"
  if [ -n "$media_addr" ]; then
    bad "$label: cozy-media bound anyway on $media_addr"
    media_stop
  else
    ok "$label: cozy-media refuses to serve ($(head -1 "$media_log"))"
  fi
}

closed absent "an absent digest set"
closed empty "an empty digest set" --token-sha256 ""
closed nothex "a non-hex digest" --token-sha256 "sha256:not-hex"
closed unprefixed "an unprefixed digest" --token-sha256 "$digest"
closed short "a truncated digest" --token-sha256 "sha256:${digest:0:63}"
closed uppercase "an uppercase digest" --token-sha256 "sha256:$(printf '%s' "$digest" | tr 'a-f' 'A-F')"
closed unsorted "an unsorted set" --token-sha256 "sha256:${digest//?/b},sha256:${digest//?/a}"
closed duplicate "a duplicated digest" --token-sha256 "sha256:$digest,sha256:$digest"
closed oversized "a 17-digest set" \
  --token-sha256 "$(for i in $(seq 0 16); do printf 'sha256:%s%s,' "$(printf '%x' $i)" "${digest:1}"; done | sed 's/,$//')"

printf 'hardening proof: %d passed, %d failed\n' "$pass" "$fail"
exit "$fail"
