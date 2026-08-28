#!/usr/bin/env bash
# The pod's hardening proof, against the REAL binary (no test suite: decisions.md #160).
# Verification here is running the real thing.
#
# cl-036 merged `cozy-bootstrap` and `cozy-media` into ONE binary, `pod-supervisor`, so every arm
# below now drives one process. Two consequences shaped this file:
#
#   * The media plane has no argv left. Its grant used to be `--token-sha256` on a second
#     binary's command line; it is now handed over in memory, so the nine adversary digest
#     sets are driven HERE through `COZY_RENTER_TOKEN_SHA256_JSON` against the real
#     entrypoint — one level earlier and one level harder, since a set the pod cannot
#     authenticate is now a pod that does not boot at all. The serving half of those arms
#     (a matching bearer admitted, every other request refused, the receipt served
#     byte-identically) moved to `go test ./internal/live -run TestPodMedia`, where it runs
#     against a real socket and a real TLS leaf instead of a plaintext loopback stand-in.
#   * The merge's two prices are arms of their own: supervision must stay out of the
#     request path, and the readiness HMAC key must be wiped once the envelope is sealed.
#
# Verifies:
#   1. Ambient RunPod provider identity (RUNPOD_*) never enters pod config.
#   2. Unknown COZY_* variables are refused by name: provider-identity aliases, the six
#      retired COZY_PROVISION_* grants, the two identity facts the env diet cut, and the old
#      spelling of the lease expiry.
#   3. The child-process environment allowlist is exactly the reviewed set, so
#      AWS/HF/RunPod/Cozy credentials never cross the child boundary.
#   4. The receipt ceiling is IMPORTED from internal/mediawire, not restated — the two
#      silent copies cl-031 found were in different repos and agreed by luck.
#   5. The renter token digests reach the media plane in memory: no file, no argv.
#   6. The readiness HMAC key is wiped — its environment slot where it is decoded, its
#      bytes where the envelope is sealed.
#   7. Supervision is not reachable from a request path: everything that execs, signals or
#      reaps is in one file of package main, and none of it is in internal/podmedia.
#   8. The entrypoint takes no arguments, and a grant it cannot authenticate is a pod that
#      refuses to boot rather than one that boots unauthenticated.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
nice -n 19 go build -o "$work/pod-supervisor" "$root/cmd/pod-supervisor"

pass=0
fail=0
ok() { printf '  PASS %s\n' "$*"; pass=$((pass + 1)); }
bad() { printf '  FAIL %s\n' "$*"; fail=$((fail + 1)); }

good_hashes="[\"$(printf 'a%.0s' $(seq 64))\"]"
base_env=(
  "PATH=/usr/bin:/bin" "HOME=$work"
  "COZY_ACQUISITION_ATTEMPT_ID=ra-hardening-proof"
  "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL=$(printf 'A%.0s' $(seq 43))"
  "COZY_RENTAL_LEASE_EXPIRY_UNIX=4102444800"
  "COZY_WORKER_INTERNAL_PORT=43100"
  "COZY_MEDIA_INTERNAL_PORT=43101"
  "COZY_RENTER_TOKEN_SHA256_JSON=$good_hashes"
)

run() { env -i "$@" timeout 15 "$work/pod-supervisor" 2>&1 || true; }

out=$(run "${base_env[@]}" \
  "RUNPOD_POD_ID=provider-native-id-must-not-enter-guest-truth" \
  "RUNPOD_PUBLIC_IP=203.0.113.10")
if grep -q 'unknown Cozy environment variable' <<<"$out"; then
  bad "ambient RUNPOD_* facts changed pod config: $out"
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

# The same rule for the two identity facts that were pure transit, and for the name
# the lease expiry used to answer to. COZY_BOOTSTRAP_RECEIPT_DEADLINE_UNIX is the dangerous
# one — it was never a receipt deadline, it is the life of the certificate this pod serves
# under — so a hub still emitting it is told by name, never silently handed a pod that then
# refuses for a missing lease it thinks it supplied.
for cut in COZY_ACQUISITION_ATTEMPT_ORDINAL COZY_RENTAL_ID COZY_BOOTSTRAP_RECEIPT_DEADLINE_UNIX; do
  out=$(run "${base_env[@]}" "$cut=whatever")
  if grep -q "unknown Cozy environment variable $cut" <<<"$out"; then
    ok "retired name $cut is refused"
  else
    bad "retired name $cut was admitted: $out"
  fi
done

# The rename is not a second spelling: the pod names the lease expiry ONCE, and its absence
# is a boot failure that says so.
out=$(env -i "PATH=/usr/bin:/bin" "HOME=$work" \
  "COZY_ACQUISITION_ATTEMPT_ID=ra-hardening-proof" \
  "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL=$(printf 'A%.0s' $(seq 43))" \
  "COZY_WORKER_INTERNAL_PORT=43100" "COZY_MEDIA_INTERNAL_PORT=43101" \
  "COZY_RENTER_TOKEN_SHA256_JSON=$good_hashes" timeout 15 "$work/pod-supervisor" 2>&1 || true)
if grep -q 'required environment variable COZY_RENTAL_LEASE_EXPIRY_UNIX is absent' <<<"$out"; then
  ok "the lease expiry is required under its own name ($(head -1 <<<"$out"))"
else
  bad "an absent lease expiry was not refused by name: $out"
fi

# The certificate is minted against the LEASE and nothing else. A boot-shaped timeout here
# would expire the pod's serving cert mid-rental, so the value has exactly one reader.
if grep -q 'mintCertificate(cfg.leaseExpiry)' "$root/cmd/pod-supervisor/state.go" &&
    grep -q 'NotAfter:     leaseExpiry.Add(time.Hour)' "$root/cmd/pod-supervisor/state.go"; then
  ok "the pod's serving certificate expires with the rental lease, not with a boot timeout"
else
  bad "the TLS leaf is no longer minted against the rental lease expiry"
fi

got=$(awk '/^func childEnvironment/,/^}$/' "$root/cmd/pod-supervisor/config.go" |
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
if grep -rhvE '^[[:space:]]*//' "$root"/cmd/pod-supervisor/*.go "$root"/internal/podmedia/*.go |
    grep -qE '(64[[:space:]]*<<[[:space:]]*10|maxReceiptEnvelopeSize)'; then
  bad "the receipt ceiling is restated in the pod binary instead of imported from internal/mediawire"
elif grep -q 'mediawire.MaxReceiptBytes' "$root/cmd/pod-supervisor/receipt.go"; then
  ok "the receipt ceiling is the media plane's own published constant, imported"
else
  bad "the pod bounds the readiness envelope by neither the published constant nor a literal"
fi

# The pod holds DIGESTS, never a token, and it holds them in memory. Nothing may write a
# credential-shaped file for the media plane to poll (that file had one writer and no
# updater, so its reload/epoch machinery could never fire), and since the merge there is no
# command line to put them on either — which is one fewer place a process list can leak.
if grep -qE 'atomicWrite\((token|hash)[A-Za-z]*Path' "$root"/cmd/pod-supervisor/*.go; then
  bad "the pod writes a token-hash file again — the digests are a launch grant"
elif grep -qE '"--token-sha256"' "$root"/cmd/pod-supervisor/*.go; then
  bad "the pod puts the renter digests on a command line — the merge removed that surface"
elif grep -q 'TokenHashes:      mediaGrant(cfg.tokenHashes)' "$root/cmd/pod-supervisor/supervise.go"; then
  ok "renter token digests reach the media plane in memory — no file, no argv"
else
  bad "the pod hands the media plane no token digests at all"
fi

# THE MERGE'S FIRST PRICE: the readiness HMAC key is wiped. Its environment slot goes where
# it is decoded — before any listener binds or any child exists — and its bytes go in the
# one call that seals the envelope. `cmd/pod-supervisor` is the whole search space; the fence
# additionally refuses to let the key be NAMED anywhere but these two files.
if grep -q 'os.Unsetenv(envReceiptKey)' "$root/cmd/pod-supervisor/config.go"; then
  ok "the readiness HMAC key's environment slot is unset where the key is decoded"
else
  bad "the readiness HMAC key stays in the environment after it is decoded"
fi
if grep -Pzoq 'k\.raw\[i\] = 0[\s\S]{0,80}k\.raw = nil' "$root/cmd/pod-supervisor/receipt.go"; then
  ok "the readiness HMAC key's bytes are zeroed and dropped where the envelope is sealed"
else
  bad "the readiness HMAC key survives the envelope it signed"
fi
if grep -q 'the readiness key was wiped' "$root/cmd/pod-supervisor/receipt.go"; then
  ok "a second signature is an error, not a second use of a wiped key"
else
  bad "the attempt key is not one-shot: nothing refuses a second signing"
fi

# THE MERGE'S SECOND PRICE: supervision is not reachable from a request path. The handlers
# live in internal/podmedia, which holds none of PID 1's privileges; everything that
# spawns, signals or reaps is in one file of package main, which nothing can import.
if grep -rlE '(exec\.Command|signal\.Notify|syscall\.SIG|\.Process\b|os/exec)' \
    "$root"/internal/podmedia/*.go >/dev/null 2>&1; then
  bad "the pod's request path holds a supervision privilege: $(grep -rlE '(exec\.Command|signal\.Notify|syscall\.SIG|\.Process\b|os/exec)' "$root"/internal/podmedia/*.go)"
else
  ok "the request path (internal/podmedia) holds none of PID 1's privileges"
fi
spawners=$(grep -rlE '(exec\.Command|\.Process\.(Signal|Kill)|cmd\.Wait)' "$root"/cmd/pod-supervisor/*.go |
  xargs -n1 basename | sort | paste -sd, -)
if [ "$spawners" = "supervise.go" ]; then
  ok "everything that execs, signals or reaps is in one file (supervise.go)"
else
  bad "supervision is spread across [$spawners] instead of supervise.go alone"
fi
if [ -d "$root/cmd/cozy-media" ] || [ -d "$root/cmd/cozy-bootstrap" ]; then
  bad "the pod is still two binaries — cl-036 merged them into cmd/pod-supervisor"
else
  ok "the pod is ONE binary: cmd/pod-supervisor, with no second main package beside it"
fi

# The entrypoint is PID 1 of a container. It takes no arguments and says so.
out=$(env -i "${base_env[@]}" timeout 15 "$work/pod-supervisor" --anything 2>&1 || true)
if grep -q 'this entrypoint takes no arguments' <<<"$out"; then
  ok "the entrypoint refuses arguments ($(head -1 <<<"$out"))"
else
  bad "the entrypoint accepted an argument: $out"
fi

# FAIL CLOSED on the credential grant. Each of these must REFUSE TO BOOT — never start the
# media plane and admit everybody, which is the failure a missing credential set invites.
# Since the merge the plane binds inside PID 1 before any child exists, so refusing here
# means nothing at all is running behind it.
sixty_four=$(printf 'a%.0s' $(seq 64))
closed() { # closed <label> <COZY_RENTER_TOKEN_SHA256_JSON value>
  local label=$1 value=$2
  local out
  out=$(env -i "${base_env[@]}" "COZY_RENTER_TOKEN_SHA256_JSON=$value" \
    timeout 15 "$work/pod-supervisor" 2>&1 || true)
  if grep -q 'COZY_RENTER_TOKEN_SHA256_JSON' <<<"$out"; then
    ok "$label: the pod refuses to boot ($(head -1 <<<"$out"))"
  else
    bad "$label: the pod did not refuse by name: $out"
  fi
}

closed "an empty digest set" '[]'
closed "a set that is not JSON" 'sha256:'"$sixty_four"
closed "a JSON object instead of an array" '{"hash":"'"$sixty_four"'"}'
closed "a non-hex digest" '["not-hex"]'
closed "a prefixed digest" '["sha256:'"$sixty_four"'"]'
closed "a truncated digest" '["'"${sixty_four:0:63}"'"]'
closed "an uppercase digest" '["'"$(printf '%s' "$sixty_four" | tr 'a-f' 'A-F')"'"]'
closed "an unsorted set" '["'"${sixty_four//a/b}"'","'"$sixty_four"'"]'
closed "a duplicated digest" '["'"$sixty_four"'","'"$sixty_four"'"]'
closed "a 17-digest set" \
  "$(printf '['; for i in $(seq 0 16); do
      printf '"%x%s"' "$i" "${sixty_four:1}"; [ "$i" -lt 16 ] && printf ','; done; printf ']')"

# An ABSENT required grant is refused by name too: a pod that defaults a credential set is
# a pod that authenticates somebody nobody provisioned.
out=$(env -i "PATH=/usr/bin:/bin" "HOME=$work" \
  "COZY_ACQUISITION_ATTEMPT_ID=ra-hardening-proof" timeout 15 "$work/pod-supervisor" 2>&1 || true)
if grep -q 'required environment variable COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL is absent' <<<"$out"; then
  ok "an incomplete environment is refused by the missing name ($(head -1 <<<"$out"))"
else
  bad "an incomplete environment was not refused by name: $out"
fi

# The complete environment gets PAST the grant and dies laying out the pod's own
# filesystem, which only exists inside the image. That is the arm's point: everything this
# proof can check without being PID 1 of that container has been checked.
out=$(run "${base_env[@]}")
if grep -q 'create private runtime directory' <<<"$out"; then
  ok "a complete grant is admitted and boot proceeds to the pod's own layout ($(head -1 <<<"$out"))"
else
  bad "a complete grant did not reach state preparation: $out"
fi

printf 'hardening proof: %d passed, %d failed\n' "$pass" "$fail"
exit "$fail"
