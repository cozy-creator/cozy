package main

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	envAttemptID     = "COZY_ACQUISITION_ATTEMPT_ID"
	envReceiptKey    = "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL"
	envLeaseExpiry   = "COZY_RENTAL_LEASE_EXPIRY_UNIX"
	envWorkerPort    = "COZY_WORKER_INTERNAL_PORT"
	envMediaPort     = "COZY_MEDIA_INTERNAL_PORT"
	envTokenHashes   = "COZY_RENTER_TOKEN_SHA256_JSON"
	maxTokenHashes   = 16
	maxIdentityBytes = 256
)

// allowedCozyEnv is the CLOSED pod environment contract, consumer half. Tensorhub's
// internal/podenv renders exactly these six names and its base-worker-image recipe refuses
// to compile an image whose pinned bootstrap admits a different set, so an addition here
// is a contract change on both sides or it is a build failure.
//
// cl-036 retired the six COZY_PROVISION_{SPEC,BUNDLE}_{DIGEST,URL,LENGTH} grants with the
// documents they fetched, and admitted three identity facts those documents had been
// smuggling. Two of them are gone again: COZY_ACQUISITION_ATTEMPT_ORDINAL and
// COZY_RENTAL_ID were pure transit — validated here, forwarded to the adapter, echoed into
// the readiness receipt, and compared by the hub's binder against the same attempt row the
// receipt HMAC key came from. The key is minted per attempt, so a receipt that verifies has
// ALREADY proven which attempt it belongs to; the echo proved nothing after it.
// COZY_ACQUISITION_ATTEMPT_ID stays because it is not transit: the adapter spends it as the
// worker's `--worker-id`, which is the identity a Claim is addressed to and a Fault is
// attributed to on the rev-2 wire.
//
// COZY_RENTAL_LEASE_EXPIRY_UNIX is the old COZY_BOOTSTRAP_RECEIPT_DEADLINE_UNIX. The name
// lied: the value is `rental start + duration cap`, and state.go mints the pod's serving
// certificate against it. A "receipt deadline" invites being shortened to a plausible boot
// timeout, which would expire the cert mid-rental.
var allowedCozyEnv = map[string]bool{
	envAttemptID: true, envReceiptKey: true, envLeaseExpiry: true,
	envWorkerPort: true, envMediaPort: true, envTokenHashes: true,
}

type config struct {
	attemptID   string
	receipt     *attemptKey
	leaseExpiry time.Time
	workerPort  uint16
	mediaPort   uint16
	tokenHashes []string
}

func parseConfig() (config, error) {
	var out config
	for _, entry := range os.Environ() {
		name, _, _ := strings.Cut(entry, "=")
		if strings.HasPrefix(name, "COZY_") && !allowedCozyEnv[name] {
			return out, fmt.Errorf("unknown Cozy environment variable %s", name)
		}
	}
	var err error
	if out.attemptID, err = parseIdentity(envAttemptID); err != nil {
		return out, err
	}
	keyText, err := requiredEnv(envReceiptKey)
	if err != nil {
		return out, err
	}
	hmacKey, err := base64.RawURLEncoding.Strict().DecodeString(keyText)
	if err != nil || len(hmacKey) != 32 || base64.RawURLEncoding.EncodeToString(hmacKey) != keyText {
		return out, fmt.Errorf("%s must be unpadded base64url encoding exactly 32 bytes", envReceiptKey)
	}
	// The raw key never becomes a field. What the rest of the program receives is the
	// ONE-SHOT attemptKey (receipt.go): it signs the readiness envelope once, zeroes
	// itself in the same call, and errors on a second caller. Nothing outside this file
	// and receipt.go can name the key at all — the fence holds that.
	out.receipt = newAttemptKey(hmacKey)
	// THE FIRST HALF OF THE KEY WIPE (cl-036). This process is PID 1 AND an HTTP parser
	// now, so the readiness key spends the shortest life this program can give it: its
	// environment slot goes the instant the bytes are decoded, which is before any
	// listener is bound, any child is exec'd, or any request byte is read. The second
	// half — zeroing the bytes themselves — is the one-shot attemptKey in receipt.go.
	// Children never saw this name anyway (childEnvironment is a five-name allowlist);
	// what this closes is /proc/1/environ and every later reader inside this process.
	if err := os.Unsetenv(envReceiptKey); err != nil {
		return out, fmt.Errorf("cannot unset %s after decoding it: %w", envReceiptKey, err)
	}
	expiryText, err := requiredEnv(envLeaseExpiry)
	if err != nil {
		return out, err
	}
	expiryUnix, err := strconv.ParseInt(expiryText, 10, 64)
	if err != nil || strconv.FormatInt(expiryUnix, 10) != expiryText {
		return out, fmt.Errorf("%s must be one canonical decimal Unix second", envLeaseExpiry)
	}
	out.leaseExpiry = time.Unix(expiryUnix, 0)
	if out.workerPort, err = parsePort(envWorkerPort); err != nil {
		return out, err
	}
	if out.mediaPort, err = parsePort(envMediaPort); err != nil {
		return out, err
	}
	if out.workerPort == out.mediaPort {
		return out, fmt.Errorf("worker and media ports must be distinct")
	}
	hashesText, err := requiredEnv(envTokenHashes)
	if err != nil {
		return out, err
	}
	if err := json.Unmarshal([]byte(hashesText), &out.tokenHashes); err != nil {
		return out, fmt.Errorf("%s must be a JSON string array", envTokenHashes)
	}
	if err := validateTokenHashes(out.tokenHashes); err != nil {
		return out, err
	}
	return out, nil
}

// parseIdentity reads one bounded opaque id. Bootstrap never interprets the attempt id; it
// hands it to the adapter, which spends it as the worker's `--worker-id` — the identity a
// RecordOwner's Claim is addressed to. The bound is the adapter's own
// (cozy_runtime.internal.config.bounded) so a value that boots here cannot be refused one
// exec later.
func parseIdentity(name string) (string, error) {
	value, err := requiredEnv(name)
	if err != nil {
		return "", err
	}
	if len(value) > maxIdentityBytes || strings.TrimSpace(value) != value {
		return "", fmt.Errorf("%s must be one untrimmed identifier of at most %d bytes",
			name, maxIdentityBytes)
	}
	return value, nil
}

func parsePort(name string) (uint16, error) {
	text, err := requiredEnv(name)
	if err != nil {
		return 0, err
	}
	port, err := strconv.ParseUint(text, 10, 16)
	if err != nil || port == 0 || strconv.FormatUint(port, 10) != text {
		return 0, fmt.Errorf("%s must be one canonical decimal TCP port", name)
	}
	return uint16(port), nil
}

func validateTokenHashes(hashes []string) error {
	if len(hashes) == 0 || len(hashes) > maxTokenHashes {
		return fmt.Errorf("%s must contain between 1 and %d hashes", envTokenHashes, maxTokenHashes)
	}
	for i, hash := range hashes {
		if len(hash) != sha256.Size*2 || hash != strings.ToLower(hash) {
			return fmt.Errorf("%s entries must be 64 lowercase hex", envTokenHashes)
		}
		if _, err := hex.DecodeString(hash); err != nil {
			return fmt.Errorf("%s entries must be 64 lowercase hex", envTokenHashes)
		}
		if i > 0 && hashes[i-1] >= hash {
			return fmt.Errorf("%s must be sorted and unique", envTokenHashes)
		}
	}
	return nil
}

func requiredEnv(name string) (string, error) {
	value, ok := os.LookupEnv(name)
	if !ok || value == "" {
		return "", fmt.Errorf("required environment variable %s is absent", name)
	}
	return value, nil
}

// childEnvironment is the exact set of names a child inherits.
//
// Provider/image environments routinely contain AWS, Hugging Face, and RunPod
// credentials. Children need only process locale/timezone and the standard TLS trust
// overrides. Class-D base inherit list — tracker-v2/spawn-allowlists.md (#616.d) is the
// authority; keep equal to that row. PATH is deliberately absent: the adapter is exec'd by
// absolute path and spawns nothing PATH-resolved. There is exactly one child since cl-036
// merged the media plane into this process; the allowlist is unchanged by that.
func childEnvironment() []string {
	allowed := map[string]bool{
		"LANG": true, "LC_ALL": true, "TZ": true,
		"SSL_CERT_FILE": true, "SSL_CERT_DIR": true,
	}
	env := make([]string, 0, len(allowed))
	for _, entry := range os.Environ() {
		name, _, _ := strings.Cut(entry, "=")
		if allowed[name] {
			env = append(env, entry)
		}
	}
	return env
}
