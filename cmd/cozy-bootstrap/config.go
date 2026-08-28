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
	envAttemptID       = "COZY_ACQUISITION_ATTEMPT_ID"
	envAttemptOrdinal  = "COZY_ACQUISITION_ATTEMPT_ORDINAL"
	envRentalID        = "COZY_RENTAL_ID"
	envReceiptKey      = "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL"
	envReceiptDeadline = "COZY_BOOTSTRAP_RECEIPT_DEADLINE_UNIX"
	envWorkerPort      = "COZY_WORKER_INTERNAL_PORT"
	envMediaPort       = "COZY_MEDIA_INTERNAL_PORT"
	envTokenHashes     = "COZY_RENTER_TOKEN_SHA256_JSON"
	maxTokenHashes     = 16
	maxIdentityBytes   = 256
	maxAttemptOrdinal  = int64(1) << 31
)

// allowedCozyEnv is the CLOSED pod environment contract, consumer half. Tensorhub's
// internal/podenv renders exactly these eight names and its build/substrate recipe refuses
// to compile an image whose pinned bootstrap admits a different set, so an addition here
// is a contract change on both sides or it is a build failure. cl-036 retired the six
// COZY_PROVISION_{SPEC,BUNDLE}_{DIGEST,URL,LENGTH} grants with the documents they fetched
// and added the three identity facts those documents were smuggling.
var allowedCozyEnv = map[string]bool{
	envAttemptID: true, envAttemptOrdinal: true, envRentalID: true,
	envReceiptKey: true, envReceiptDeadline: true, envWorkerPort: true, envMediaPort: true,
	envTokenHashes: true,
}

type config struct {
	attemptID      string
	attemptOrdinal int64
	rentalID       string
	hmacKey        []byte
	deadline       time.Time
	workerPort     uint16
	mediaPort      uint16
	tokenHashes    []string
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
	if out.attemptOrdinal, err = parseOrdinal(envAttemptOrdinal); err != nil {
		return out, err
	}
	if out.rentalID, err = parseIdentity(envRentalID); err != nil {
		return out, err
	}
	keyText, err := requiredEnv(envReceiptKey)
	if err != nil {
		return out, err
	}
	out.hmacKey, err = base64.RawURLEncoding.Strict().DecodeString(keyText)
	if err != nil || len(out.hmacKey) != 32 || base64.RawURLEncoding.EncodeToString(out.hmacKey) != keyText {
		return out, fmt.Errorf("%s must be unpadded base64url encoding exactly 32 bytes", envReceiptKey)
	}
	deadlineText, err := requiredEnv(envReceiptDeadline)
	if err != nil {
		return out, err
	}
	deadlineUnix, err := strconv.ParseInt(deadlineText, 10, 64)
	if err != nil || strconv.FormatInt(deadlineUnix, 10) != deadlineText {
		return out, fmt.Errorf("%s must be one canonical decimal Unix second", envReceiptDeadline)
	}
	out.deadline = time.Unix(deadlineUnix, 0)
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

// parseIdentity reads one bounded opaque id. Bootstrap never interprets an attempt or
// rental id; it hands it to the adapter, which echoes it in the readiness receipt so
// tensorhub's binder can join the receipt to the attempt row it froze. The bound is the
// adapter's own (cozy_runtime.internal.config.bounded) so a value that boots here cannot
// be refused one exec later.
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

func parseOrdinal(name string) (int64, error) {
	text, err := requiredEnv(name)
	if err != nil {
		return 0, err
	}
	ordinal, err := strconv.ParseInt(text, 10, 64)
	if err != nil || ordinal <= 0 || ordinal > maxAttemptOrdinal ||
		strconv.FormatInt(ordinal, 10) != text {
		return 0, fmt.Errorf("%s must be one positive canonical decimal no greater than %d",
			name, maxAttemptOrdinal)
	}
	return ordinal, nil
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
// authority; keep equal to that row. PATH is deliberately absent: cozy-media and the
// adapter are exec'd by absolute path and spawn nothing PATH-resolved.
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
