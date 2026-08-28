package main

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	envProvisionSpecDigest   = "COZY_PROVISION_SPEC_DIGEST"
	envProvisionSpecURL      = "COZY_PROVISION_SPEC_URL"
	envProvisionSpecLength   = "COZY_PROVISION_SPEC_LENGTH"
	envProvisionBundleDigest = "COZY_PROVISION_BUNDLE_DIGEST"
	envProvisionBundleURL    = "COZY_PROVISION_BUNDLE_URL"
	envProvisionBundleLength = "COZY_PROVISION_BUNDLE_LENGTH"
	envReceiptKey            = "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL"
	envReceiptDeadline       = "COZY_BOOTSTRAP_RECEIPT_DEADLINE_UNIX"
	envWorkerPort            = "COZY_WORKER_INTERNAL_PORT"
	envMediaPort             = "COZY_MEDIA_INTERNAL_PORT"
	envTokenHashes           = "COZY_RENTER_TOKEN_SHA256_JSON"
	maxTokenHashes           = 16
	maxProvisionDocumentSize = int64(8 << 20)
)

var allowedCozyEnv = map[string]bool{
	envProvisionSpecDigest: true, envProvisionSpecURL: true, envProvisionSpecLength: true,
	envProvisionBundleDigest: true, envProvisionBundleURL: true, envProvisionBundleLength: true,
	envReceiptKey: true, envReceiptDeadline: true, envWorkerPort: true, envMediaPort: true,
	envTokenHashes: true,
}

type artifactGrant struct {
	kind   string
	digest string
	want   [sha256.Size]byte
	url    *url.URL
	length int64
}

type config struct {
	spec        artifactGrant
	bundle      artifactGrant
	hmacKey     []byte
	deadline    time.Time
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
	if out.spec, err = parseGrant("provision spec", envProvisionSpecDigest, envProvisionSpecURL, envProvisionSpecLength); err != nil {
		return out, err
	}
	if out.bundle, err = parseGrant("provision bundle", envProvisionBundleDigest, envProvisionBundleURL, envProvisionBundleLength); err != nil {
		return out, err
	}
	if out.spec.digest == out.bundle.digest {
		return out, fmt.Errorf("provision spec and provision bundle digests must remain distinct")
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

func parseGrant(kind, digestEnv, urlEnv, lengthEnv string) (artifactGrant, error) {
	grant := artifactGrant{kind: kind}
	var err error
	grant.digest, err = requiredEnv(digestEnv)
	if err != nil {
		return grant, err
	}
	if len(grant.digest) != len("sha256:")+sha256.Size*2 || !strings.HasPrefix(grant.digest, "sha256:") ||
		grant.digest != strings.ToLower(grant.digest) {
		return grant, fmt.Errorf("%s must be sha256:<64 lowercase hex>", digestEnv)
	}
	digestBytes, err := hex.DecodeString(strings.TrimPrefix(grant.digest, "sha256:"))
	if err != nil {
		return grant, fmt.Errorf("%s must be sha256:<64 lowercase hex>", digestEnv)
	}
	copy(grant.want[:], digestBytes)
	urlText, err := requiredEnv(urlEnv)
	if err != nil {
		return grant, err
	}
	grant.url, err = url.Parse(urlText)
	if err != nil || grant.url.Scheme != "https" || grant.url.Host == "" || grant.url.User != nil || grant.url.Fragment != "" {
		return grant, fmt.Errorf("%s must be one credential-free HTTPS URL without a fragment", urlEnv)
	}
	lengthText, err := requiredEnv(lengthEnv)
	if err != nil {
		return grant, err
	}
	grant.length, err = strconv.ParseInt(lengthText, 10, 64)
	if err != nil || grant.length <= 0 || grant.length > maxProvisionDocumentSize ||
		strconv.FormatInt(grant.length, 10) != lengthText {
		return grant, fmt.Errorf("%s must be one positive canonical decimal byte length no greater than 8 MiB", lengthEnv)
	}
	return grant, nil
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
// overrides used by exact-grant HTTPS clients. Class-D base inherit list —
// tracker-v2/spawn-allowlists.md (#616.d) is the authority; keep equal to that row. PATH
// is deliberately absent: cozy-media and the adapter are exec'd by absolute path and
// spawn nothing PATH-resolved.
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
