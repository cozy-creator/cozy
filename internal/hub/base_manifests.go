package hub

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

const maxActiveBaseManifests = 32

// ActiveBaseManifest is Tensorhub's exact current pre-spend authority. Creator stages only the
// digest-fenced WheelhouseManifest bytes; profile/image facts are diagnostic readback.
type ActiveBaseManifest struct {
	WheelhouseManifestDigest string          `json:"wheelhouse_manifest_digest"`
	WheelhouseManifest       json.RawMessage `json:"wheelhouse_manifest"`
	BaseWorkerImageDigest    string          `json:"base_worker_image_digest"`
	CompatibilityProfile     json.RawMessage `json:"compatibility_profile"`
	PlatformTarget           json.RawMessage `json:"platform_target"`
	ActivatedAt              string          `json:"activated_at"`
}

func (c *Client) ActiveBaseManifests(ctx context.Context) ([]ActiveBaseManifest, *exit.Error) {
	var out []ActiveBaseManifest
	if e := c.do(ctx, call{method: http.MethodGet, path: "/v1/base-worker-manifests",
		responseBytes: maxDocument}, &out); e != nil {
		return nil, e
	}
	if len(out) == 0 || len(out) > maxActiveBaseManifests {
		return nil, exit.Named(exit.Conflict, "hub.active_base_manifests_invalid",
			"Tensorhub returned %d active base manifests", len(out)).
			WithRemedy("activate at least one and at most %d approved base wheelhouses",
				maxActiveBaseManifests)
	}
	prior := ""
	for index := range out {
		row := &out[index]
		sum := sha256.Sum256(row.WheelhouseManifest)
		measured := "sha256:" + hex.EncodeToString(sum[:])
		if row.WheelhouseManifestDigest != measured ||
			!validHubDigest(row.BaseWorkerImageDigest) || !json.Valid(row.WheelhouseManifest) ||
			!json.Valid(row.CompatibilityProfile) || !json.Valid(row.PlatformTarget) ||
			strings.TrimSpace(row.ActivatedAt) == "" ||
			prior != "" && row.WheelhouseManifestDigest <= prior {
			return nil, exit.Named(exit.Conflict, "hub.active_base_manifests_invalid",
				"Tensorhub active base manifest %d has invalid identity or ordering", index)
		}
		prior = row.WheelhouseManifestDigest
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].WheelhouseManifestDigest < out[j].WheelhouseManifestDigest
	})
	return out, nil
}

func validHubDigest(value string) bool {
	if len(value) != 71 || !strings.HasPrefix(value, "sha256:") {
		return false
	}
	_, err := hex.DecodeString(value[7:])
	return err == nil && strings.ToLower(value) == value
}
