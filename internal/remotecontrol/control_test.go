package remotecontrol

import (
	"fmt"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
	"github.com/cozy-creator/cozy-creator/internal/hub"
)

func TestValidateOverlayReceiptHardcut(t *testing.T) {
	environmentDigest := "sha256:" + strings.Repeat("1", 64)
	baseDigest := "sha256:" + strings.Repeat("2", 64)
	projectDigest := "sha256:" + strings.Repeat("3", 64)
	environment := canonical.Doc{
		"project_wheel_digest":       projectDigest,
		"wheelhouse_manifest_digest": baseDigest,
	}
	valid := fmt.Sprintf(`{"base_family_digest":%q,"environment_spec_digest":%q,"format":"cozy.runtime.EndpointOverlayReceipt/1","overlay_wheels":[{"digest":%q,"distribution":"probe","owner":"project","version":"1.0.0"}],"project_wheel_digest":%q}`,
		baseDigest, environmentDigest, projectDigest, projectDigest)

	for _, test := range []struct {
		name    string
		raw     string
		wantErr bool
	}{
		{name: "compact overlay receipt", raw: valid},
		{name: "legacy receipt format", raw: strings.Replace(valid,
			"cozy.runtime.EndpointOverlayReceipt/1", "cozy.worker.v1.InstalledEnvironmentReceipt/1", 1), wantErr: true},
		{name: "wrong base family", raw: strings.Replace(valid, baseDigest,
			"sha256:"+strings.Repeat("4", 64), 1), wantErr: true},
		{name: "no project owner", raw: strings.Replace(valid, `"owner":"project"`, `"owner":"custom"`, 1), wantErr: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := validateOverlayReceipt([]byte(test.raw), hub.ExactControlDocument{Digest: environmentDigest}, environment)
			if (err != nil) != test.wantErr {
				t.Fatalf("validateOverlayReceipt() error = %v, wantErr %v", err, test.wantErr)
			}
		})
	}
}
