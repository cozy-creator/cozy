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
	lockDigest := "sha256:" + strings.Repeat("4", 64)
	customDigest := "sha256:" + strings.Repeat("5", 64)
	target := map[string]canonical.Value{
		"accelerator_abi": "cu126", "accelerator_backend": "cuda", "libc": "glibc2.39",
		"os_arch": "linux/amd64", "python_abi": "cp312",
	}
	environment := canonical.Doc{
		"platform_target": target, "project_wheel_digest": projectDigest,
		"wheelhouse_manifest_digest": baseDigest,
	}
	project := canonical.Doc{
		"digest": projectDigest, "distribution": "probe", "filename": "probe-1.0.0-py3-none-any.whl",
		"length": int64(123), "tags": []canonical.Value{"py3-none-any"}, "version": "1.0.0",
	}
	resolved := fmt.Sprintf(`{"format":"ResolvedWheelSet/1","lock_digest":%q,"platform_target":{"accelerator_abi":"cu126","accelerator_backend":"cuda","libc":"glibc2.39","os_arch":"linux/amd64","python_abi":"cp312"},"wheelhouse_manifest_digest":%q,"wheels":[{"digest":%q,"distribution":"addon","filename":"addon-2.0.0-py3-none-any.whl","length":456,"tags":["py3-none-any"],"version":"2.0.0"}]}`,
		lockDigest, baseDigest, customDigest)
	customRow := fmt.Sprintf(`{"digest":%q,"distribution":"addon","owner":"custom","version":"2.0.0"}`, customDigest)
	projectRow := fmt.Sprintf(`{"digest":%q,"distribution":"probe","owner":"project","version":"1.0.0"}`, projectDigest)
	valid := fmt.Sprintf(`{"base_family_digest":%q,"environment_spec_digest":%q,"format":"cozy.runtime.EndpointOverlayReceipt/1","overlay_wheels":[%s,%s],"project_wheel_digest":%q}`,
		baseDigest, environmentDigest, customRow, projectRow, projectDigest)
	extraRow := `{"digest":"sha256:6666666666666666666666666666666666666666666666666666666666666666","distribution":"zed","owner":"custom","version":"1"}`

	for _, test := range []struct {
		name    string
		raw     string
		wantErr bool
	}{
		{name: "exact compact overlay receipt", raw: valid},
		{name: "legacy receipt format", raw: strings.Replace(valid,
			"cozy.runtime.EndpointOverlayReceipt/1", "cozy.worker.v1.InstalledEnvironmentReceipt/1", 1), wantErr: true},
		{name: "missing custom wheel", raw: strings.Replace(valid, customRow+",", "", 1), wantErr: true},
		{name: "extra custom wheel", raw: strings.Replace(valid, projectRow, projectRow+","+extraRow, 1), wantErr: true},
		{name: "substituted custom wheel", raw: strings.Replace(valid, customDigest,
			"sha256:"+strings.Repeat("7", 64), 1), wantErr: true},
		{name: "wrong base family", raw: strings.Replace(valid, baseDigest,
			"sha256:"+strings.Repeat("8", 64), 1), wantErr: true},
		{name: "no project owner", raw: strings.Replace(valid, `"owner":"project"`, `"owner":"custom"`, 1), wantErr: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := validateOverlayReceipt([]byte(test.raw), hub.ExactControlDocument{Digest: environmentDigest},
				environment, project, []byte(resolved))
			if (err != nil) != test.wantErr {
				t.Fatalf("validateOverlayReceipt() error = %v, wantErr %v", err, test.wantErr)
			}
		})
	}
}
