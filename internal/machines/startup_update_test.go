package machines

import (
	"encoding/json"
	"slices"
	"testing"
)

func TestStartupUpdatePreservesExplicitInstallArtifacts(t *testing.T) {
	for _, tc := range []struct {
		name, metadata string
		want           []string
	}{
		{"published", `{"host":{"name":"cozy-machine","sha256":"published-agent-hash"},"runtime":{"name":"cozy-runtime 0.18.85"},"tensorfs":{"name":"tensorfs 0.3.78"}}`, []string{"COZY_STARTUP_UPDATE=auto"}},
		{"explicit-host", `{"host_pinned":true,"host":{"name":"cozy-machine","sha256":"host-hash"}}`, []string{"COZY_STARTUP_UPDATE=off"}},
		{"stable-runtime-wheel", `{"runtime":{"name":"cozy_runtime-0.18.85-cp312-abi3-linux_x86_64.whl","sha256":"runtime-hash"}}`, []string{"COZY_STARTUP_UPDATE=off"}},
		{"tensorfs-wheel", `{"tensorfs":{"name":"tensorfs-0.3.78-cp312-abi3-linux_x86_64.whl","sha256":"tensorfs-hash"}}`, []string{"COZY_STARTUP_UPDATE=off"}},
		{"older-pinned-host", `{"host_pinned":true}`, []string{"COZY_STARTUP_UPDATE=off"}},
		{"published-with-added-metadata", `{"host":{"sha256":"published-agent-hash"},"future_fact":true}`, []string{"COZY_STARTUP_UPDATE=auto"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var installed Installed
			if err := json.Unmarshal([]byte(tc.metadata), &installed); err != nil {
				t.Fatal(err)
			}
			if got := installed.startupEnvironment(); !slices.Equal(got, tc.want) {
				t.Fatalf("startup environment = %q, want %q", got, tc.want)
			}
		})
	}
}
