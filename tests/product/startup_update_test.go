package producttest

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/machines"
)

// The public launcher consumes installed selection metadata and starts a real
// process. Its witness records only the update policy, never credentials.
func TestStartupUpdatePreservesExplicitInstallArtifacts(t *testing.T) {
	for _, test := range []struct{ name, metadata, want string }{
		{"published", `{"host":{"name":"cozy-machine","sha256":"published-agent-hash"},"runtime":{"name":"cozy-runtime 0.18.85"},"tensorfs":{"name":"tensorfs 0.3.78"}}`, "auto"},
		{"explicit-host", `{"host_pinned":true,"host":{"name":"cozy-machine","sha256":"host-hash"}}`, "off"},
		{"stable-runtime-wheel", `{"runtime":{"name":"cozy_runtime-0.18.85-cp312-abi3-linux_x86_64.whl","sha256":"runtime-hash"}}`, "off"},
		{"tensorfs-wheel", `{"tensorfs":{"name":"tensorfs-0.3.78-cp312-abi3-linux_x86_64.whl","sha256":"tensorfs-hash"}}`, "off"},
		{"older-pinned-host", `{"host_pinned":true}`, "off"},
		{"published-with-added-metadata", `{"host":{"sha256":"published-agent-hash"},"future_fact":true}`, "auto"},
	} {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			host := machines.NewHost(dir, "", nil)
			t.Cleanup(func() { _ = host.Stop(context.Background()) })
			binary := filepath.Join(host.Root(), "usr/local/bin/cozy-machine")
			must(t, os.MkdirAll(filepath.Dir(binary), 0755))
			must(t, os.WriteFile(binary, []byte("#!/bin/sh\ntrap 'printf \"%s\\n\" \"$?\" > \"$COZY_MACHINE_ROOT/startup-exit\"' EXIT\ncat \"$COZY_MACHINE_ROOT/etc/cozy/software-policy.json\" > \"$COZY_MACHINE_ROOT/startup-policy\"\n"), 0700))
			var installed map[string]any
			must(t, json.Unmarshal([]byte(test.metadata), &installed))
			artifact, ok := installed["host"].(map[string]any)
			if !ok {
				artifact = map[string]any{}
				installed["host"] = artifact
			}
			artifact["module"] = machines.AgentModule // this witness represents the independent agent, never legacy migration
			encoded, err := json.Marshal(installed)
			must(t, err)
			must(t, os.WriteFile(filepath.Join(dir, "installed.json"), encoded, 0600))
			// Launch may seed an absent config, but cannot erase/overwrite existing operator settings.
			config := filepath.Join(host.Root(), "etc/cozy/runtime.yaml")
			must(t, os.MkdirAll(filepath.Dir(config), 0755))
			before := []byte("gpu:\n  budget: 3GiB\noperator_extension: keep\n")
			must(t, os.WriteFile(config, before, 0644))
			host.GPUBudget = "1GiB"
			_, problem := host.Ensure(t.Context(), "", nil, true)
			if after, err := os.ReadFile(config); err != nil || string(after) != string(before) {
				t.Fatalf("launch overwrote machine config: %s %v", after, err)
			}

			// A shell fixture runs as /bin/sh, so the agent's executable identity
			// check can return before its writes finish. Wait for its own observed
			// completion, using the test context rather than a guessed delay.
			for {
				exited, err := os.ReadFile(filepath.Join(host.Root(), "startup-exit"))
				if err == nil && len(exited) > 0 {
					if strings.TrimSpace(string(exited)) != "0" {
						t.Fatalf("startup witness process failed: %q (launch: %v)", exited, problem)
					}
					break
				}
				if err != nil && !os.IsNotExist(err) {
					t.Fatalf("cannot observe startup witness completion: %v", err)
				}
				select {
				case <-t.Context().Done():
					t.Fatalf("startup witness did not finish: %v (launch: %v)", t.Context().Err(), problem)
				case <-time.After(10 * time.Millisecond):
				}
			}
			witness, err := os.ReadFile(filepath.Join(host.Root(), "startup-policy"))
			if err != nil {
				t.Fatalf("agent process wrote no policy witness: %v (launch: %v)", err, problem)
			}
			var policy struct {
				Mode  string `json:"startup_update"`
				Agent string `json:"agent"`
			}
			must(t, json.Unmarshal(witness, &policy))
			if policy.Mode != test.want {
				t.Fatalf("launched policy %q, want %q", witness, test.want)
			}
			wantAgent := "bundled"
			if pinned, _ := installed["host_pinned"].(bool); pinned {
				wantAgent = "explicit"
			}
			if policy.Agent != wantAgent {
				t.Fatalf("launched agent selection %q, want %q", policy.Agent, wantAgent)
			}

		})
	}
}
