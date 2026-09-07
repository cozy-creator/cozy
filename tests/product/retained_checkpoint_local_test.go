package producttest

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/tfs"
	"github.com/cozy-creator/cozy/internal/transfer"
)

// Shares the real isolated Hub fixture with its native/Runtime proof. No release
// row, named local alias, fake byte resolver or paid provider is created here.
func TestRetainedCheckpointLocalFetchKeepsNativeRoots(t *testing.T) {
	fixture, binary := os.Getenv("COZY_CHECKPOINT_FIXTURE"), os.Getenv("COZY_TEST_TFS")
	if fixture == "" || binary == "" {
		t.Skip("requires the isolated retained-checkpoint Hub fixture and native tfs")
	}
	var facts struct {
		HubURL            string `json:"hub_url"`
		Model, Checkpoint string
		ManifestLength    int64 `json:"manifest_length"`
	}
	raw, err := os.ReadFile(fixture)
	must(t, err)
	must(t, json.Unmarshal(raw, &facts))
	area := t.TempDir()
	cfg := config.Config{Tfs: binary, TensorFSRoot: filepath.Join(area, "store"), HubURL: facts.HubURL}
	tool, problem := tfs.Open(cfg)
	fatal(t, problem)
	client := hub.New(cfg, "checkpoint-local-proof")
	fetch := &transfer.Fetch{Tool: tool, Hub: client, Spec: facts.Model + "@" + facts.Checkpoint,
		Scratch: filepath.Join(area, "work"), Locks: filepath.Join(area, "locks")}
	selected, problem := fetch.Resolve(context.Background())
	fatal(t, problem)
	if selected.ManifestID != facts.Checkpoint || selected.Release != "" || selected.Lane != "" {
		t.Fatal("digest resolution invented naming")
	}
	first, problem := fetch.Acquire(context.Background(), selected)
	fatal(t, problem)
	if first.ManifestID != facts.Checkpoint || first.ManifestLength != facts.ManifestLength || first.Moved == 0 {
		t.Fatal("cold checkpoint fetch did not admit exact bytes")
	}
	second, problem := fetch.Acquire(context.Background(), selected)
	fatal(t, problem)
	if second.ManifestID != first.ManifestID || second.Moved != 0 {
		t.Fatal("warm checkpoint fetch moved bytes or changed identity")
	}
	rows, problem := tool.Releases(filepath.Join(area, "releases.jsonl"))
	fatal(t, problem)
	if len(rows) != 0 {
		t.Fatal("digest fetch created a release or local alias")
	}
	ref, problem := hub.ParseRef(facts.Model)
	fatal(t, problem)
	length, problem := tool.RetainedCheckpoint(ref.Org, ref.Name, facts.Checkpoint, fetch.Scratch)
	fatal(t, problem)
	if length != facts.ManifestLength {
		t.Fatal("native repository omitted the retained checkpoint")
	}
	// Retain a second valid manifest under the same release-less repository. This
	// catches existence checks that look only at release rows and replace its roots.
	manifestPath := filepath.Join(area, "manifest.json")
	fatal(t, tool.Manifest(facts.Checkpoint, manifestPath))
	original, err := os.ReadFile(manifestPath)
	must(t, err)
	other := bytes.Replace(original, []byte("model.cozytensors"), []byte("other.cozytensors"), 1)
	if bytes.Equal(other, original) {
		t.Fatal("fixture had no expected native entry")
	}
	otherID, err := canonical.Spell(canonical.Digest(other))
	must(t, err)
	otherPath := filepath.Join(area, "other-manifest.json")
	must(t, os.WriteFile(otherPath, other, 0600))
	_, problem = tool.AdmitManifest(otherPath, otherID, int64(len(other)))
	fatal(t, problem)
	fatal(t, tool.VerifyManifest(otherID))
	fatal(t, tool.CommitCheckpoint(ref.Org, ref.Name, otherID, int64(len(other)), fetch.Scratch))
	for _, id := range []string{facts.Checkpoint, otherID} {
		length, problem := tool.RetainedCheckpoint(ref.Org, ref.Name, id, fetch.Scratch)
		fatal(t, problem)
		if length <= 0 {
			t.Fatal("second retention dropped an earlier root")
		}
	}
	rows, problem = tool.Releases(filepath.Join(area, "releases-after.jsonl"))
	fatal(t, problem)
	if len(rows) != 0 {
		t.Fatal("second checkpoint created a release")
	}
	t.Log("real Hub digest resolution, native cold/warm local fetch, two retained roots and zero releases passed")
}
