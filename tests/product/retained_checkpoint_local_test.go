package producttest

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/secret"
	"github.com/cozy-creator/cozy/internal/tfs"
	"github.com/cozy-creator/cozy/internal/transfer"
)

var retainedCheckpointFixture = flag.String("retained-checkpoint-fixture", "", "isolated Hub checkpoint proof coordinates")
var retainedCheckpointTFS = flag.String("retained-checkpoint-tfs", "", "native tfs binary for checkpoint input proof")

// Shares the real isolated Hub fixture with its native/Runtime proof. No release
// row, named local alias, fake byte resolver or paid provider is created here.
func TestRetainedCheckpointLocalFetchKeepsNativeRoots(t *testing.T) {
	fixture, binary := *retainedCheckpointFixture, *retainedCheckpointTFS
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
	// The shared Hub fixture also retains a mixed source snapshot. Exercise the
	// exact Creator read methods, including the real private carrier body, so an
	// anonymous model-read change cannot silently break the owner source path.
	sibling := []byte("private source-carrier sibling")
	siblingID, err := canonical.Spell(canonical.Digest(sibling))
	must(t, err)
	var mixedDocument map[string]any
	must(t, json.Unmarshal(original, &mixedDocument))
	mixedDocument["entries"] = append(mixedDocument["entries"].([]any), map[string]any{
		"path": "source.bin", "kind": "file", "blob": map[string]any{
			"sha256": strings.TrimPrefix(siblingID, "sha256:"), "length": len(sibling)},
	})
	mixed, err := json.Marshal(mixedDocument)
	must(t, err)
	mixedID, err := canonical.Spell(canonical.Digest(mixed))
	must(t, err)
	if _, problem := client.CheckpointManifest(context.Background(), ref, mixedID); problem == nil {
		t.Fatal("anonymous Creator read exposed a mixed source snapshot")
	}
	if _, problem := client.CheckpointReads(context.Background(), ref, mixedID, []string{siblingID}); problem == nil {
		t.Fatal("anonymous Creator read exposed a private source carrier")
	}
	owner := client.WithToken(secret.New("checkpoint-proof-admin"), "synthetic Hub fixture")
	actual, problem := owner.CheckpointManifest(context.Background(), ref, mixedID)
	fatal(t, problem)
	if !bytes.Equal(actual, mixed) {
		t.Fatal("owner Creator read changed the retained source manifest")
	}
	grants, problem := owner.CheckpointReads(context.Background(), ref, mixedID, []string{siblingID})
	fatal(t, problem)
	if len(grants) != 1 || grants[0].ObjectID != siblingID || grants[0].Length != int64(len(sibling)) {
		t.Fatal("owner source grant changed exact carrier identity")
	}
	response, err := http.Get(grants[0].URL)
	must(t, err)
	carrier, err := io.ReadAll(response.Body)
	response.Body.Close()
	must(t, err)
	if response.StatusCode != http.StatusOK || !bytes.Equal(carrier, sibling) {
		t.Fatal("owner source carrier transfer failed")
	}
	if _, problem := owner.CheckpointReads(context.Background(), ref, facts.Checkpoint, []string{siblingID}); problem == nil {
		t.Fatal("owner credential bypassed exact checkpoint membership")
	}
	wrong := client.WithToken(secret.New("wrong-owner"), "synthetic refusal control")
	if _, problem := wrong.CheckpointManifest(context.Background(), ref, mixedID); problem == nil {
		t.Fatal("unrecognized owner credential exposed private source")
	}
	t.Log("real Hub digest resolution, native cold/warm local fetch, two retained roots and zero releases passed")
	t.Log("exact Creator owner manifest/read methods transferred private source bytes; anonymous and wrong-scope controls refused")
}
