package producttest

import (
	"context"
	"encoding/json"
	"io"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
)

// Root-authorized real source release for the composed restart test. It is complete
// native data, not a placeholder; no H3 release or other fixture pointer is changed.
func TestPublishRecoverySourceFixture(t *testing.T) {
	if *publicationHub == "" || *publicationHome == "" || *publicationPython == "" || *publicationModel == "" {
		t.Skip("requires explicit existing task source fixture destination")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	cfg := config.Config{HubURL: *publicationHub, Home: *publicationHome}
	auth := accountauth.New(cfg)
	client := hub.New(cfg, "recovery-source-proof").WithTokenSource(auth)
	account, problem := client.CurrentAccount(ctx)
	fatal(t, problem)
	ref, problem := hub.ParseRef(*publicationModel)
	fatal(t, problem)
	if ref.Org != account.Name {
		t.Fatal("source fixture must belong to current account")
	}
	command := exec.CommandContext(ctx, *publicationPython, "testdata/recovery-source.py", filepath.Join(t.TempDir(), "native"))
	command.Stderr = io.Discard
	input, err := command.StdinPipe()
	must(t, err)
	output, err := command.StdoutPipe()
	must(t, err)
	must(t, command.Start())
	defer func() {
		input.Close()
		if command.ProcessState == nil {
			command.Process.Kill()
			command.Wait()
		}
	}()
	decoder := json.NewDecoder(output)
	var source struct {
		Manifest string       `json:"manifest_id"`
		Length   int64        `json:"manifest_length"`
		Objects  []hub.Object `json:"objects"`
	}
	must(t, decoder.Decode(&source))
	const release, lane, operation = "0.0.0-recovery-source.20260906", "native-source", "recovery-source-fixture-20260906"
	prior, problem := client.ModelRelease(ctx, ref, release)
	if problem == nil {
		if len(prior.Lanes) != 1 || prior.Lanes[0].Lane != lane || prior.Lanes[0].CheckpointID != source.Manifest {
			t.Fatal("existing fixture release differs; refuse overwrite")
		}
	} else {
		if problem.Code != exit.NotFound {
			fatal(t, problem)
		}
		opened, problem := client.OpenPublication(ctx, ref, operation, source.Objects, "retain complete native source for recovery proof")
		fatal(t, problem)
		if opened.Publication.Operation != operation {
			t.Fatal("publication changed source operation")
		}
		ids := make([]string, len(source.Objects))
		for i, object := range source.Objects {
			ids[i] = object.ID
		}
		grants, problem := client.GrantKnownTransfers(ctx, ref, operation, ids, "complete recovery source fixture")
		fatal(t, problem)
		for _, grant := range grants.Grants {
			must(t, json.NewEncoder(input).Encode(map[string]any{"action": "upload", "grant": grant}))
			var done struct {
				HTTP   int    `json:"http_status"`
				Error  string `json:"proof_error"`
				Line   int
				Code   string
				Detail string
			}
			must(t, decoder.Decode(&done))
			if done.Error != "" {
				t.Fatalf("native source transfer refused: %s line%d %s %s", done.Error, done.Line, done.Code, done.Detail)
			}
			if done.HTTP != 200 && done.HTTP != 201 && done.HTTP != 204 && done.HTTP != 412 {
				t.Fatal("native source PUT refused")
			}
		}
		accepted, problem := client.VerifyPublicationObjects(ctx, ref, operation, ids)
		fatal(t, problem)
		if len(accepted.Objects) != len(ids) {
			t.Fatal("source closure verification is incomplete")
		}
		for _, row := range accepted.Objects {
			if row.State != "accepted" {
				t.Fatal("source closure is not held")
			}
		}
		finalized, problem := client.FinalizePublication(ctx, ref, operation, hub.FinalizePublicationRequest{ManifestID: source.Manifest, ManifestLength: source.Length}, "retain complete native recovery source")
		fatal(t, problem)
		if finalized.CheckpointID != source.Manifest {
			t.Fatal("source finalization changed native manifest")
		}
		_, problem = client.UpdateModelRelease(ctx, ref, release, 0, map[string]string{lane: source.Manifest}, nil, "actual tiny source for end-to-end recovery test; no H3 quality claim")
		fatal(t, problem)
	}
	resolved, problem := client.ResolveModel(ctx, ref.String()+"@"+release, lane)
	fatal(t, problem)
	if resolved.ManifestID != source.Manifest || resolved.Release != release || resolved.Lane != lane {
		t.Fatal("public source readback changed exact native root")
	}
	must(t, json.NewEncoder(input).Encode(map[string]string{"action": "stop"}))
	must(t, command.Wait())
	t.Logf("complete source release readback: %s@%s/%s manifest=%s length=%d objects=%d bytes=%d", ref.String(), release, lane, source.Manifest, source.Length, resolved.Objects, resolved.Bytes)
}
