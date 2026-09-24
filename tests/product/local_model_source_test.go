package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestLocalModelSourceStaysLocalAfterCanonicalization(t *testing.T) {
	root, _, _, _, _ := runModelCatalog(t)
	source := filepath.Join(root, "fixture.safetensors")
	must(t, os.WriteFile(source, []byte("small local source"), 0o600))
	startDaemonProcess(t, root)
	code, out := runCozy(t, root, "model", "download", source, "local/classification-proof", "--json", "--full")
	if code != 0 {
		t.Fatalf("local source was refused before submission: %d %s", code, out)
	}
	var result struct {
		ID string `json:"id"`
	}
	must(t, json.Unmarshal([]byte(out), &result))
	if result.ID == "" {
		t.Fatalf("no submitted run: %s", out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	transfer, problem := store.ModelTransferOf(result.ID)
	fatal(t, problem)
	if transfer == nil || !transfer.LocalOnly || transfer.Source != "file:"+source {
		t.Fatalf("local path lost its placement at submission: %+v", transfer)
	}
}
