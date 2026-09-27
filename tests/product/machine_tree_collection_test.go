package producttest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/resultfiles"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestMachineTreeResultUsesClosedFinalMetadata(t *testing.T) {
	schema, err := canonical.NormalizeJCS([]byte(`{"fields":[{"name":"files","type":{"input":"tree"},"wire":"required","asset_bound":{"max_bytes":100}}]}`))
	must(t, err)
	asset := map[string]any{"asset_ref": childDigest("4"), "digest": childDigest("4"), "kind": "tree", "size_bytes": 79}
	envelope := func(value map[string]any) *pb.ResultEnvelope {
		raw, err := json.Marshal(map[string]any{"files": value})
		must(t, err)
		raw, err = canonical.NormalizeJCS(raw)
		must(t, err)
		return &pb.ResultEnvelope{InlineResult: raw, ResultSchemaDigest: canonical.Digest(schema)}
	}
	fatal(t, launch.ValidateMachineResult(schema, envelope(asset)))
	for name, change := range map[string]func(map[string]any){
		"missing digest": func(a map[string]any) { delete(a, "digest") },
		"changed digest": func(a map[string]any) { a["digest"] = childDigest("5") },
		"changed kind":   func(a map[string]any) { a["kind"] = "file" },
		"oversized":      func(a map[string]any) { a["size_bytes"] = 101 },
	} {
		t.Run(name, func(t *testing.T) {
			copy := map[string]any{}
			for k, v := range asset {
				copy[k] = v
			}
			change(copy)
			if launch.ValidateMachineResult(schema, envelope(copy)) == nil {
				t.Fatal("invalid final Tree metadata accepted")
			}
		})
	}
}

func TestTreeCollectionRejectsMalformedClosureAndExportsIndependentBytes(t *testing.T) {
	data := []byte("verified report bytes")
	hash := sha256.Sum256(data)
	blob := hex.EncodeToString(hash[:])
	entry := func(name string) map[string]any {
		return map[string]any{"path": name, "kind": "file", "blob": map[string]any{"sha256": blob, "length": len(data)}}
	}
	manifest := func(entries ...map[string]any) []byte {
		raw, err := json.Marshal(map[string]any{"entries": entries})
		must(t, err)
		raw, err = canonical.NormalizeJCS(raw)
		must(t, err)
		return raw
	}
	for name, entries := range map[string][]map[string]any{
		"traversal": {entry("../escape")}, "absolute": {entry("/escape")}, "backslash": {entry(`dir\escape`)}, "not clean": {entry("dir/./file")}, "duplicate": {entry("file"), entry("file")}, "ancestor": {entry("file"), entry("file/inside")},
	} {
		t.Run(name, func(t *testing.T) {
			if _, problem := resultfiles.ParseTreeManifest(manifest(entries...), int64(len(entries)*len(data))); problem == nil {
				t.Fatal("invalid tree paths accepted")
			}
		})
	}
	// A newer Runtime's additive member is ignored; kind alone decides what a member is.
	additive := entry("file")
	additive["mode"] = 420
	if members, problem := resultfiles.ParseTreeManifest(manifest(additive), int64(len(data))); problem != nil || len(members) != 1 {
		t.Fatalf("an additive manifest member refused the tree: %v", problem)
	}
	link := entry("file")
	link["kind"] = "symlink"
	if _, problem := resultfiles.ParseTreeManifest(manifest(link), int64(len(data))); problem == nil {
		t.Fatal("a non-file tree member was accepted")
	}
	good := manifest(entry("nested/report.json"), entry("second.txt"))
	if _, problem := resultfiles.ParseTreeManifest(good, int64(len(data))); problem == nil {
		t.Fatal("incomplete content size accepted")
	}
	root := t.TempDir()
	source := filepath.Join(root, "manifest")
	must(t, os.WriteFile(source, good, 0600))
	must(t, os.Mkdir(source+".files", 0700))
	must(t, os.WriteFile(filepath.Join(source+".files", blob), data, 0600))
	digestHash := sha256.Sum256(good)
	digest := "sha256:" + hex.EncodeToString(digestHash[:])
	destination, problem := resultfiles.MaterializeTree(source, filepath.Join(root, "exports"), digest, int64(len(good)), int64(2*len(data)))
	fatal(t, problem)
	again, problem := resultfiles.MaterializeTree(source, filepath.Join(root, "exports"), digest, int64(len(good)), int64(2*len(data)))
	fatal(t, problem)
	if again != destination {
		t.Fatal("tree replay changed its destination")
	}
	output := filepath.Join(destination, "nested/report.json")
	out, err := os.ReadFile(output)
	must(t, err)
	if string(out) != string(data) || !strings.HasSuffix(destination, hex.EncodeToString(digestHash[:])) {
		t.Fatal("tree export lost its exact member or identity")
	}
	must(t, os.WriteFile(output, []byte("user edit"), 0600))
	retained, err := os.ReadFile(filepath.Join(source+".files", blob))
	must(t, err)
	if string(retained) != string(data) {
		t.Fatal("user copy changed internal tree custody")
	}
	if _, problem := resultfiles.MaterializeTree(source, filepath.Join(root, "exports"), digest, int64(len(good)), int64(2*len(data))); problem == nil {
		t.Fatal("edited tree destination was silently overwritten")
	}
	must(t, os.WriteFile(filepath.Join(source+".files", blob), []byte("changed custody"), 0600))
	if _, problem := resultfiles.MaterializeTree(source, filepath.Join(root, "new-export"), digest, int64(len(good)), int64(2*len(data))); problem == nil {
		t.Fatal("changed member custody exported")
	}
}
