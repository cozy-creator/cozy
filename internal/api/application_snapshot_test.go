package api

import (
	"encoding/json"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestCapturedTreeReplayPreservesApplicationNumbersAndSnapshot(t *testing.T) {
	digest := "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	prior := records.AssetBinding{FieldPath: "folder", Digest: digest, Snapshot: &records.ByteInputSnapshot{Reference: "authored", Path: "/removed/authored/tree"}}
	recorded := records.Request{Assets: []records.AssetBinding{prior}}
	payload, assets, problem := replayMachineInputSnapshots([]byte(`{"seed":18446744073709551615,"number":1e0,"folder":"authored","ordered":[2,1]}`), nil, []string{"authored=/removed/source"}, recorded)
	if problem != nil || len(assets) != 1 || assets[0].Snapshot != prior.Snapshot {
		t.Fatalf("captured snapshot changed: %+v %v", assets, problem)
	}
	var value struct {
		Seed   uint64
		Number json.Number
		Folder string
	}
	if err := json.Unmarshal(payload, &value); err != nil || value.Seed != ^uint64(0) || value.Number.String() != "1.0" || value.Folder != digest {
		t.Fatalf("captured tree rewrite changed application values: %s %v", payload, err)
	}
}
