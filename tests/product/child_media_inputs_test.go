package producttest

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"image"
	"image/png"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestChildMediaInheritsOnlyParentInputsAndKeepsIndependentCustody(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	var encoded bytes.Buffer
	must(t, png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 16, 16))))
	original := filepath.Join(t.TempDir(), "original.png")
	must(t, os.WriteFile(original, encoded.Bytes(), 0600))
	unlock := inputasset.Guard()
	asset, problem := inputasset.Stage(layout, records.AssetBinding{FieldPath: "original", LocalPath: original}, 4096)
	fatal(t, problem)
	parent, _, problem := store.Submit(records.Request{ID: "req-media-parent", IdemKey: "media-parent", BodyDigest: childDigest("8"), Package: "local/parent", Entrypoint: "run", Kind: "job", Payload: []byte(`{}`), RetainWork: true, Assets: []records.AssetBinding{asset}})
	fatal(t, problem)
	unlock()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent = offerChildParent(t, store, parent)
	ep := launch.Entrypoint{Name: "score", Request: launch.Struct{Fields: []launch.Field{{Name: "media", Wire: "required", Type: json.RawMessage(`{"union":[{"tag":"image","fields":[{"name":"image","wire":"required","type":{"asset":"image"},"asset_bound":{"max_bytes":4096}}]},{"tag":"video","fields":[{"name":"video","wire":"required","type":{"asset":"video"}}]}],"tag_field":"type"}`)}}}}
	payload, _ := json.Marshal(map[string]any{"media": map[string]string{"type": "image", "image": asset.Digest}})
	forwarded, problem := launch.InheritChildAssets(&ep, payload, parent.Assets)
	fatal(t, problem)
	if len(forwarded) != 1 || forwarded[0].FieldPath != "media.image" || forwarded[0].LocalPath != asset.LocalPath || forwarded[0].Digest != asset.Digest {
		t.Fatalf("unexpected inherited media: %+v", forwarded)
	}
	call := records.Request{ID: "req-media-child", IdemKey: "media-child", BodyDigest: childDigest("3"), Package: "local/score", Entrypoint: "score", Kind: "job", Payload: payload, ParentRequestID: parent.ID, ParentCallIndex: 0, ChildIntentDigest: childDigest("4"), ChildTargetDigest: childDigest("5"), Assets: forwarded}
	child, _, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	held, problem := store.RequestRow(child.ID)
	fatal(t, problem)
	if len(held.Assets) != 1 || held.Assets[0].Digest != asset.Digest {
		t.Fatal("child lost its independent input binding")
	}
	key, problem := records.OperationKey(child)
	fatal(t, problem)
	relocated := child
	relocated.Assets = append([]records.AssetBinding(nil), child.Assets...)
	relocated.Assets[0].LocalPath = "/different/private/store"
	same, problem := records.OperationKey(relocated)
	fatal(t, problem)
	if key != same {
		t.Fatal("private path contaminated computation identity")
	}
	foreign, _ := json.Marshal(map[string]any{"media": map[string]string{"type": "image", "image": childDigest("9")}})
	if _, problem := launch.InheritChildAssets(&ep, foreign, parent.Assets); problem == nil {
		t.Fatal("foreign input accepted")
	}
	forged := call
	forged.ID = "req-forged"
	forged.IdemKey = "forged"
	forged.ParentCallIndex = 1
	forged.Assets = append([]records.AssetBinding(nil), forwarded...)
	forged.Assets[0].LocalPath = original
	if _, _, problem := store.SubmitChild(forged, 1, childDigest("1"), "private-boot"); problem == nil {
		t.Fatal("forged input path accepted by atomic admission")
	}
	must(t, os.WriteFile(asset.LocalPath, []byte("changed"), 0600))
	if _, problem := launch.InheritChildAssets(&ep, payload, parent.Assets); problem == nil {
		t.Fatal("changed staged media accepted")
	}
	must(t, os.WriteFile(asset.LocalPath, encoded.Bytes(), 0600))
	db, err := sql.Open("sqlite", layout.DB)
	must(t, err)
	defer db.Close()
	_, err = db.Exec(`UPDATE requests SET state='succeeded' WHERE id=?`, parent.ID)
	must(t, err)
	unlock = inputasset.Guard()
	fatal(t, inputasset.DropUnowned(layout, store, parent.Assets))
	unlock()
	if _, err := os.Stat(asset.LocalPath); err != nil {
		t.Fatal("parent cleanup removed child's owned bytes", err)
	}
	_, err = db.Exec(`UPDATE requests SET state='succeeded' WHERE id=?`, child.ID)
	must(t, err)
	unlock = inputasset.Guard()
	fatal(t, inputasset.DropUnowned(layout, store, child.Assets))
	unlock()
	if _, err := os.Stat(asset.LocalPath); !os.IsNotExist(err) {
		t.Fatal("settled input bytes stayed pinned", err)
	}

}

func TestJobMediaRequiresLocalCredentialBeforePathRead(t *testing.T) {
	root := t.TempDir()
	daemon := startDaemonProcess(t, root)
	response := daemon.call(t, http.MethodPost, "/v1/local/jobs", map[string]any{
		"package": "local/proof", "function": "score", "input": map[string]any{},
		"local_assets": []map[string]any{{"field_path": "image", "local_path": "/not-a-granted-file", "digest": childDigest("8"), "length": 1}},
	}, "Authorization", "")
	if response.Status != http.StatusUnauthorized {
		t.Fatalf("untrusted job media reached file resolution: %s", response.brief())
	}
	if requests := listInvocations(t, root); len(requests) != 0 {
		t.Fatalf("unauthorized media created requests: %+v", requests)
	}
}
