package producttest

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/publication"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// checkpointHub is the stand-in Hub's private checkpoint upload: an operation-keyed
// publication, one grant per object, the object store, and finalization.
type checkpointHub struct {
	mu        sync.Mutex
	lengths   map[string]int64
	stored    map[string][]byte
	accepted  map[string]bool
	pushes    map[string]int
	finalized map[string]string // checkpoint -> operation
}

func serveCheckpoints(t *testing.T, peer *fakeRentalHub) *checkpointHub {
	h := &checkpointHub{lengths: map[string]int64{}, stored: map[string][]byte{}, accepted: map[string]bool{}, pushes: map[string]int{}, finalized: map[string]string{}}
	mux := peer.mux
	mux.HandleFunc("GET /v1/models/{org}/{name}/checkpoints/{id}", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		if _, done := h.finalized[r.PathValue("id")]; !done {
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":{"code":"checkpoint.not_found","message":"no such checkpoint"}}`))
			return
		}
		_, _ = w.Write(h.stored[r.PathValue("id")])
	})
	mux.HandleFunc("PUT /v1/models/{org}/{name}/publications/{operation}", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Objects []struct {
				ID     string `json:"object_id"`
				Length int64  `json:"length"`
			} `json:"objects"`
		}
		must(t, json.NewDecoder(r.Body).Decode(&body))
		h.mu.Lock()
		defer h.mu.Unlock()
		rows := []map[string]any{}
		for _, object := range body.Objects {
			h.lengths[object.ID] = object.Length
			state := "claimed"
			if h.accepted[object.ID] {
				state = "accepted"
			}
			rows = append(rows, map[string]any{"object_id": object.ID, "length": object.Length, "state": state})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"created": true,
			"publication": map[string]any{"operation": r.PathValue("operation"), "state": "open", "objects": rows}})
	})
	mux.HandleFunc("POST /v1/models/{org}/{name}/publications/{operation}/grants", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			IDs []string `json:"object_ids"`
		}
		must(t, json.NewDecoder(r.Body).Decode(&body))
		now := time.Now().Unix()
		grants := []map[string]any{}
		h.mu.Lock()
		defer h.mu.Unlock()
		for _, id := range body.IDs {
			grants = append(grants, map[string]any{"object_id": id, "length": h.lengths[id], "url": peer.server.URL + "/objects/" + id,
				"required_headers": map[string]string{}, "expires_at_unix": now + 3600})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"grants": grants, "held": []any{}, "server_time_unix": now})
	})
	mux.HandleFunc("PUT /objects/{id}", func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		must(t, err)
		if objectID(raw) != r.PathValue("id") {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		h.mu.Lock()
		defer h.mu.Unlock()
		h.stored[r.PathValue("id")], h.accepted[r.PathValue("id")] = raw, true
		h.pushes[r.PathValue("id")]++
	})
	mux.HandleFunc("POST /v1/models/{org}/{name}/publications/{operation}/finalize", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			ID     string `json:"manifest_id"`
			Length int64  `json:"manifest_length"`
		}
		must(t, json.NewDecoder(r.Body).Decode(&body))
		h.mu.Lock()
		defer h.mu.Unlock()
		var objects int
		var total int64
		for id, raw := range h.stored {
			if id != body.ID {
				objects, total = objects+1, total+int64(len(raw))
			}
		}
		h.finalized[body.ID] = r.PathValue("operation")
		result := map[string]any{"publish_id": r.PathValue("operation"), "checkpoint_id": body.ID,
			"manifest": map[string]any{"sha256": strings.TrimPrefix(body.ID, "sha256:"), "length": body.Length},
			"objects":  objects, "bytes": total, "state": "checkpointed"}
		_ = json.NewEncoder(w).Encode(map[string]any{"operation": r.PathValue("operation"), "state": "completed",
			"status_url": r.URL.Path, "result": result})
	})
	return h
}

// servePodClosure answers the pod's side of NativeArtifactTransfer from the held output's
// closure: inventory pages, and each granted object pushed straight to the Hub. The first
// push of busyOnce is refused busy, as a pod whose transfer slots are full refuses it.
func servePodClosure(t *testing.T, f *retainedWeights, busyOnce string) {
	ids := make([]string, 0, len(f.objects))
	for id := range f.objects {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	rows := make([][]any, 0, len(ids))
	for _, id := range ids {
		rows = append(rows, []any{id, len(f.objects[id])})
	}
	inventory, problem := publication.Canonical(rows)
	fatal(t, problem)
	var busy sync.Once
	// The pod answers each command on the machine connection; no WorkerControl session exists.
	f.pod.artifactTransfer = func(_ context.Context, call *pb.NativeArtifactTransferCall) (*pb.NativeArtifactTransferStatus, error) {
		if err := f.pod.verifyClaim(call.Claim, false); err != nil {
			return nil, err
		}
		command := call.Request
		status := &pb.NativeArtifactTransferStatus{EffectId: command.EffectId, Source: command.Source, Manifest: command.Manifest,
			CommandId: command.CommandId, GrantRevision: command.GrantRevision}
		if command.Grant == nil {
			status.ClosureDigest = canonical.Digest(inventory)
			for _, id := range ids {
				status.Objects = append(status.Objects, &pb.WeightsObjectRef{ObjectId: id, Length: uint64(len(f.objects[id]))})
			}
			return status, nil
		}
		grant := command.Grant
		status.ObjectId = grant.ObjectId
		refused := false
		if grant.ObjectId == busyOnce {
			busy.Do(func() { refused = true })
		}
		if refused {
			status.Outcome, status.SafeCode, status.SafeDetail = pb.WeightsUploadOutcome_WEIGHTS_UPLOAD_OUTCOME_REFUSED, "native_artifact_busy", "no free transfer slot"
			return status, nil
		}
		raw := f.objects[grant.ObjectId]
		put, err := http.NewRequest(http.MethodPut, grant.Url, bytes.NewReader(raw))
		if err != nil {
			return nil, err
		}
		answer, err := http.DefaultClient.Do(put)
		if err != nil {
			return nil, err
		}
		answer.Body.Close()
		status.Outcome, status.ChecksumSha256, status.TransferredBytes = pb.WeightsUploadOutcome_WEIGHTS_UPLOAD_OUTCOME_UPLOADED, grant.ObjectId, uint64(len(raw))
		return status, nil
	}
}

// A retained output is uploaded from the rental holding it to a private checkpoint without
// running anything again: the pod pushes each closure object straight to the Hub under
// grants this host's owner credentials minted. A pod refusing one push busy is asked again
// and only that object moves again; a repeated upload returns the landed checkpoint.
func TestRetainedOutputUploadsFromItsRentalWithoutRunningAgain(t *testing.T) {
	f := newRetainedWeights(t)
	hub := serveCheckpoints(t, f.hub)
	var tensor string
	for id := range f.objects {
		if id != f.manifest {
			tensor = id
			break
		}
	}
	servePodClosure(t, f, tensor)
	startDaemonProcess(t, f.root)
	f.settle(t)
	// The importing client holds only retained-byte custody, not the source
	// installation. Upload must use its explicit publication Hub without replaying it.
	if f.request.InstallID != "" {
		t.Fatal("retained-output fixture unexpectedly owns source installation")
	}

	type uploaded struct {
		State       string   `json:"state"`
		Checkpoint  string   `json:"checkpoint"`
		Destination string   `json:"destination"`
		Next        []string `json:"next"`
	}
	upload := func() uploaded {
		code, out, errs := runCozyWithin(t, f.root, "run", "upload", f.request.ID+"#model", "proof/model", "--await", "--json")
		var got uploaded
		if code != 0 || json.Unmarshal([]byte(out), &got) != nil {
			t.Fatalf("the retained output was not uploaded [exit %d]: %s%s", code, out, errs)
		}
		return got
	}
	first := upload()
	if first.State != "uploaded" || first.Checkpoint != f.manifest || first.Destination != "proof/model" ||
		len(first.Next) != 1 || first.Next[0] != "cozy model publish proof/model --release <label> --lane model="+f.manifest {
		t.Fatalf("upload did not land the output's checkpoint: %+v", first)
	}
	hub.mu.Lock()
	for id, raw := range f.objects {
		if !bytes.Equal(hub.stored[id], raw) {
			t.Errorf("the Hub does not hold object %s", id)
		}
		// The busy refusal moved no bytes, so every object crossed exactly once.
		if hub.pushes[id] != 1 {
			t.Errorf("object %s was pushed %d times, want once", id, hub.pushes[id])
		}
	}
	operation := hub.finalized[f.manifest]
	hub.mu.Unlock()
	if operation == "" {
		t.Fatal("the checkpoint was never finalized")
	}
	awaitLog(t, filepath.Join(f.root, "daemon.log"), "upload of model to proof/model: ", time.Second)

	again := upload()
	hub.mu.Lock()
	defer hub.mu.Unlock()
	if again.State != "uploaded" || again.Checkpoint != f.manifest || len(hub.finalized) != 1 || hub.pushes[tensor] != 1 {
		t.Fatalf("a repeated upload did not return the landed checkpoint: %+v, finalized %v", again, hub.finalized)
	}
}
