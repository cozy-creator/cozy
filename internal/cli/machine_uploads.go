package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/publication"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// retainedOutput is one output a finished run holds on the machine it ran on: the custody
// record, its artifact, and the machine (this computer's, or a rental).
type retainedOutput struct {
	hold     records.MachineModelRetention
	artifact records.ModelArtifact
	machine  string
}

// Upload sends a retained output of a finished run from the rental holding it to a private
// checkpoint in destination; nothing runs again. The bytes go from the rental to the Hub
// under this host's owner credentials. An upload in progress is joined and a finished one
// returned, so a repeated call resumes rather than starts over.
func (m *machineRuns) Upload(request records.Request, output, destination string) (records.OutputUpload, *exit.Error) {
	retained, problem := m.retainedOutput(request, output)
	if problem != nil {
		return records.OutputUpload{}, problem
	}
	ref, problem := hub.ParseRef(destination)
	if problem != nil {
		return records.OutputUpload{}, problem
	}
	upload := records.OutputUpload{Request: request.ID, Output: retained.artifact.OutputSlot, Destination: ref.String(),
		Operation: uploadOperation(request.ID, retained.artifact.OutputSlot, ref.String())}
	if _, problem := ownedPublication(m.context.forHub(request.Hub), ref); problem != nil {
		return upload, problem
	}
	upload, _, problem = m.store.BeginOutputUpload(retained.machine, request.Hub, upload)
	if problem == nil && upload.State == "uploading" {
		m.startUpload(upload)
	}
	return upload, problem
}

// StopUpload ends a canceled upload's attempt in flight; its operation is already final.
func (m *machineRuns) StopUpload(id string) {
	if stop, ok := m.uploading.Load(id); ok {
		stop.(context.CancelFunc)()
	}
}

// uploadOperation names one upload by what it sends where, so its Hub publication resumes.
func uploadOperation(request, output, destination string) string {
	sum := sha256.Sum256([]byte(request + "\x00" + output + "\x00" + destination))
	return "upload-" + hex.EncodeToString(sum[:])
}

func (m *machineRuns) retainedOutput(request records.Request, output string) (retainedOutput, *exit.Error) {
	holds, problem := m.store.MachineModelRetentions(request.ID)
	if problem != nil {
		return retainedOutput{}, problem
	}
	var found []retainedOutput
	var names []string
	for _, hold := range holds {
		artifact, _ := records.DecodeModelArtifact(hold.Artifact)
		if hold.State != "held" || artifact == nil {
			continue
		}
		names = append(names, artifact.OutputSlot)
		if output == "" || artifact.OutputSlot == output {
			found = append(found, retainedOutput{hold: hold, artifact: *artifact})
		}
	}
	sort.Strings(names)
	run := m.runName(request)
	switch {
	case len(names) == 0:
		return retainedOutput{}, exit.Named(exit.NotFound, "upload.no_retained_output", "%s holds no retained output", run)
	case len(found) == 0:
		return retainedOutput{}, exit.Named(exit.NotFound, "upload.no_retained_output", "%s retains %s, not %q", run, strings.Join(names, ", "), output)
	case len(found) > 1:
		return retainedOutput{}, exit.Named(exit.Validation, "upload.output_ambiguous", "%s retains %s; name one as <run>#<output>", run, strings.Join(names, ", "))
	}
	link, problem := m.store.MachineExecution(request.ID)
	if problem != nil || link == nil {
		return retainedOutput{}, firstProblem(problem, exit.New(exit.Conflict, "the run has no machine execution"))
	}
	if !machines.IsLocal(link.MachineID) {
		rented, problem := m.store.RentalRow(link.MachineID)
		if problem != nil {
			return retainedOutput{}, problem
		}
		if rented == nil || records.RentalTerminalState(rented.State) {
			return retainedOutput{}, exit.Named(exit.Conflict, "upload.rental_ended",
				"%s was retained on rental %s, which has ended; its bytes are gone", found[0].artifact.OutputSlot, link.MachineID).
				WithRemedy("run the job again with --upload-to <org/model>")
		}
	}
	found[0].machine = link.MachineID
	return found[0], nil
}

// startUpload runs one upload until it lands, is refused for good, or is canceled. Only
// the Hub's or the machine's own refusal ends it; one that does not answer is asked again.
func (m *machineRuns) startUpload(upload records.OutputUpload) {
	ctx, stop := context.WithCancel(m.ctx)
	if _, running := m.uploading.LoadOrStore(upload.ID, stop); running {
		stop()
		return
	}
	go func() {
		defer m.uploading.Delete(upload.ID)
		defer stop()
		said := ""
		for delay := time.Second; ctx.Err() == nil; delay = min(2*delay, 30*time.Second) {
			checkpoint, problem := m.uploadOnce(ctx, upload)
			if ctx.Err() != nil {
				return
			}
			if problem == nil {
				upload.State, upload.Checkpoint = "uploaded", checkpoint
				_ = m.store.SettleOutputUpload(upload)
				return
			}
			if !transient(problem) {
				upload.State, upload.ErrorCode, upload.Error = "failed", problem.ErrName(), problem.Message
				_ = m.store.SettleOutputUpload(upload)
				return
			}
			if problem.Message != said {
				said = problem.Message
				fmt.Fprintf(m.context.Out, "upload #%d of %s to %s: %s; retrying\n", upload.Number, upload.Output, upload.Destination, problem.Message)
			}
			select {
			case <-ctx.Done():
			case <-time.After(delay):
			}
		}
	}()
}

func (m *machineRuns) uploadOnce(ctx context.Context, upload records.OutputUpload) (string, *exit.Error) {
	request, problem := m.store.RequestRow(upload.Request)
	if problem != nil || request == nil {
		return "", firstProblem(problem, exit.New(exit.NotFound, "the run is gone"))
	}
	retained, problem := m.retainedOutput(*request, upload.Output)
	if problem != nil {
		return "", problem
	}
	ref, problem := hub.ParseRef(upload.Destination)
	if problem != nil {
		return "", problem
	}
	owner, problem := ownedPublication(m.context.forHub(request.Hub), ref)
	if problem != nil {
		return "", problem
	}
	digest, err := canonical.Raw(retained.artifact.Manifest.Digest)
	if err != nil {
		return "", exit.New(exit.Conflict, "retained output names an invalid manifest")
	}
	source := &pb.DerivedRetentionRequest{WeightsTransactionId: retained.hold.TransactionID,
		TensorfsReceiptDigest: retained.hold.ReceiptDigest, RetentionId: retained.hold.RetentionID}
	manifest := &pb.Ref{Digest: digest, Length: uint64(retained.artifact.Manifest.Length)}
	connection, problem := m.connectAtHub(machines.AttachOnly(ctx), retained.machine, request.Hub, orchestrator.Holder{Number: upload.Number, What: "upload " + upload.Output + " to " + upload.Destination})
	if problem != nil {
		return "", problem
	}
	defer connection.Close()
	objects, closure, problem := orchestrator.HeldClosure(ctx, connection.artifactTransfer, upload.Operation, source, manifest)
	if problem != nil {
		return "", problem
	}
	intent := publication.UploadIntent{Request: publication.UploadRequest{Artifact: retained.artifact, Destination: ref.String()},
		Objects: objects, ClosureDigest: closure}
	held := func() *exit.Error {
		_, problem := m.retainedOutput(*request, upload.Output)
		return problem
	}
	checkpoint, problem := publication.UploadCheckpoint(ctx, owner, upload.Operation, intent, held,
		func(ctx context.Context, grant hub.Grant, serverTime int64) *exit.Error {
			return orchestrator.PushHeld(ctx, connection.artifactTransfer, upload.Operation, source, manifest, grant, serverTime)
		})
	return checkpoint.Checkpoint, problem
}

func firstProblem(problems ...*exit.Error) *exit.Error {
	for _, problem := range problems {
		if problem != nil {
			return problem
		}
	}
	return nil
}

// handleRunUpload is `cozy run upload <run>[#<output>] <org/model>`: a retained output
// goes from the rental holding it to a private checkpoint, under its own number; no
// release is published.
func handleRunUpload(ctx *Context) *exit.Error {
	run, slot, _ := strings.Cut(ctx.Inv.Args[0], "#")
	ref, problem := hub.ParseRef(ctx.Inv.Args[1])
	if problem != nil {
		return problem
	}
	local, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	upload, problem := local.UploadJobOutput(run, slot, ref.String())
	if problem != nil {
		return problem
	}
	if ctx.Inv.Bool("--await") {
		return watchOperation(ctx, local, upload)
	}
	return emit(ctx, operationRecord(upload))
}
