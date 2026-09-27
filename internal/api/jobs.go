package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/modeltransfer"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
	"github.com/cozy-creator/cozy/internal/scratch"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// The job family is mounted under /v1/local/ because its typed input trees are local
// directories the caller already owns. Keeping filesystem paths out of the proposed
// common core preserves that boundary.
//
// The event plane is reused, and that is not an exception: a job is a request row in the
// one lifecycle authority, so `GET /v1/requests/{id}/events` streams a job's lifecycle
// with no second event authority anywhere. `cozy run list` is that route's client.

// JobSubmission is the job submit body.
type JobSubmission struct {
	AllowPublish    []string               `json:"allow_publish,omitempty"`
	TimeoutMS       int64                  `json:"timeout_ms,omitempty"`
	RequestedRental string                 `json:"requested_rental,omitempty"`
	LocalAssets     []records.AssetBinding `json:"local_assets,omitempty"`
	Package         string                 `json:"package"`
	Function        string                 `json:"function"`
	Input           json.RawMessage        `json:"input"`
	InstallID       string                 `json:"install_id,omitempty"`
	Release         string                 `json:"release,omitempty"`
	Rental          bool                   `json:"rental,omitempty"`
	RentNew         bool                   `json:"rent_new,omitempty"`
	RentalRequired  bool                   `json:"rental_required,omitempty"`
	RetainWork      bool                   `json:"retain_work,omitempty"`
	RetryOf         string                 `json:"retry_of,omitempty"`
	OutputDirectory string                 `json:"output_directory,omitempty"`
	// Worker pins an internal production step to the already-attached rental that
	// prepared its source Manifests. It is admitted only with the CLI credential.
	Worker string `json:"worker,omitempty"`
	// Models are exact derive-only Manifest capabilities. No path or model bytes
	// cross this local API.
	Models        []orchestrator.ModelRef      `json:"models,omitempty"`
	ModelTransfer *records.ModelTransferIntent `json:"model_transfer,omitempty"`
	// Org is the publishing org whose SCRATCH repo this job lands in
	// (`<org>/_job-<request-id>`). It defaults to `local` — a local host has no identity
	// plane yet (decisions #229) and inventing one would be a fake account.
	Org string `json:"org,omitempty"`
	// Trees are the typed input trees as `ref=<directory>`. The grant is the read
	// capability: a `Tree` field naming a ref that is not here never hydrates.
	Trees []string `json:"trees,omitempty"`
	// PlannedSourceBytes is what a script ingest will pull, so a rental bought for it
	// is sized to the ingest.
	PlannedSourceBytes int64 `json:"planned_source_bytes,omitempty"`
}

// JobHandle is the 202 answer.
type JobHandle struct {
	MachineExecution bool   `json:"machine_execution,omitempty"`
	Number           int64  `json:"number"`
	JobID            string `json:"job_id"`
	Status           string `json:"status"`
	Attempt          uint64 `json:"attempt"`
	Package          string `json:"package"`
	Function         string `json:"function"`
	Repo             string `json:"publication_repo,omitempty"`
	StatusURL        string `json:"status_url"`
	CancelURL        string `json:"cancel_url"`
	EventsURL        string `json:"events_url"`
	QueuePosition    *int   `json:"queue_position,omitempty"`
	QueueDepth       *int   `json:"queue_depth,omitempty"`
	Replay           bool   `json:"idempotent_replay"`
}

func (s *Server) submitJob(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, MaxBody+1))
	if err != nil {
		s.refuse(w, r, http.StatusBadRequest, "unreadable_body",
			"the request body could not be read: "+err.Error(), "")
		return
	}
	if len(body) > MaxBody {
		s.refuse(w, r, http.StatusRequestEntityTooLarge, "body_too_large",
			"a submission body is bounded at "+strconv.Itoa(MaxBody)+" bytes",
			"typed input only; a job's bulk input arrives as a materialized tree")
		return
	}
	var sub JobSubmission
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	err = decoder.Decode(&sub)
	var trailing any
	if err == nil {
		err = decoder.Decode(&trailing)
	}
	if err != io.EOF {
		detail := "multiple JSON values"
		if err != nil {
			detail = err.Error()
		}
		s.refuse(w, r, http.StatusBadRequest, "malformed_body",
			"the submission is not one closed JSON object: "+detail, staleDaemonRemedy(err))
		return
	}
	passThrough := sub.ModelTransfer != nil && sub.Package == "" && sub.Function == ""
	if !passThrough && (sub.Package == "" || sub.Function == "") {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request",
			"a job submission names a package and a job function",
			`{"package":"org/name","function":"census","input":{…}}`)
		return
	}
	// A job's trees name HOST DIRECTORIES that become read/write worker grants — the same
	// authority local_assets carry on /v1/requests (requests.go) — so they take the same
	// gate: a browser bearer must never name host paths (credentials.go).
	if (len(sub.LocalAssets) > 0 || len(sub.Trees) > 0 || sub.Worker != "" || sub.RequestedRental != "" || len(sub.Models) > 0 ||
		sub.ModelTransfer != nil || sub.RetryOf != "" || sub.OutputDirectory != "" || len(sub.AllowPublish) > 0) &&
		!s.cliAuthenticated(r) {
		s.refuse(w, r, http.StatusForbidden, "cli_credential_required",
			"trees name host filesystem directories and require the OS-protected CLI credential",
			"use `cozy run --input-tree <ref>=<dir>`; this build exposes no browser tree-upload route")
		return
	}
	named := sub.RequestedRental
	if named == "" {
		named = sub.Worker
	}
	selectedHub, e := s.submissionHub(r, named)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if sub.RetryOf != "" {
		prior, problem := s.store.RequestByReference(sub.RetryOf)
		if problem != nil {
			s.refuseTyped(w, r, problem)
			return
		}
		if prior == nil || !prior.IsJob() {
			s.refuse(w, r, http.StatusNotFound, "not_found", "no prior job "+sub.RetryOf+" on this host", "")
			return
		}
		if sub.Worker != "" && sub.Worker != prior.Worker {
			s.refuse(w, r, http.StatusConflict, "request.retry_worker_changed", "retry must retain the predecessor's machine", "")
			return
		}
		if sub.RequestedRental != "" && sub.RequestedRental != prior.RequestedRental {
			s.refuseTyped(w, r, exit.New(exit.Conflict, "retry cannot change the requested rental"))
			return
		}
		sub.RequestedRental = prior.RequestedRental
		sub.RetryOf, sub.Worker = prior.ID, prior.Worker
		// A retry continues its predecessor's work, which belongs to that request's hub.
		selectedHub = s.requestHub(*prior)
		sub.RetainWork = true
		sub.Rental, sub.RentalRequired = prior.Rental, prior.RentalRequired
	}
	key := strings.TrimSpace(r.Header.Get("Idempotency-Key"))
	existing, e := s.store.RequestByIdempotencyKey(key)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if existing != nil && !existing.RetainWork && existing.RequestedRental == "" && (existing.State == "canceled" || existing.State == "failed" || existing.State == "refused") {
		released, e := s.store.ReleaseCanceledIdempotencyKey(key)
		if e != nil {
			s.refuseTyped(w, r, e)
			return
		}
		if released {
			existing = nil
		}
	}
	var inputStage *scratch.Dir
	defer func() {
		if inputStage != nil {
			inputStage.Release()
		}
	}()
	var spec orchestrator.Submission
	if existing != nil && s.requestHub(*existing) != selectedHub {
		s.refuseTyped(w, r, idempotencyHubConflict(key, s.requestHub(*existing), selectedHub))
		return
	}
	if existing != nil {
		if e = s.verifyJobReplayInstall(r.Context(), sub, *existing); e != nil {
			s.refuseTyped(w, r, e)
			return
		}
		spec, e = replayJobSubmission(sub, *existing)
		if e == nil && spec.LocalInstallationID != "" && spec.InstallID == "" {
			spec.InstallID = sub.InstallID
		}
		if e == nil {
			export, problem := s.store.OutputExportOf(existing.ID)
			if problem != nil {
				e = problem
			} else if export != nil {
				directory := sub.OutputDirectory
				if directory == "" {
					directory = s.layout.PackageOutputs(spec.Package)
				}
				spec.OutputExport = &records.OutputExportIntent{Directory: directory, Outputs: export.Outputs}
			}
		}
	} else {
		if sub.Rental || sub.RentalRequired || sub.RentNew {
			unlock := localpackage.Guard()
			defer unlock()
		}
		var inputDeclaration *launch.Entrypoint
		spec, inputDeclaration, e = s.resolveJob(r.Context(), selectedHub, sub)
		if e == nil && spec.OutputExport != nil {
			e = resultfiles.Preflight(spec.OutputExport.Directory)
		}
		if e == nil {
			spec.Assets, e = bindAssets(spec.Assets)
		}
		if e == nil {
			inputStage, e = s.freezeMachineInputs(&spec, inputDeclaration)
		}
	}
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if e = modeltransfer.ValidateSubmission(spec); e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if e = records.NormalizeModelTransferIntent(spec.ModelTransfer); e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	spec.IdemKey, spec.Hub = key, selectedHub
	if len(sub.AllowPublish) > 0 {
		spec.AllowPublish, e = hub.NormalizePublicationRepositories(sub.AllowPublish)
		if e != nil {
			s.refuseTyped(w, r, e)
			return
		}
		if !spec.Rental || (spec.LocalInstallationID == "" && !publishedMachineJob(spec)) || s.machineExecutions == nil {
			s.refuseTyped(w, r, exit.Named(exit.Structural, "publication.machine_identity_required", "--allow-publish requires a Runtime-owned rented transaction with its own certificate identity"))
			return
		}
	}
	if sub.TimeoutMS < 0 || sub.TimeoutMS > int64((1<<63-1)/int64(time.Millisecond)) {
		s.refuseTyped(w, r, exit.New(exit.Validation, "job timeout is outside the supported duration range"))
		return
	}
	spec.TimeoutMS = sub.TimeoutMS
	if existing != nil {
		_, spec.DeadlineUnixMS, e = s.store.RequestExecutionTiming(existing.ID)
		if e != nil {
			s.refuseTyped(w, r, e)
			return
		}
	} else if spec.TimeoutMS > 0 {
		spec.DeadlineUnixMS = uint64(time.Now().UnixMilli() + spec.TimeoutMS)
	}
	digest, e := jobSubmissionDigest(spec)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	spec.BodyDigest = digest
	if s.machineExecutions != nil && (spec.LocalInstallationID != "" || publishedMachineJob(spec)) {
		if existing == nil {
			spec.MachineExecutionObserver = true
		} else if link, problem := s.store.MachineExecution(existing.ID); problem == nil {
			spec.MachineExecutionObserver = link != nil
		}
	}
	if spec.MachineExecutionObserver && (uncapturedRootBytes(spec.Assets) || len(spec.Trees) > 0 || spec.ModelTransfer != nil) {
		s.refuseTyped(w, r, exit.Named(exit.Structural, "machine_execution.inputs_not_staged", "this input shape has no machine-side staging path yet; no execution or rental was submitted"))
		return
	}

	// NO CAPACITY IS A STATE, never a refusal — this route cannot answer "busy". The
	// orchestrator records the row, queues it and makes the worker resident; several jobs
	// submitted at once queue against ONE worker and drain in submission order
	// (owner directive, decisions #394 / cr-019).
	recorded, fresh, e := s.recordSubmission(spec)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if inputStage != nil {
		if recorded.ID == spec.RequestID {
			inputStage.Detach()
		} else {
			inputStage.Release()
		}
		inputStage = nil
	}
	jobID, attempt := recorded.ID, uint64(recorded.Ordinal)
	if fresh || spec.MachineExecutionObserver {
		defer s.activateRecorded(recorded)
	}
	publicationRepo := ""
	if !recorded.RetainsLocalOutputs() {
		publicationRepo = home.ScratchRepo(recorded.Org, recorded.ID)
	}
	handle := JobHandle{
		MachineExecution: spec.MachineExecutionObserver,
		Number:           recorded.Number, JobID: jobID, Status: s.publicStatusOf(recorded), Attempt: attempt,
		Package: recorded.Package, Function: recorded.Entrypoint,
		Repo:      publicationRepo,
		StatusURL: "/v1/local/jobs/" + jobID,
		CancelURL: "/v1/local/jobs/" + jobID + "/cancel",
		EventsURL: "/v1/requests/" + jobID + "/events",
		Replay:    !fresh,
	}
	if handle.Replay && settledRequestState(recorded.State) {
		s.orchestrator.RetryOutputExport(recorded.ID)
	}
	if handle.Status == "queued" {
		if position, depth := s.orchestrator.QueueState(recorded.ID); position > 0 {
			handle.QueuePosition, handle.QueueDepth = &position, &depth
		}
	}
	status := http.StatusAccepted
	if handle.Replay {
		status = http.StatusOK
	}
	s.ok(w, r, status, handle)
}

// Installed local roots and every rented root use Runtime submission. A model
// transfer has no machine-side staging path and stays with its own coordinator.
func publishedMachineJob(spec orchestrator.Submission) bool {
	return (spec.Rental || spec.MachineExecutionObserver) && spec.ModelTransfer == nil
}

func replayJobSubmission(sub JobSubmission,
	recorded records.Request,
) (orchestrator.Submission, *exit.Error) {
	payload := []byte(sub.Input)
	if len(payload) == 0 {
		payload = []byte("{}")
	}
	var problem *exit.Error
	payload, sub.LocalAssets, problem = replayMachineInputSnapshots(payload, sub.LocalAssets, sub.Trees, recorded)
	if problem != nil {
		return orchestrator.Submission{}, problem
	}
	models := append([]orchestrator.ModelRef(nil), sub.Models...)
	if len(models) == 0 && !recorded.ModelTransfer.HasAcquisition() {
		models = append(models, recorded.Models...)
	}
	var weightsOutputs []orchestrator.WeightsOutput
	if recorded.WeightsOutputs != "" {
		if err := json.Unmarshal([]byte(recorded.WeightsOutputs), &weightsOutputs); err != nil {
			return orchestrator.Submission{}, exit.Internalf(
				"cannot replay job %s weights outputs: %s", recorded.ID, err)
		}
	}
	outputs := []string(nil)
	if recorded.Outputs != "" {
		outputs = strings.Split(recorded.Outputs, ",")
	}
	trees := []string(nil)
	if recorded.Trees != "" {
		trees = strings.Split(recorded.Trees, ",")
	}
	org := strings.TrimSpace(sub.Org)
	if org == "" {
		org = recorded.Org
	}
	transfer := sub.ModelTransfer
	if transfer == nil {
		transfer = recorded.ModelTransfer
	}
	packageName, function := sub.Package, sub.Function
	if packageName == "" && function == "" {
		packageName, function = recorded.Package, recorded.Entrypoint
	}
	params := []string(nil)
	if transfer != nil {
		for slot := range transfer.SourceProfiles {
			params = append(params, slot)
		}
		sort.Strings(params)
	}
	return orchestrator.Submission{Kind: "job", RetainWork: sub.RetainWork, RetryOf: sub.RetryOf, Package: packageName,
		ChildArtifacts: recorded.ChildArtifacts, OutputDirectory: sub.OutputDirectory,
		Entrypoint: function, Payload: payload, Org: org, Assets: append([]records.AssetBinding(nil), sub.LocalAssets...),
		InstallID: recorded.InstallID, Release: recorded.Release,
		LocalInstallationID: recorded.LocalInstallationID,
		PlanID:              recorded.PlanID, Outputs: outputs, WeightsOutputs: weightsOutputs,
		NeedsAccelerator: recorded.NeedsAccelerator, Trees: trees, Worker: recorded.Worker,
		Rental: sub.Rental || sub.RentalRequired || sub.RentNew || sub.RequestedRental != "", RentalRequired: sub.RentalRequired || sub.RentNew || sub.RequestedRental != "",
		RequestedRental: sub.RequestedRental, RentNew: sub.RentNew,
		Models: models, ModelTransfer: transfer, ProducerParams: params, PlannedSourceBytes: sub.PlannedSourceBytes}, nil
}

// resolveJob turns package+function into the orchestrator's Submission. The
// `job_descriptor_id` is resolved HERE, from the installed package's own PackageInterface —
// a client never names a digest, exactly as it never names a binding plan id.
func (s *Server) resolveJob(ctx context.Context, hub string, sub JobSubmission) (orchestrator.Submission, *launch.Entrypoint, *exit.Error) {
	if sub.RentNew && sub.Worker != "" {
		return orchestrator.Submission{}, nil, exit.Usagef("a fresh rental cannot name an existing worker")
	}
	if sub.ModelTransfer != nil && sub.Package == "" && sub.Function == "" {
		if sub.Rental || sub.RentalRequired || sub.RentNew {
			return orchestrator.Submission{}, nil, exit.Named(exit.Unavailable,
				"model_transfer.rented_pass_through_unavailable",
				"rented pass-through has no typed TensorFS source profiles")
		}
		return orchestrator.Submission{Kind: "job", Package: "cozy/platform",
			Entrypoint: "model-pass-through", Payload: []byte("{}"), Org: "local",
			PlanID: "sha256:" + strings.Repeat("0", 64), ModelTransfer: sub.ModelTransfer}, nil, nil
	}
	out := orchestrator.Submission{
		Kind: "job", RetainWork: sub.RetainWork, RetryOf: sub.RetryOf, Package: sub.Package, Entrypoint: sub.Function,
		Payload: []byte(sub.Input), Org: strings.TrimSpace(sub.Org), Assets: append([]records.AssetBinding(nil), sub.LocalAssets...),
		Release: sub.Release, OutputDirectory: sub.OutputDirectory,
		Rental: sub.Rental || sub.RentalRequired || sub.RentNew || sub.RequestedRental != "", RentalRequired: sub.RentalRequired || sub.RentNew || sub.RequestedRental != "",
		RequestedRental: sub.RequestedRental, RentNew: sub.RentNew,
		Worker: sub.Worker, Models: append([]orchestrator.ModelRef(nil), sub.Models...),
		ModelTransfer: sub.ModelTransfer, PlannedSourceBytes: sub.PlannedSourceBytes,
	}
	if problem := s.validateRequestedRental(out.RequestedRental, hub); problem != nil {
		return out, nil, problem
	}
	if len(out.Payload) == 0 {
		out.Payload = []byte("{}")
	}
	normalized, err := canonical.NormalizeJCS(out.Payload)
	if err != nil {
		return out, nil, exit.New(exit.Validation, "job input cannot be encoded as canonical JSON: %s", err)
	}
	out.Payload = normalized
	if out.Org == "" {
		out.Org = "local"
	}
	if e := validOrg(out.Org); e != nil {
		return out, nil, e
	}
	if s.packages == nil {
		return out, nil, exit.Unavailablef("this Cozy daemon resolves no packages")
	}
	if sub.Worker != "" && !sub.Rental && !sub.RentalRequired {
		return out, nil, exit.Named(exit.Validation, "rental.job_worker_without_rental",
			"a pinned remote worker requires rental authorization")
	}
	if out.Rental {
		if strings.HasPrefix(sub.Package, "local/") {
			if sub.InstallID != "" {
				resolved, problem := s.resolveLocalJob(ctx, sub, out, sub.InstallID)
				return resolved, nil, problem
			}
			refreshed, editable, _, refreshProblem := s.refreshPackage(sub.Package)
			if refreshProblem != nil {
				return out, nil, refreshProblem
			}
			if !editable {
				return out, nil, exit.Named(exit.Conflict, "local_package_install_invalid",
					"%s is not one editable local package", sub.Package)
			}
			resolved, problem := s.resolveLocalJob(ctx, sub, out, refreshed)
			return resolved, nil, problem
		}
		if sub.InstallID != "" || sub.Release == "" {
			return out, nil, exit.Named(exit.Validation, "rental.job_release_incomplete",
				"remote jobs require one exact published release")
		}
		if len(sub.Trees) > 0 && (s.machineExecutions == nil || (out.RequestedRental == "" && out.Worker == "")) {
			return out, nil, exit.Named(exit.Validation, "rental.job_tree_worker_required",
				"Tree inputs require a named private Runtime worker")
		}
		logical, job, problem := s.packages.ResolveRemoteJob(hub,
			sub.Package, sub.Release, sub.Function, sub.Models,
			sub.ModelTransfer.HasAcquisition())
		if problem != nil {
			return out, nil, problem
		}
		if problem := validateInputs(job, &out); problem != nil {
			return out, nil, problem
		}
		out.PlanID, out.Outputs = logical.DescriptorID, logical.Outputs
		out.WeightsOutputs, out.NeedsAccelerator = logical.WeightsOutputs, logical.NeedsAccelerator
		out.ProducerParams = logical.ProducerParams
		out.Models = append([]orchestrator.ModelRef(nil), logical.Models...)
		if problem := s.deriveOutputExport(job, &out); problem != nil {
			return out, nil, problem
		}
		out.Trees = append([]string(nil), sub.Trees...)
		return out, job, nil
	}
	if sub.InstallID == "" {
		refreshed, editable, _, refreshProblem := s.refreshPackage(sub.Package)
		if refreshProblem != nil {
			return out, nil, refreshProblem
		}
		if editable {
			sub.InstallID = refreshed
		}
	}
	if s.machineExecutions != nil && strings.HasPrefix(sub.Package, "local/") && sub.InstallID != "" {
		resolved, problem := s.resolveLocalJob(ctx, sub, out, sub.InstallID)
		return resolved, nil, problem
	}
	var jobs []launch.JobFacts
	var e *exit.Error
	if sub.InstallID != "" {
		var spec orchestrator.WorkerLaunchSpec
		spec, e = s.packages.ResolveInstall(sub.InstallID, nil)
		if e == nil && spec.Placement.Package != sub.Package {
			e = exit.Named(exit.Conflict, "install_package_mismatch",
				"install %s serves %s, not %s", sub.InstallID, spec.Placement.Package, sub.Package)
		}
		if e == nil {
			jobs, e = s.packages.JobsInstall(sub.InstallID)
			out.InstallID = sub.InstallID
		}
		if e == nil && s.machineExecutions != nil {
			installed, problem := s.store.Install(sub.InstallID)
			if problem != nil {
				return out, nil, problem
			}
			if installed != nil && installed.SourceKind == "tensorhub" {
				out.Release = installed.Version
				out.MachineExecutionObserver = true
			}
		}
	} else {
		jobs, e = s.packages.Jobs(sub.Package)
	}
	if e != nil {
		return out, nil, e
	}
	names := make([]string, 0, len(jobs))
	for _, job := range jobs {
		if !job.Internal {
			names = append(names, job.Name)
		}
		if job.Name != sub.Function {
			continue
		}
		out.PlanID = job.DescriptorID
		out.ChildArtifacts = job.RetainsArtifacts
		out.ReleaseImplicitWork = (job.Result.Fields != nil || job.Result.Input == "model") && len(job.WeightsOutputs) == 0 && (len(job.Outputs) == 0 || job.RetainsArtifacts)
		out.Outputs = job.Outputs
		out.WeightsOutputs = job.WeightsOutputs
		out.NeedsAccelerator = job.NeedsAccelerator
		out.ProducerParams = job.ModelParams
		if problem := validateInputs(&launch.Entrypoint{Name: job.Name, Kind: "job", Internal: job.Internal, Request: job.Request, Assets: job.Assets}, &out); problem != nil {
			return out, nil, problem
		}
		if problem := s.deriveOutputExport(&launch.Entrypoint{Result: job.Result}, &out); problem != nil {
			return out, nil, problem
		}
	}
	if out.PlanID == "" {
		return out, nil, exit.Named(exit.NotFound, "unknown_job",
			"%s registers no job named %q", sub.Package, sub.Function).
			WithRemedy("it registers: %s", strings.Join(names, ", "))
	}
	if out.MachineExecutionObserver && out.ModelTransfer != nil {
		out.MachineExecutionObserver = false
	}
	for _, pair := range sub.Trees {
		ref, dir, ok := strings.Cut(pair, "=")
		if !ok || ref == "" || dir == "" {
			return out, nil, exit.Usagef("%q is not ref=<directory>", pair)
		}
		abs, err := filepath.Abs(dir)
		if err != nil {
			return out, nil, exit.Usagef("%q is not a resolvable directory: %s", dir, err)
		}
		out.Trees = append(out.Trees, ref+"="+abs)
	}
	return out, nil, nil
}

func (s *Server) resolveLocalJob(ctx context.Context, sub JobSubmission,
	out orchestrator.Submission, installID string,
) (orchestrator.Submission, *exit.Error) {
	spec, problem := s.packages.ResolveInstall(installID, nil)
	if problem != nil || spec.Placement.Package != sub.Package {
		if problem != nil {
			return out, problem
		}
		return out, exit.Named(exit.Conflict, "install_package_mismatch",
			"install %s serves %s, not %s", installID, spec.Placement.Package, sub.Package)
	}
	jobs, problem := s.packages.JobsInstall(installID)
	if problem != nil {
		return out, problem
	}
	for _, job := range jobs {
		if job.Name != sub.Function {
			continue
		}
		out.PlanID, out.Outputs = job.DescriptorID, job.Outputs
		out.ChildArtifacts = job.RetainsArtifacts
		out.ReleaseImplicitWork = (job.Result.Fields != nil || job.Result.Input == "model") && len(job.WeightsOutputs) == 0 && (len(job.Outputs) == 0 || job.RetainsArtifacts)
		out.WeightsOutputs, out.NeedsAccelerator = job.WeightsOutputs, job.NeedsAccelerator
		out.ProducerParams = job.ModelParams
		if problem := validateInputs(&launch.Entrypoint{Name: job.Name, Kind: "job", Internal: job.Internal, Request: job.Request, Assets: job.Assets}, &out); problem != nil {
			return out, problem
		}
		if problem := s.deriveOutputExport(&launch.Entrypoint{Result: job.Result}, &out); problem != nil {
			return out, problem
		}
	}
	if out.PlanID == "" {
		return out, exit.Named(exit.NotFound, "unknown_job",
			"%s registers no job named %q", sub.Package, sub.Function)
	}
	revision, problem := s.packages.PrepareLocal(ctx, installID)
	if problem != nil {
		return out, problem
	}
	out.InstallID = installID
	out.Release = revision.Release
	out.Trees = append([]string(nil), sub.Trees...)
	out.LocalInstallationID = revision.ID
	return out, nil
}

func validateJobPayload(pkg string, job launch.JobFacts, payload json.RawMessage) *exit.Error {
	return launch.ValidatePayload(pkg,
		&launch.Entrypoint{Name: job.Name, Kind: "job", Request: job.Request}, payload)
}

// validOrg keeps the SCRATCH REPO's name spellable. The org is one path segment of
// `<org>/_job-<request-id>`, so a separator or a dot component in it would move the
// publication root — which is the escape the destination fence exists to refuse. Refusing
// the org outright is the stronger form: the bad name never reaches a path at all.
func validOrg(org string) *exit.Error {
	for _, c := range org {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '-':
		default:
			return exit.Named(exit.Validation, "invalid_org",
				"%q is not an org name: %q is not [A-Za-z0-9-]", org, string(c)).
				WithRemedy("the org is one segment of the scratch repo <org>/_job-<id>")
		}
	}
	if org == "" {
		return exit.Named(exit.Validation, "invalid_org", "an org name is not empty")
	}
	return nil
}

func jobSubmissionDigest(spec orchestrator.Submission) (string, *exit.Error) {
	weightsOutputs := make([]canonical.Value, 0, len(spec.WeightsOutputs))
	for _, output := range spec.WeightsOutputs {
		weightsOutputs = append(weightsOutputs, map[string]canonical.Value{
			"max_bytes": int64(output.MaxBytes), "mime_type": output.MimeType,
			"output_id": output.OutputID,
		})
	}
	modelRefs := append([]orchestrator.ModelRef(nil), spec.Models...)
	sort.Slice(modelRefs, func(i, j int) bool { return modelRefs[i].Slot < modelRefs[j].Slot })
	models := make([]canonical.Value, 0, len(modelRefs))
	for _, model := range modelRefs {
		models = append(models, map[string]canonical.Value{
			"package": model.Package, "slot": model.Slot, "model": model.Model,
			"release": model.Release, "lane": model.Lane, "manifest": model.Manifest,
			"manifest_length": model.ManifestLength,
		})
	}
	doc := map[string]canonical.Value{
		"kind":            "job",
		"package":         spec.Package,
		"function":        spec.Entrypoint,
		"plan_id":         spec.PlanID,
		"install_id":      spec.InstallID,
		"org":             spec.Org,
		"input":           base64.StdEncoding.EncodeToString(spec.Payload),
		"outputs":         strings.Join(spec.Outputs, ","),
		"weights_outputs": weightsOutputs,
		"trees":           strings.Join(spec.Trees, ","),
		"models":          models,
	}
	if spec.LocalInstallationID != "" {
		// The immutable code identity outlives its reclaimable intake install.
		delete(doc, "install_id")
		doc["local_installation_id"] = spec.LocalInstallationID
	}
	if spec.TimeoutMS > 0 {
		doc["timeout_ms"] = spec.TimeoutMS
	}
	if len(spec.AllowPublish) > 0 {
		values := make([]canonical.Value, 0, len(spec.AllowPublish))
		for _, repository := range spec.AllowPublish {
			values = append(values, repository)
		}
		doc["allow_publish"] = values
	}
	if assets := assetIdentity(spec.Assets); len(assets) > 0 {
		doc["assets"] = assets
	}
	if spec.RetainWork {
		doc["retain_work"] = true
	}
	if spec.RetryOf != "" {
		doc["retry_of"] = spec.RetryOf
	}
	if spec.Rental {
		doc["rental"] = true
		doc["release"] = spec.Release
	}
	if spec.RentNew {
		doc["rent_new"] = true
	}
	if spec.RentalRequired {
		doc["rental_required"] = true
	}
	if spec.OutputDirectory != "" {
		doc["output_directory"] = spec.OutputDirectory
	}
	if spec.RequestedRental != "" {
		doc["requested_rental"] = spec.RequestedRental
	}
	if spec.ModelTransfer != nil {
		encoded, err := json.Marshal(spec.ModelTransfer)
		if err != nil {
			return "", exit.Internalf("cannot encode model transfer intent: %s", err)
		}
		doc["model_transfer"] = base64.StdEncoding.EncodeToString(encoded)
	}
	data, err := canonical.Write(doc)
	if err != nil {
		return "", exit.Internalf("cannot canonicalize the job submission: %s", err)
	}
	spelled, err := canonical.Spell(canonical.Digest(data))
	if err != nil {
		return "", exit.Internalf("cannot digest the job submission: %s", err)
	}
	return spelled, nil
}

// ------------------------------------------------------------------------- status

// JobState is one job's document: the lifecycle a request has, plus the two facts only a
// job has — its publication and its running bill.
type JobState struct {
	MachineExecution *MachineExecutionView `json:"machine_execution,omitempty"`
	ParentRequestID  string                `json:"parent_request_id,omitempty"`
	ParentCallIndex  *int64                `json:"parent_call_index,omitempty"`
	ReusedFrom       string                `json:"reused_from,omitempty"`
	RetainWork       bool                  `json:"retain_work,omitempty"`
	Retaining        bool                  `json:"retaining,omitempty"`
	RetryAvailable   bool                  `json:"retry_available,omitempty"`
	StoppedEventID   int64                 `json:"stopped_event_id,omitempty"`
	RetryOf          string                `json:"retry_of,omitempty"`
	ReuseScope       string                `json:"reuse_scope,omitempty"`
	Number           int64                 `json:"number"`
	JobID            string                `json:"job_id"`
	Status           string                `json:"status"`
	Package          string                `json:"package"`
	Function         string                `json:"function"`
	Attempt          uint64                `json:"attempt"`
	Attempts         int                   `json:"attempts"`
	// Queued is the job's position in the dispatch queue while it waits for a worker,
	// counted from 1. Absent once it has an attempt — a running job is not queued.
	QueuePosition *int `json:"queue_position,omitempty"`
	QueueDepth    *int `json:"queue_depth,omitempty"`
	// Requeues and RetryBudget are the orchestrator's RETRY PROJECTION made visible: how
	// much of the durable budget the neutral outcomes have already spent, and what the
	// bound is. A settlement that exhausted it names the budget in `error`.
	Requeues         int64             `json:"requeues"`
	RetryBudget      int64             `json:"retry_budget"`
	Progress         map[string]any    `json:"progress,omitempty"`
	Stage            string            `json:"stage,omitempty"`
	QueuedMS         int64             `json:"queued_ms"`
	ExecutionMS      int64             `json:"execution_ms"`
	Metrics          map[string]any    `json:"metrics,omitempty"`
	ErrorType        string            `json:"error_type,omitempty"`
	Error            string            `json:"error,omitempty"`
	CanceledBy       string            `json:"canceled_by,omitempty"`
	Result           any               `json:"result,omitempty"`
	Outputs          []MediaRef        `json:"outputs"`
	OutputExport     *OutputExportRef  `json:"output_export,omitempty"`
	Weights          []WeightsRef      `json:"weights,omitempty"`
	Checkpoints      []JobCheckpoint   `json:"checkpoints,omitempty"`
	ModelOutputs     map[string]string `json:"model_outputs,omitempty"`
	ModelDestination string            `json:"model_destination,omitempty"`
	// ModelSources is the per-member source projection, and it exists because the
	// aggregate beside it answers "how far" and nothing answers "why". Run 205 sat at
	// 44 of 48 verified for 2h55m with four members permanently failed, and every read
	// surface showed a percentage: `safe_code` and `safe_detail` reached the durable row
	// and the live frame and then stopped. Only members that are NOT verified are listed —
	// a verified member has nothing to say — so a healthy transfer carries none of this.
	ModelSources  []ModelSourceState `json:"model_sources,omitempty"`
	Publication   *PublicationRef    `json:"publication,omitempty"`
	NativeOutputs []NativeOutputRef  `json:"native_outputs,omitempty"`
	// Bill is ABSENT unless this host was configured with an explicit local rate. There
	// is no `$0.00`: a fabricated zero is a claim about money nobody made (cl-004).
	Bill      *JobBill   `json:"bill,omitempty"`
	Triage    *TriageRef `json:"triage,omitempty"`
	CreatedAt string     `json:"created_at"`
	EventsURL string     `json:"events_url"`
}

// ModelSourceState is one selected source file's transfer, as a client sees it. `safe_code`
// and `safe_detail` are the POD's own words about these bytes, bounded and sanitized at the
// border they came from; this host neither rewrites nor summarizes them.
// NativeOutputRef describes retained bytes without inventing a local publication.
type NativeOutputRef struct {
	OutputID  string `json:"output_id"`
	Kind      string `json:"kind"`
	Digest    string `json:"digest"`
	SizeBytes int64  `json:"size_bytes"`
	MediaType string `json:"media_type"`
	State     string `json:"state"`
}

type ModelSourceState struct {
	Member      string `json:"member"`
	State       string `json:"state"`
	Transferred int64  `json:"transferred_bytes"`
	Length      int64  `json:"length"`
	SafeCode    string `json:"safe_code,omitempty"`
	SafeDetail  string `json:"safe_detail,omitempty"`
}

// maxJobModelSources bounds the projection. A selection may hold up to 20,000 members and
// this document is re-read on every poll of `cozy run watch`, so the rows are truncated
// rather than allowed to become the response. The counts beside them stay exact.
const maxJobModelSources = 32

// WeightsRef is Cozy's durable scratch adoption projection. It exposes no path or
// TensorFS internals: the exact Runtime receipt digest and Cozy-derived private root id
// are the handles a later explicit promotion will consume.
type WeightsRef struct {
	Attempt       int64  `json:"attempt"`
	OutputSlot    string `json:"output_slot"`
	Disposition   string `json:"disposition"`
	Outcome       string `json:"outcome,omitempty"`
	ReceiptDigest string `json:"weights_receipt_digest,omitempty"`
	ScratchRootID string `json:"scratch_root_id,omitempty"`
}

// PublicationRef is the durable publication, as a client sees it. `root` is this host's
// own path and is a LOCAL-module fact: the durable location is the point of the thing,
// and a local caller is the one who owns the directory.
type PublicationRef struct {
	Repo    string `json:"repo"`
	Root    string `json:"root"`
	Status  string `json:"status"`
	Cause   string `json:"cause,omitempty"`
	Entries int64  `json:"entries"`
	Bytes   int64  `json:"bytes"`
	// CHECKPOINTS ARE ABSENT ON PURPOSE. A job that produces canonical bytes publishes
	// them through the runtime's own publication transaction, and this host will root and
	// project ONE typed receipt from that border when it exists (cr-009 seam). Guessing at
	// one from an author's result field would be a second publication protocol.
	CommittedAt string `json:"committed_at"`
}

// JobCheckpoint is one journaled checkpoint DECLARATION. Not a durable-save receipt: no
// component stores the bytes yet (#553a), and the attempt is part of the identity because
// two attempts of one request are two runs with two checkpoint sets.
type JobCheckpoint struct {
	Attempt       int64  `json:"attempt"`
	OperationKey  string `json:"operation_key"`
	LogicalKey    string `json:"logical_key"`
	ContentDigest string `json:"content_digest"`
	ReceiptID     string `json:"receipt_id"`
	Outcome       string `json:"outcome"`
}

// JobBill is the running cost, and it exists only where a rate does.
type JobBill struct {
	RateMicroUSDPerHour int64  `json:"rate_micro_usd_per_hour"`
	MicroUSD            int64  `json:"micro_usd"`
	Source              string `json:"source"`
}

func (s *Server) getJob(w http.ResponseWriter, r *http.Request) {
	row, ok := s.jobRow(w, r)
	if !ok {
		return
	}
	var problem *exit.Error
	if r.URL.Query().Get("observe") != "false" { // as getRequest
		problem = s.refreshMachineExecution(r.Context(), row)
	}
	if current, e := s.store.RequestRow(row.ID); e == nil && current != nil {
		current.Number = row.Number
		row = *current
	}
	state := s.jobStateOf(row)
	if state.MachineExecution != nil && problem != nil {
		state.MachineExecution.ObservationError = problem.Message
	}
	s.ok(w, r, http.StatusOK, state)
}

func (s *Server) jobRow(w http.ResponseWriter, r *http.Request) (records.Request, bool) {
	reference := r.PathValue("id")
	row, e := s.store.RequestByReference(reference)
	if e != nil {
		s.refuseTyped(w, r, e)
		return records.Request{}, false
	}
	if row == nil || !row.IsJob() {
		s.refuse(w, r, http.StatusNotFound, "not_found", "no job "+reference+" on this host",
			"`cozy run list` lists the jobs this host recorded")
		return records.Request{}, false
	}
	return *row, true
}

func (s *Server) jobStateOf(row records.Request) JobState {
	if link, problem := s.store.MachineExecution(row.ID); problem == nil && link != nil {
		return s.machineJobState(row, link)
	}
	retaining, retentionProblem := s.store.RequestRetaining(row)
	if retentionProblem != nil {
		retaining = row.RetainWork
	}
	state := JobState{
		RetainWork:     row.RetainWork,
		Retaining:      retaining,
		RetryAvailable: s.store.RetainedRetryAvailable(row),
		RetryOf:        row.RetryOf, ReuseScope: row.ReuseScope,
		Number: row.Number, JobID: row.ID, Status: s.publicStatusOf(row), Package: row.Package,
		Function: row.Entrypoint, Attempt: uint64(row.Ordinal),
		Requeues: row.Requeues, RetryBudget: orchestrator.MaxRequeues,
		Outputs: []MediaRef{}, CreatedAt: row.CreatedAt,
		EventsURL:    "/v1/requests/" + row.ID + "/events",
		OutputExport: s.outputExportOf(row.ID),
	}
	if row.State == "blocked" && state.Status == "failed" {
		state.StoppedEventID = s.store.StoppedEventID(row)
	}
	if row.State == "blocked" || row.State == "failed" {
		state.ErrorType, _, state.Error, _ = s.store.SettledFailure(row.ID)
	}
	if row.ParentRequestID != "" {
		state.ParentRequestID = row.ParentRequestID
		index := row.ParentCallIndex
		state.ParentCallIndex = &index
		state.ReusedFrom = row.ReusedFrom
	}
	if row.ModelTransfer != nil {
		if transfer, problem := s.store.ModelTransferOf(row.ID); problem == nil && transfer != nil {
			state.ModelDestination = transfer.Destination
			state.ModelOutputs = make(map[string]string, len(transfer.Checkpoints))
			for slot, checkpoint := range transfer.Checkpoints {
				state.ModelOutputs[slot] = checkpoint
			}
			if rows, weightsProblem := s.store.AllModelTransferWeights(row.ID, row.Ordinal); weightsProblem == nil {
				for _, weights := range rows {
					if weights.FinalID != "" {
						state.ModelOutputs[weights.OutputSlot] = weights.ManifestID
					}
				}
			}
			if row.State == "failed" {
				state.ErrorType, state.Error = transfer.ErrorCode, transfer.SafeError
			}
			// READ ONCE, WHATEVER THE STATE. The rows used to be read only while
			// `materializing`, which is to say only while there was nothing to explain: a
			// transfer that has already failed is exactly when the reason is wanted.
			// Completed transfer publication proves the accepted source preparation finished.
			// Raw transfer counters may be absent when conversion ran through native calls.
			if statuses, statusProblem := s.store.ModelTransferSourceStatuses(row.ID); statusProblem == nil && transfer.State != "completed" {
				var transferred, total int64
				verified, converted := 0, 0
				for _, status := range statuses {
					transferred += status.Transferred
					if status.State == "converted" {
						total += status.Transferred
					} else {
						total += status.Length
					}
					if status.State == "verified" {
						verified++
						continue
					}
					if status.State == "converted" {
						converted++
						continue
					}
					if len(state.ModelSources) < maxJobModelSources {
						state.ModelSources = append(state.ModelSources, ModelSourceState{
							Member: status.Member, State: status.State,
							Transferred: status.Transferred, Length: status.Length,
							SafeCode: status.SafeCode, SafeDetail: status.SafeDetail})
					}
				}
				if transfer.State == "materializing" {
					state.Stage = "source materialization"
					// members_verified/members_total is the "44 of 48" a human reads. It
					// was in the daemon log and on no client surface at all.
					state.Progress = map[string]any{"stage": state.Stage,
						"transferred_bytes": transferred, "total_bytes": total,
						"members_verified": verified, "members_converted": converted, "members_total": len(statuses)}
					if total > 0 {
						state.Progress["fraction"] = float64(transferred) / float64(total)
					}
				}
			}
		}
	}
	if row.Ordinal == 0 && (row.State == "submitted" || row.State == "queued") {
		if position, depth := s.orchestrator.QueueState(row.ID); position > 0 {
			state.QueuePosition, state.QueueDepth = &position, &depth
		}
	}
	attempts, _ := s.store.Attempts(row.ID)
	state.Attempts = len(attempts)
	outs, _ := s.store.VisibleOutputs(row.ID)
	for _, o := range outs {
		state.Outputs = append(state.Outputs, MediaRef{
			OutputID: o.OutputID, MediaID: o.MediaID, URL: "/v1/media/" + o.MediaID,
			MimeType: o.MimeType, Length: o.Length, Digest: o.Digest,
		})
	}
	if rows, e := s.store.Checkpoints(row.ID); e == nil {
		for _, c := range rows {
			state.Checkpoints = append(state.Checkpoints, JobCheckpoint{
				Attempt: c.Attempt, OperationKey: c.OperationKey, LogicalKey: c.LogicalKey,
				ContentDigest: c.ContentDigest, ReceiptID: c.ReceiptID, Outcome: c.Outcome,
			})
		}
	}
	if row.ParentRequestID == "" && row.RetainsLocalOutputs() {
		if outputs, problem := s.store.ByteOutputs(row.ID, row.Ordinal); problem == nil {
			holds, _ := s.store.NativeArtifactRetentions(row.ID)
			for _, output := range outputs {
				kind, custody := "file", "pending"
				if output.MimeType == "application/vnd.cozy.tree-manifest" {
					kind = "tree"
				}
				for _, hold := range holds {
					if hold.Kind == "result" && hold.ProducerID == row.ID && hold.ProducerAttempt == output.Attempt && hold.ProducerOutputID == output.OutputID {
						custody = hold.State
					}
				}
				if custody == "held" {
					custody = "retained"
				}
				state.NativeOutputs = append(state.NativeOutputs, NativeOutputRef{OutputID: output.OutputID, Kind: kind, Digest: output.Digest, SizeBytes: output.ContentBytes, MediaType: output.MimeType, State: custody})
			}
		}
	}
	if p, e := s.store.PublicationOf(row.ID); e == nil && p != nil {
		state.Publication = &PublicationRef{
			Repo: p.Repo, Root: p.Root, Status: p.Status, Cause: p.Cause,
			Entries: p.Entries, Bytes: p.Bytes, CommittedAt: p.CommittedAt,
		}
	}
	if rows, e := s.store.WeightsFinalizationsOf(row.ID); e == nil {
		for _, weights := range rows {
			receiptDigest := weights.ReceiptDigest
			if receiptDigest == "" {
				receiptDigest = weights.ResultReceiptDigest
			}
			state.Weights = append(state.Weights, WeightsRef{
				Attempt: weights.Attempt, OutputSlot: weights.OutputSlot,
				Disposition: weights.Disposition, Outcome: weights.ResultOutcome,
				ReceiptDigest: receiptDigest, ScratchRootID: weights.ScratchRootID,
			})
		}
	}
	// THE ELAPSED CLOCK is the authority's own timestamps, and the LIVE progress is the
	// lossy lane's latest tick — replayed on connect, never durable, never load-bearing.
	if state.Status == "canceled" {
		if actor, errType, errText, problem := s.store.CancelAttribution(row.ID); problem == nil {
			state.CanceledBy = actor
			if state.Error == "" && errText != "" {
				state.ErrorType, state.Error = errType, errText
			}
		}
	}
	terminalAt, _ := s.store.TerminalEventAt(row.ID)
	if terminalAt == "" && state.StoppedEventID != 0 {
		terminalAt = s.store.StoppedEventAt(row)
	}
	state.QueuedMS = queuedMS(row, attempts, terminalAt)
	state.ExecutionMS = executionMS(row, attempts, terminalAt)
	if frame, ok := s.orchestrator.LatestFrame(row.ID); ok {
		if value, ok := frame.Value.(map[string]any); ok {
			state.Progress = value
			if stage, ok := value["stage"].(string); ok {
				state.Stage = stage
			}
		}
	}
	// THE BILL. Absent unless a rate was configured — see JobBill. `$0.00` for a job
	// that cost real electricity is a fabricated fact, and this host does not make one.
	if rate := s.cfg.LocalRateMicroUSDPerHour; rate > 0 {
		state.Bill = &JobBill{
			RateMicroUSDPerHour: rate,
			MicroUSD:            rate * state.ExecutionMS / 3_600_000,
			Source:              "COZY_LOCAL_RATE_MICRO_USD_PER_HOUR",
		}
	}
	if len(attempts) == 0 {
		if row.ReusedFrom != "" {
			if prior, problem := s.store.Attempts(row.ReusedFrom); problem == nil && len(prior) > 0 {
				last := prior[len(prior)-1]
				if last.State == "closed" && last.TerminalStatus == "SUCCEEDED" {
					if doc, err := canonical.Read(last.TerminalBody, &pb.AttemptOutcomeBody{}); err == nil {
						state.Result = inlineJobResult(doc)
					}
				}
			}
		}
		return state
	}
	last := attempts[len(attempts)-1]
	if last.TriageSubject != "" {
		state.Triage = &TriageRef{
			AttemptKey: last.AttemptKey, SubjectID: last.TriageSubject,
			URL:    "/v1/local/attempts/" + last.AttemptKey + "/triage",
			Length: last.TriageLength, Kept: last.TriageKept,
		}
	}
	if len(last.TerminalBody) == 0 {
		return state
	}
	doc, err := canonical.Read(last.TerminalBody, &pb.AttemptOutcomeBody{})
	if err != nil {
		return state
	}
	metrics := doc.Sub("metrics")
	state.Metrics = map[string]any{
		"runtime_ms": metrics.Int("runtime_ms"), "queue_ms": metrics.Int("queue_ms"),
		"handler_ms": metrics.Int("handler_ms"),
	}
	if last.TerminalStatus != "SUCCEEDED" && state.ErrorType == "" {
		state.ErrorType, state.Error = last.TerminalCause, last.SafeMessage
	}
	state.Result = inlineJobResult(doc)
	if row.State == "finalizing" && len(state.NativeOutputs) > 0 {
		state.Result = nil
		state.Stage = "retaining native results"
		state.ErrorType, state.Error, _ = s.store.NativeResultWait(row.ID, row.Ordinal)
	}
	return state
}

func inlineJobResult(doc canonical.Doc) any {
	if inline := doc.Sub("result").Str("inline_result"); inline != "" {
		if decoded, err := base64.StdEncoding.DecodeString(inline); err == nil {
			var typed any
			if json.Unmarshal(decoded, &typed) == nil {
				return typed
			}
		}
	}
	return nil
}

// parseStamp reads the authority's own RFC3339Nano timestamps. An unreadable one answers
// the zero time, which the caller renders as "no elapsed clock" rather than as zero
// milliseconds — the difference between "not known" and "instant".
func parseStamp(s string) time.Time {
	t, err := time.Parse(time.RFC3339Nano, s)
	if err != nil {
		return time.Time{}
	}
	return t
}

// queuedMS is the time a request spent WAITING: from submission to its first dispatch,
// or — for a request that never dispatched — to its terminal, or to now while it waits.
func queuedMS(row records.Request, attempts []records.Attempt, terminalAt string) int64 {
	began := parseStamp(row.CreatedAt)
	if began.IsZero() {
		return 0
	}
	end := time.Now().UTC()
	if len(attempts) > 0 {
		if dispatched := parseStamp(attempts[0].DispatchedAt); !dispatched.IsZero() {
			end = dispatched
		}
	} else if settledRequestState(row.State) {
		if terminal := parseStamp(terminalAt); !terminal.IsZero() {
			end = terminal
		}
	}
	return max(end.Sub(began).Milliseconds(), 0)
}

// executionMS is the time a request spent RUNNING: each attempt from its dispatch to its
// close (or to the terminal, or to now while it runs), summed. Queue time is never in it.
func executionMS(row records.Request, attempts []records.Attempt, terminalAt string) int64 {
	var total int64
	now := time.Now().UTC()
	terminal := parseStamp(terminalAt)
	for _, attempt := range attempts {
		began := parseStamp(attempt.DispatchedAt)
		if began.IsZero() {
			continue
		}
		end := now
		switch {
		case attempt.ClosedAt != "":
			if closed := parseStamp(attempt.ClosedAt); !closed.IsZero() {
				end = closed
			}
		case settledRequestState(row.State) && !terminal.IsZero():
			end = terminal
		}
		total += max(end.Sub(began).Milliseconds(), 0)
	}
	return total
}

func settledRequestState(state string) bool {
	switch state {
	case "succeeded", "failed", "canceled", "refused", "abandoned":
		return true
	default:
		return false
	}
}

// ------------------------------------------------------------------------- cancel

func (s *Server) cancelJob(w http.ResponseWriter, r *http.Request) {
	row, ok := s.jobRow(w, r)
	if !ok {
		return
	}
	actor := requestActor(r)
	if s.machineJobControl(w, r, row, "cancel", actor) {
		return
	}
	retaining, problem := s.store.RequestRetaining(row)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	if retaining && row.State != "finalizing" {
		if e := s.orchestrator.CancelRetainedRequest(row.ID, actor); e != nil {
			s.refuseTyped(w, r, e)
			return
		}
		updated, e := s.store.RequestRow(row.ID)
		if e != nil || updated == nil {
			s.refuse(w, r, http.StatusInternalServerError, "internal", "canceled request cannot be read", "")
			return
		}
		s.ok(w, r, http.StatusAccepted, s.jobStateOf(*updated))
		return
	}
	if status := contractStatus(row.State); status == "completed" || status == "failed" || status == "canceled" {
		s.ok(w, r, http.StatusOK, s.jobStateOf(row))
		return
	}
	attempts, e := s.store.Attempts(row.ID)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if len(attempts) == 0 {
		// A QUEUED job has nothing running, and cancelling it is still a real act: it
		// leaves the queue and settles, so a client that asked never has to wonder.
		if e := s.orchestrator.CancelQueued(row.ID, actor); e != nil {
			s.refuseTyped(w, r, e)
			return
		}
		updated, e := s.store.RequestRow(row.ID)
		if e != nil || updated == nil {
			s.refuse(w, r, http.StatusInternalServerError, "internal",
				"the queued job was canceled and cannot be read back", "")
			return
		}
		s.ok(w, r, http.StatusOK, s.jobStateOf(*updated))
		return
	}
	last := attempts[len(attempts)-1]
	if row.ModelTransfer != nil && row.State == "finalizing" {
		if e := s.orchestrator.CancelModelTransferFinalization(row.ID, actor); e != nil {
			s.refuseTyped(w, r, e)
			return
		}
		s.ok(w, r, http.StatusAccepted, map[string]any{
			"job_id": row.ID, "attempt": last.Attempt, "status": "cancel_requested",
			"note": "destination finalization is stopping; provider teardown precedes the canceled terminal",
		})
		return
	}
	if last.State == "closed" || last.State == "dispatch_aborted" {
		if e := s.orchestrator.CancelQueued(row.ID, actor); e != nil {
			s.refuseTyped(w, r, e)
			return
		}
		updated, e := s.store.RequestRow(row.ID)
		if e != nil || updated == nil {
			s.refuse(w, r, http.StatusInternalServerError, "internal",
				"the queued job was canceled and cannot be read back", "")
			return
		}
		s.ok(w, r, http.StatusOK, s.jobStateOf(*updated))
		return
	}
	if last.State == "terminal" {
		s.refuse(w, r, http.StatusConflict, "terminal_ack_pending",
			"the current job attempt has a terminal whose retry/settlement projection is not acknowledged yet",
			"retry cancellation after the terminal ack")
		return
	}
	grace := orchestrator.ClientCancelGraceMS
	if v := r.URL.Query().Get("grace_ms"); v != "" {
		if n, err := strconv.ParseUint(v, 10, 64); err == nil && n <= 120000 {
			grace = n
		}
	}
	if e := s.orchestrator.CancelClient(row.ID, uint64(last.Attempt), grace, actor); e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	s.ok(w, r, http.StatusAccepted, map[string]any{
		"job_id": row.ID, "attempt": last.Attempt, "status": "cancel_requested",
		"note": "the attempt's own journaled terminal settles it; watch the event stream",
	})
}
