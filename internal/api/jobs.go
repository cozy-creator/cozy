package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
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
	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/modeltransfer"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
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
	Package        string          `json:"package"`
	Function       string          `json:"function"`
	Input          json.RawMessage `json:"input"`
	InstallID      string          `json:"install_id,omitempty"`
	Release        string          `json:"release,omitempty"`
	Rental         bool            `json:"rental,omitempty"`
	RentalRequired bool            `json:"rental_required,omitempty"`
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
	Trees       []string               `json:"trees,omitempty"`
	LocalAssets []records.AssetBinding `json:"local_assets,omitempty"`
}

// JobHandle is the 202 answer.
type JobHandle struct {
	Number        int64  `json:"number"`
	JobID         string `json:"job_id"`
	Status        string `json:"status"`
	Attempt       uint64 `json:"attempt"`
	Package       string `json:"package"`
	Function      string `json:"function"`
	Repo          string `json:"publication_repo"`
	StatusURL     string `json:"status_url"`
	CancelURL     string `json:"cancel_url"`
	EventsURL     string `json:"events_url"`
	QueuePosition *int   `json:"queue_position,omitempty"`
	QueueDepth    *int   `json:"queue_depth,omitempty"`
	Replay        bool   `json:"idempotent_replay"`
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
			"the submission is not one closed JSON object: "+detail, "")
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
	if (len(sub.LocalAssets) > 0 || len(sub.Trees) > 0 || sub.Worker != "" || len(sub.Models) > 0 ||
		sub.ModelTransfer != nil) &&
		!s.cliAuthenticated(r) {
		s.refuse(w, r, http.StatusForbidden, "cli_credential_required",
			"trees name host filesystem directories and require the OS-protected CLI credential",
			"use `cozy run --input-tree <ref>=<dir>`; this build exposes no browser tree-upload route")
		return
	}
	if len(sub.LocalAssets) > 0 {
		unlock := inputasset.Guard()
		defer unlock()
	}
	key := strings.TrimSpace(r.Header.Get("Idempotency-Key"))
	existing, e := s.store.RequestByIdempotencyKey(key)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if existing != nil && (existing.State == "canceled" || existing.State == "failed" || existing.State == "refused") {
		released, e := s.store.ReleaseCanceledIdempotencyKey(key)
		if e != nil {
			s.refuseTyped(w, r, e)
			return
		}
		if released {
			existing = nil
		}
	}
	var spec orchestrator.Submission
	if existing != nil {
		spec, e = replayJobSubmission(sub, *existing)
	} else {
		if sub.Rental || sub.RentalRequired {
			unlock := localpackage.Guard()
			defer unlock()
		}
		spec, e = s.resolveJob(r.Context(), sub)
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
	if existing == nil {
		spec.Assets, e = s.stageAssets(spec.Assets)
		if e != nil {
			s.refuseTyped(w, r, e)
			return
		}
		if len(spec.Assets) > 0 {
			defer func() {
				if problem := inputasset.DropUnowned(s.layout, s.store, spec.Assets); problem != nil {
					fmt.Fprintf(s.log, "job input cleanup deferred: %s\n", problem.Message)
				}
			}()
		}
	}
	spec.IdemKey = key
	digest, e := jobSubmissionDigest(spec)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	spec.BodyDigest = digest

	// NO CAPACITY IS A STATE, never a refusal — this route cannot answer "busy". The
	// orchestrator records the row, queues it and makes the worker resident; several jobs
	// submitted at once queue against ONE worker and drain in submission order
	// (owner directive, decisions #394 / cr-019).
	recorded, fresh, e := s.recordSubmission(spec, existing == nil)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	jobID, attempt := recorded.ID, uint64(recorded.Ordinal)
	if fresh {
		defer s.activateRecorded(recorded)
	}
	handle := JobHandle{
		Number: recorded.Number, JobID: jobID, Status: contractStatus(recorded.State), Attempt: attempt,
		Package: recorded.Package, Function: recorded.Entrypoint,
		Repo:      home.ScratchRepo(recorded.Org, recorded.ID),
		StatusURL: "/v1/local/jobs/" + jobID,
		CancelURL: "/v1/local/jobs/" + jobID + "/cancel",
		EventsURL: "/v1/requests/" + jobID + "/events",
		Replay:    !fresh,
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

func replayJobSubmission(sub JobSubmission,
	recorded records.Request,
) (orchestrator.Submission, *exit.Error) {
	payload := []byte(sub.Input)
	if len(payload) == 0 {
		payload = []byte("{}")
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
	return orchestrator.Submission{Kind: "job", Package: packageName,
		Entrypoint: function, Payload: payload, Org: org,
		InstallID: recorded.InstallID, Release: recorded.Release,
		LocalPackageDigest: recorded.LocalPackageDigest,
		PlanID:             recorded.PlanID, Outputs: outputs, WeightsOutputs: weightsOutputs,
		NeedsAccelerator: recorded.NeedsAccelerator, Trees: trees, Worker: recorded.Worker,
		Rental: sub.Rental || sub.RentalRequired, RentalRequired: sub.RentalRequired,
		Models: models, Assets: sub.LocalAssets, ModelTransfer: transfer, ProducerParams: params}, nil
}

// resolveJob turns package+function into the orchestrator's Submission. The
// `job_descriptor_id` is resolved HERE, from the installed package's own PackageInterface —
// a client never names a digest, exactly as it never names a binding plan id.
func (s *Server) resolveJob(ctx context.Context, sub JobSubmission) (orchestrator.Submission, *exit.Error) {
	if sub.ModelTransfer != nil && sub.Package == "" && sub.Function == "" {
		if sub.Rental || sub.RentalRequired {
			return orchestrator.Submission{}, exit.Named(exit.Unavailable,
				"model_transfer.rented_pass_through_unavailable",
				"rented pass-through has no typed TensorFS source profiles")
		}
		return orchestrator.Submission{Kind: "job", Package: "cozy/platform",
			Entrypoint: "model-pass-through", Payload: []byte("{}"), Org: "local",
			PlanID: "sha256:" + strings.Repeat("0", 64), ModelTransfer: sub.ModelTransfer}, nil
	}
	out := orchestrator.Submission{
		Kind: "job", Package: sub.Package, Entrypoint: sub.Function,
		Payload: []byte(sub.Input), Org: strings.TrimSpace(sub.Org), Assets: sub.LocalAssets,
		Release: sub.Release,
		Rental:  sub.Rental || sub.RentalRequired, RentalRequired: sub.RentalRequired,
		Worker: sub.Worker, Models: append([]orchestrator.ModelRef(nil), sub.Models...),
		ModelTransfer: sub.ModelTransfer,
	}
	if len(out.Payload) == 0 {
		out.Payload = []byte("{}")
	}
	if out.Org == "" {
		out.Org = "local"
	}
	if e := validOrg(out.Org); e != nil {
		return out, e
	}
	if s.packages == nil {
		return out, exit.Unavailablef("this Cozy daemon resolves no packages")
	}
	if sub.Worker != "" && !sub.Rental && !sub.RentalRequired {
		return out, exit.Named(exit.Validation, "rental.job_worker_without_rental",
			"a pinned remote worker requires rental authorization")
	}
	if out.Rental {
		if strings.HasPrefix(sub.Package, "local/") {
			refreshed, editable, _, refreshProblem := s.refreshPackage(sub.Package)
			if refreshProblem != nil {
				return out, refreshProblem
			}
			if !editable {
				return out, exit.Named(exit.Conflict, "local_package_install_invalid",
					"%s is not one editable local package", sub.Package)
			}
			return s.resolveLocalJob(ctx, sub, out, refreshed)
		}
		if sub.InstallID != "" || sub.Release == "" || len(sub.Trees) > 0 {
			return out, exit.Named(exit.Validation, "rental.job_release_incomplete",
				"remote jobs require one exact published release and no local input trees")
		}
		logical, job, problem := s.packages.ResolveRemoteJob(
			sub.Package, sub.Release, sub.Function, sub.Models,
			sub.ModelTransfer.HasAcquisition())
		if problem != nil {
			return out, problem
		}
		if problem := validateInputs(job, &out); problem != nil {
			return out, problem
		}
		out.PlanID, out.Outputs = logical.DescriptorID, logical.Outputs
		out.WeightsOutputs, out.NeedsAccelerator = logical.WeightsOutputs, logical.NeedsAccelerator
		out.ProducerParams = logical.ProducerParams
		out.Models = append([]orchestrator.ModelRef(nil), logical.Models...)
		return out, nil
	}
	refreshed, editable, _, refreshProblem := s.refreshPackage(sub.Package)
	if refreshProblem != nil {
		return out, refreshProblem
	}
	if editable {
		sub.InstallID = refreshed
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
	} else {
		jobs, e = s.packages.Jobs(sub.Package)
	}
	if e != nil {
		return out, e
	}
	names := make([]string, 0, len(jobs))
	for _, job := range jobs {
		names = append(names, job.Name)
		if job.Name != sub.Function {
			continue
		}
		out.PlanID = job.DescriptorID
		out.Outputs = job.Outputs
		out.WeightsOutputs = job.WeightsOutputs
		out.NeedsAccelerator = job.NeedsAccelerator
		out.ProducerParams = job.ModelParams
		if problem := validateInputs(&launch.Entrypoint{Name: job.Name, Kind: "job", Request: job.Request}, &out); problem != nil {
			return out, problem
		}
	}
	if out.PlanID == "" {
		return out, exit.Named(exit.NotFound, "unknown_job",
			"%s registers no job named %q", sub.Package, sub.Function).
			WithRemedy("it registers: %s", strings.Join(names, ", "))
	}
	for _, pair := range sub.Trees {
		ref, dir, ok := strings.Cut(pair, "=")
		if !ok || ref == "" || dir == "" {
			return out, exit.Usagef("%q is not ref=<directory>", pair)
		}
		abs, err := filepath.Abs(dir)
		if err != nil {
			return out, exit.Usagef("%q is not a resolvable directory: %s", dir, err)
		}
		out.Trees = append(out.Trees, ref+"="+abs)
	}
	return out, nil
}

func (s *Server) resolveLocalJob(ctx context.Context, sub JobSubmission,
	out orchestrator.Submission, installID string,
) (orchestrator.Submission, *exit.Error) {
	if len(sub.Trees) > 0 {
		return out, exit.Named(exit.Validation, "rental.job_local_tree_unsupported",
			"remote jobs cannot grant directories from the Creator host")
	}
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
		out.WeightsOutputs, out.NeedsAccelerator = job.WeightsOutputs, job.NeedsAccelerator
		out.ProducerParams = job.ModelParams
		if problem := validateInputs(&launch.Entrypoint{Name: job.Name, Kind: "job", Request: job.Request}, &out); problem != nil {
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
	out.LocalPackageDigest = revision.Digest
	return out, nil
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
	if assets := assetIdentities(spec.Assets); len(assets) > 0 {
		doc["assets"] = assets
	}
	if spec.Rental {
		doc["rental"] = true
		doc["release"] = spec.Release
	}
	if spec.RentalRequired {
		doc["rental_required"] = true
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
	Number   int64  `json:"number"`
	JobID    string `json:"job_id"`
	Status   string `json:"status"`
	Package  string `json:"package"`
	Function string `json:"function"`
	Attempt  uint64 `json:"attempt"`
	Attempts int    `json:"attempts"`
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
	ModelSources []ModelSourceState `json:"model_sources,omitempty"`
	Publication  *PublicationRef    `json:"publication,omitempty"`
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
	s.ok(w, r, http.StatusOK, s.jobStateOf(row))
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
	state := JobState{
		Number: row.Number, JobID: row.ID, Status: contractStatus(row.State), Package: row.Package,
		Function: row.Entrypoint, Attempt: uint64(row.Ordinal),
		Requeues: row.Requeues, RetryBudget: orchestrator.MaxRequeues,
		Outputs: []MediaRef{}, CreatedAt: row.CreatedAt,
		EventsURL: "/v1/requests/" + row.ID + "/events",
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
			if statuses, statusProblem := s.store.ModelTransferSourceStatuses(row.ID); statusProblem == nil {
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
	if inline := doc.Sub("result").Str("inline_result"); inline != "" {
		if decoded, err := base64.StdEncoding.DecodeString(inline); err == nil {
			var typed any
			if json.Unmarshal(decoded, &typed) == nil {
				state.Result = typed
			}
		}
	}
	return state
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
