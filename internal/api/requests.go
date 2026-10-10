package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"github.com/cozy-creator/cozy/internal/archive"
	"io"
	"net/http"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
	"github.com/cozy-creator/cozy/internal/scratch"
)

// The request-level contract: a 202 handle out of submit, a lifecycle document out of
// status, and explicit cancellation. Cozy implements this surface today; other hosts
// require their own implementation and conformance proof.

// Submission is the request body. `input` is the package's own typed payload and is
// carried VERBATIM: the orchestrator digests exactly the bytes the client sent, so a
// re-submit under one key compares the same request identity.
type Submission struct {
	MachineEndpoint *machineendpoint.Endpoint `json:"machine_endpoint,omitempty"`
	RequestedRental string                    `json:"requested_rental,omitempty"`
	Package         string                    `json:"package"`
	Function        string                    `json:"function"`
	Input           json.RawMessage           `json:"input"`
	// InstallID is the immutable local install selected by the CLI. It is opaque to users;
	// omitting it asks the daemon to resolve the active package pointer.
	InstallID string   `json:"install_id,omitempty"`
	Release   string   `json:"release,omitempty"`
	Outputs   []string `json:"outputs,omitempty"`
	PlanID    string   `json:"plan_id,omitempty"`
	// LocalAssets is the local API's out-of-band input set. Source paths remain borrowed
	// and are reverified before local reads or remote upload. The typed payload carries
	// only its opaque reference.
	LocalAssets    []records.AssetBinding  `json:"local_assets,omitempty"`
	Rental         bool                    `json:"rental,omitempty"`
	RentNew        bool                    `json:"rent_new,omitempty"`
	RentalRequired bool                    `json:"rental_required,omitempty"`
	Models         []orchestrator.ModelRef `json:"models,omitempty"`
	// OutputDirectory is the caller's --out. Empty means the package's own store under
	// outputs/, which every run exports to.
	OutputDirectory string `json:"output_directory,omitempty"`
	// AttentionKernel is an optional developer execution-path pin.
	AttentionKernel string `json:"attention_kernel,omitempty"`
	// Ignored names the undeclared fields the client dropped from Input; the run records
	// them as its warning.
	Ignored    []string `json:"ignored_fields,omitempty"`
	AttemptKey string   `json:"-"`
}

// Handle is the 202 answer: the request's id and where to go next. Verbatim from the
// AsyncRequestHandle cozy.art consumes, plus `attempt` (which a local client can act on
// because there is no queue-position story to tell yet).
type Handle struct {
	Number        int64  `json:"number"`
	RequestID     string `json:"request_id"`
	Status        string `json:"status"`
	Attempt       uint64 `json:"attempt"`
	StatusURL     string `json:"status_url"`
	ResponseURL   string `json:"response_url"`
	CancelURL     string `json:"cancel_url"`
	EventsURL     string `json:"events_url"`
	QueuePosition *int   `json:"queue_position,omitempty"`
	QueueDepth    *int   `json:"queue_depth,omitempty"`
	// Replay is true when this key was already recorded: the SAME request answers, and
	// nothing new was started. A client that retried a timed-out POST needs to know it
	// did not create a second execution, and inferring it from equal ids is a guess.
	Replay bool `json:"idempotent_replay"`
}

func (s *Server) submit(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, MaxBody+1))
	if err != nil {
		s.refuse(w, r, http.StatusBadRequest, "unreadable_body",
			"the request body could not be read: "+err.Error(), "")
		return
	}
	if len(body) > MaxBody {
		s.refuse(w, r, http.StatusRequestEntityTooLarge, "body_too_large",
			"a submission body is bounded at "+strconv.Itoa(MaxBody)+" bytes",
			"typed input only; assets arrive as references")
		return
	}
	var sub Submission
	decoder := json.NewDecoder(bytes.NewReader(body))
	err = decoder.Decode(&sub)
	var trailing any
	if err == nil {
		err = decoder.Decode(&trailing)
		if err == nil {
			err = fmt.Errorf("multiple JSON values")
		}
	}
	if err != io.EOF {
		s.refuse(w, r, http.StatusBadRequest, "malformed_body",
			"the submission is not a JSON object: "+err.Error(), "")
		return
	}
	if sub.Package == "" || sub.Function == "" {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request",
			"a submission names a package and a function",
			`{"package":"org/name","function":"denoise","input":{…}}`)
		return
	}
	if s.scopedEndpoint != "" && (sub.MachineEndpoint == nil || sub.MachineEndpoint.Name() != s.scopedEndpoint) {
		s.refuseTyped(w, r, exit.Named(exit.Credential, "machine.endpoint_scope", "foreground controller requires its explicitly pinned target; no local fallback is allowed"))
		return
	}
	if problem := launch.ValidateAttentionOverride(sub.AttentionKernel); problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	if (len(sub.LocalAssets) > 0 || sub.OutputDirectory != "" || sub.RequestedRental != "" || sub.MachineEndpoint != nil) && !s.cliAuthenticated(r) {
		s.refuse(w, r, http.StatusForbidden, "cli_credential_required",
			"local assets and output directories require the OS-protected CLI credential",
			"use `cozy run --asset <field-path>=<file>`; this build exposes no browser asset-upload route")
		return
	}
	if sub.MachineEndpoint != nil {
		if sub.Rental || sub.RentalRequired || sub.RentNew || sub.RequestedRental != "" {
			s.refuseTyped(w, r, exit.Usagef("an explicit owned machine endpoint cannot also select or buy a rental"))
			return
		}
		validator, ok := s.machineExecutions.(interface {
			ValidateEndpoint(context.Context, *machineendpoint.Endpoint) *exit.Error
		})
		if !ok {
			s.refuseTyped(w, r, exit.Named(exit.Unavailable, "machine.endpoint_unavailable", "this controller cannot execute explicit machine endpoints; nothing was submitted"))
			return
		}
		if problem := validator.ValidateEndpoint(r.Context(), sub.MachineEndpoint); problem != nil {
			s.refuseTyped(w, r, problem)
			return
		}
	}
	if problem := validateModelAdapters(sub.Models); problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	selectedHub, e := s.hubOf(r)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	key := strings.TrimSpace(r.Header.Get("Idempotency-Key"))
	existing, e := s.store.RequestByIdempotencyKey(key)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if existing != nil && s.requestHub(*existing) != selectedHub {
		s.refuseTyped(w, r, idempotencyHubConflict(key, s.requestHub(*existing), selectedHub))
		return
	}
	var spec orchestrator.Submission
	var inputStage *scratch.Dir
	if existing != nil {
		// Resolve an existing key from the durable request identity, not from resources
		// retained only while it can execute. The caller's asset claims are still hashed
		// below, so a different body conflicts, but its source is not
		// opened merely to answer an already-recorded request.
		spec = replaySubmission(sub, *existing)
		if link, problem := s.store.MachineExecution(existing.ID); problem == nil && link != nil {
			spec.MachineExecutionObserver = true
		}
		// The digest covers the DERIVED export rows, and a replay resolves no
		// entrypoint to re-derive them from. The recorded export row is that exact
		// derivation, frozen at first submit; the caller still asserts directory and
		// payload hash, so a replay naming a different --out conflicts loudly. Without
		// this, "repeat the same idempotency key" — the product's own durable-export
		// remedy — could never match its own recorded body.
		if spec.OutputExport != nil {
			recorded, e := s.store.OutputExportOf(existing.ID)
			if e != nil {
				s.refuseTyped(w, r, e)
				return
			}
			if recorded != nil {
				spec.OutputExport.Outputs = append([]records.OutputExportEntry(nil), recorded.Outputs...)
			}
		}
	} else {
		var lease *install.Lease
		defer func() { lease.Release() }()
		spec, e = s.resolvePlan(r.Context(), selectedHub, sub, &lease)
		if e != nil {
			s.refuseTyped(w, r, e)
			return
		}
		spec.Warnings = ignoredWarnings(sub.Package, sub.Function, sub.Ignored)
		// The DAEMON is the process that publishes --out, so the daemon probes the
		// destination — here, where the path is finally resolved, before an input is
		// bound or a row recorded. A doomed export refuses in milliseconds instead
		// of after GPU minutes.
		if spec.OutputExport != nil {
			if e := resultfiles.Preflight(spec.OutputExport.Directory); e != nil {
				s.refuseTyped(w, r, e)
				return
			}
		}
		spec.Assets, e = bindAssets(spec.Assets)
		if e != nil {
			s.refuseTyped(w, r, e)
			return
		}
		// An inference root is Runtime's to execute on whichever machine runs it: its root
		// bytes are frozen for native staging before anything is recorded.
		if s.machineExecutions == nil {
			s.refuseTyped(w, r, exit.Unavailablef("this Cozy daemon runs no machines"))
			return
		}
		spec.MachineExecutionObserver = true
		inputStage, e = s.freezeMachineInputs(&spec, &launch.Entrypoint{Name: spec.Entrypoint})
		defer inputStage.Release()
		if e != nil {
			s.refuseTyped(w, r, e)
			return
		}
	}
	spec.IdemKey, spec.Hub = key, selectedHub
	if sub.MachineEndpoint != nil {
		spec.MachineEndpoint = sub.MachineEndpoint
		spec.Worker = sub.MachineEndpoint.Name()
	}
	// THE BODY DIGEST is over the whole submission the key names, not over the payload
	// alone: one key that named a different FUNCTION must conflict as loudly as one
	// that named different input. The canonical preimage is the digest's subject, so
	// two clients that spell the same submission differently still agree.
	digest, e := submissionDigest(spec)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	spec.BodyDigest = digest

	recorded, fresh, e := s.recordSubmission(spec)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if inputStage != nil && recorded.ID == spec.RequestID {
		inputStage.Detach()
	}
	attempt := uint64(recorded.Ordinal)
	if fresh {
		defer s.activateRecorded(recorded)
	}
	handle := s.handleOf(recorded, attempt)
	handle.Replay = !fresh
	if handle.Replay && (handle.Status == "completed" || handle.Status == "failed" || handle.Status == "canceled") {
		s.orchestrator.RetryOutputExport(recorded.ID)
	}
	// 202 means "this host started work", and it may not be said twice for one key.
	// A recorded answer is a 200 with the same handle: the client already has it.
	status := http.StatusAccepted
	if handle.Replay {
		status = http.StatusOK
	}
	s.ok(w, r, status, handle)
}

// idempotencyHubConflict refuses a key already naming a request on another hub: one
// key cannot mean two submissions.
func idempotencyHubConflict(key, recorded, selected string) *exit.Error {
	return exit.Named(exit.Conflict, "request.hub_conflict",
		"idempotency key %q already names a request on Tensorhub %s, not %s", key, recorded, selected).
		WithRemedy("use a new idempotency key, or repeat the request with --tensorhub=%s", recorded)
}

func replaySubmission(sub Submission, recorded records.Request) orchestrator.Submission {
	// Resolve omitted install/release defaults from the original admission. They
	// were filled during first submission and remain part of the same identity.
	installID, release := sub.InstallID, sub.Release
	if installID == "" {
		installID = recorded.InstallID
	}
	if release == "" {
		release = recorded.Release
	}
	planID := sub.PlanID
	if planID == "" {
		planID = recorded.PlanID
	}
	outputs := append([]string(nil), sub.Outputs...)
	if len(outputs) == 0 && recorded.Outputs != "" {
		outputs = strings.Split(recorded.Outputs, ",")
	}
	assets := append([]records.AssetBinding(nil), sub.LocalAssets...)
	sort.Slice(assets, func(i, j int) bool { return assets[i].FieldPath < assets[j].FieldPath })
	models := append([]orchestrator.ModelRef(nil), sub.Models...)
	if len(models) == 0 {
		models = append(models, recorded.Models...)
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Slot < models[j].Slot })
	payload := []byte(sub.Input)
	if len(payload) == 0 {
		payload = []byte("{}")
	}
	return orchestrator.Submission{
		Package: sub.Package, Entrypoint: sub.Function, Payload: payload,
		Outputs: outputs, PlanID: planID, Worker: recorded.Worker, Assets: assets,
		InstallID: installID, Release: release,
		LocalInstallationID: recorded.LocalInstallationID,
		Rental:              sub.Rental || sub.RentalRequired || sub.RentNew || sub.RequestedRental != "",
		RentalRequired:      sub.RentalRequired || sub.RentNew || sub.RequestedRental != "",
		RequestedRental:     sub.RequestedRental, RentNew: sub.RentNew,
		Models: models, NeedsAccelerator: recorded.NeedsAccelerator,
		OutputDirectory: sub.OutputDirectory,
		// Replays compare the caller's requested execution path with the original.
		AttentionKernel: sub.AttentionKernel,
	}
}

// submissionDigest is the canonical identity of one submission. It uses the SAME writer
// the protocol documents use, so the digest a client can reproduce is the digest the
// authority recorded.
func assetIdentity(bindings []records.AssetBinding) []canonical.Value {
	bindings = append([]records.AssetBinding(nil), bindings...)
	sort.Slice(bindings, func(i, j int) bool { return bindings[i].FieldPath < bindings[j].FieldPath })
	assets := make([]canonical.Value, 0, len(bindings))
	for _, asset := range bindings {
		assets = append(assets, map[string]canonical.Value{
			"field_path": asset.FieldPath,
			"digest":     asset.Digest,
			"length":     asset.Length,
			"media_type": asset.MediaType,
			"order":      int64(asset.Order),
		})
	}
	return assets
}

func submissionDigest(spec orchestrator.Submission) (string, *exit.Error) {
	doc := map[string]canonical.Value{
		"kind":       "serve",
		"package":    spec.Package,
		"function":   spec.Entrypoint,
		"install_id": spec.InstallID,
		"release":    spec.Release,
		"input":      base64.StdEncoding.EncodeToString(spec.Payload),
		"outputs":    strings.Join(spec.Outputs, ","),
	}
	if spec.MachineEndpoint != nil {
		doc["machine_endpoint"] = spec.MachineEndpoint.Name()
	}
	assets := assetIdentity(spec.Assets)
	if len(assets) > 0 {
		doc["assets"] = assets
	}
	if spec.Rental {
		doc["rental"] = true
	}
	if spec.RentNew {
		doc["rent_new"] = true
	}
	if spec.RentalRequired {
		doc["rental_required"] = true
	}
	if spec.AttentionKernel != "" {
		doc["attention_kernel"] = spec.AttentionKernel
	}
	if spec.RequestedRental != "" {
		doc["requested_rental"] = spec.RequestedRental
	}
	if len(spec.Models) > 0 {
		refs := append([]orchestrator.ModelRef(nil), spec.Models...)
		sort.Slice(refs, func(i, j int) bool { return refs[i].Slot < refs[j].Slot })
		models := make([]canonical.Value, 0, len(refs))
		for _, model := range refs {
			models = append(models, modelSelectionIdentity(model))
		}
		doc["models"] = models
	}
	// The caller's explicit --out is part of what the key names; the derived default and
	// the PackageInterface-derived entries are facts of the package, not of the submission.
	if spec.OutputDirectory != "" {
		doc["output_directory"] = spec.OutputDirectory
	}
	// The pinned rental is NOT in it: a worker id says WHERE the same work runs, and two
	// submissions of one key that differ only in placement are the same request. What the
	// row records is what a requeue re-derives; the digest is about meaning.
	data, err := canonical.Write(doc)
	if err != nil {
		return "", exit.Internalf("cannot canonicalize the submission: %s", err)
	}
	spelled, err := canonical.Spell(canonical.Digest(data))
	if err != nil {
		return "", exit.Internalf("cannot digest the submission: %s", err)
	}
	return spelled, nil
}

func (s *Server) handleOf(row records.Request, attempt uint64) Handle {
	base := "/v1/requests/" + row.ID
	h := Handle{
		Number: row.Number, RequestID: row.ID, Status: s.publicStatusOf(row), Attempt: attempt,
		StatusURL: base, ResponseURL: base, CancelURL: base + "/cancel",
		EventsURL: base + "/events",
	}
	if h.Status == "queued" {
		if position, depth := s.orchestrator.QueueState(row.ID); position > 0 {
			h.QueuePosition, h.QueueDepth = &position, &depth
		}
	}
	return h
}

// contractStatus projects the authority's own state names onto the contract's vocabulary.
// The two are deliberately not the same word list: the store records what happened to a
// row, the contract says what a client should do next.
func contractStatus(state string) string {
	return records.PublicRunStatus(state)
}

func (s *Server) publicStatusOf(row records.Request) string {
	return records.PublicRunStatus(row.State)
}

// resolvePlan turns the client's package/function into the orchestrator's Submission.
// The plan id is resolved through the LOCAL resolver — the same object `start` uses — so
// a client never names a plan digest and a submission can never bind a binding this host
// did not install. The local install it resolves is leased into lease, which the caller
// releases once the request is recorded.
func (s *Server) resolvePlan(ctx context.Context, hub string, sub Submission, lease **install.Lease) (orchestrator.Submission, *exit.Error) {
	out := orchestrator.Submission{
		Package: sub.Package, Entrypoint: sub.Function, Payload: []byte(sub.Input),
		Outputs: sub.Outputs, PlanID: sub.PlanID, Assets: sub.LocalAssets,
		Release: sub.Release,
		Rental:  sub.Rental || sub.RentalRequired || sub.RentNew || sub.RequestedRental != "", RentalRequired: sub.RentalRequired || sub.RentNew || sub.RequestedRental != "",
		RequestedRental: sub.RequestedRental, RentNew: sub.RentNew,
		Models:          append([]orchestrator.ModelRef(nil), sub.Models...),
		AttentionKernel: sub.AttentionKernel,
		OutputDirectory: sub.OutputDirectory,
	}
	if problem := s.validateRequestedRental(out.RequestedRental); problem != nil {
		return out, problem
	}
	if len(out.Payload) == 0 {
		out.Payload = []byte("{}")
	}
	// Only immutable package metadata is validated here. The scheduler chooses and records
	// the machine after admission.
	if s.packages == nil {
		return out, exit.Unavailablef("this Cozy daemon resolves no packages")
	}
	if strings.HasPrefix(sub.Package, "local/") {
		if sub.InstallID != "" {
			var problem *exit.Error
			if *lease, problem = install.LeaseInstall(s.layout, s.store, sub.InstallID); problem != nil {
				return out, problem
			}
			return s.resolveLocalServing(ctx, sub, out, sub.InstallID)
		}
		refreshed, editable, leased, refreshProblem := s.leaseRefreshed(sub.Package)
		*lease = leased
		if refreshProblem != nil {
			return out, refreshProblem
		}
		if !editable {
			return out, exit.Named(exit.Conflict, "local_package_install_invalid",
				"%s is not one editable local package", sub.Package)
		}
		return s.resolveLocalServing(ctx, sub, out, refreshed)
	}
	release, e := s.installedRelease(hub, sub.Package, sub.Release, sub.InstallID)
	if e != nil {
		return out, e
	}
	sub.Release, sub.InstallID, out.Release = release, "", release
	if sub.Release == "" {
		return out, exit.Unavailablef("remote execution requires one exact Tensorhub package release")
	}
	logical, entrypoint, e := s.packages.ResolveRemoteRelease(hub,
		sub.Package, sub.Release, sub.Function, sub.Models)
	if e != nil {
		return out, e
	}
	if logical.Package != sub.Package || logical.Release != sub.Release {
		return out, exit.Named(exit.Conflict, "install_package_mismatch",
			"queued remote release does not match the resolved Tensorhub release")
	}
	out.PlanID = logical.PlanID
	out.Models = append([]orchestrator.ModelRef(nil), logical.Models...)
	out.NeedsAccelerator = logical.NeedsAccelerator
	if len(out.Outputs) == 0 {
		out.Outputs = logical.Outputs
	}
	if e := validateInputs(entrypoint, &out); e != nil {
		return out, e
	}
	if e := s.deriveOutputExport(entrypoint, &out); e != nil {
		return out, e
	}
	return out, nil
}

// installedRelease is the published release a local install pins. A machine prepares the
// release itself, so the install names only which release a caller meant.
func (s *Server) installedRelease(origin, pkg, release, installID string) (string, *exit.Error) {
	if installID == "" {
		return release, nil
	}
	installed, problem := s.store.Install(installID)
	if problem != nil {
		return "", problem
	}
	if installed == nil {
		return release, nil
	}
	if !installed.PublishedAt(origin) || installed.Package != pkg || release != "" && release != installed.Version {
		return "", exit.Named(exit.Conflict, "install_package_mismatch", "install %s is not %s %s from Tensorhub", installID, pkg, release)
	}
	return installed.Version, nil
}

// deriveOutputExport records where this run's result files will be published: the
// caller's --out, else the package's own store under outputs/. A callable with no result
// files exports nothing and records no obligation.
func (s *Server) deriveOutputExport(entrypoint *launch.Entrypoint, out *orchestrator.Submission) *exit.Error {
	paths := launch.AssetPaths(entrypoint.Result)
	// Jobs export every declared asset, including generic files. Native model
	// artifacts and weight outputs remain separate and have no asset paths. A list of
	// assets exports each product published into it.
	if len(paths) == 0 && len(launch.AssetLists(entrypoint.Result)) == 0 && (out.Kind == "job" || len(out.Outputs) == 0) {
		return nil
	}
	intent, problem := s.outputExportIntent(out)
	if problem != nil {
		return problem
	}
	fixed := 0
	for _, outputID := range out.Outputs {
		if !strings.HasSuffix(outputID, ".*") {
			fixed++
		}
	}
	if out.Kind != "job" && len(paths) != fixed {
		return exit.Named(exit.Validation, "output_export_set_mismatch",
			"package result declares %d asset paths for %d granted outputs", len(paths), len(out.Outputs))
	}
	granted := make(map[string]bool, len(out.Outputs))
	for _, outputID := range out.Outputs {
		granted[outputID] = true
	}
	intent.Outputs = make([]records.OutputExportEntry, 0, len(paths))
	for _, outputID := range paths {
		if !granted[outputID] {
			return exit.Named(exit.Validation, "output_export_set_mismatch",
				"result asset %s is not an exact granted output", outputID)
		}
		spec, ok := launch.ResultAssetSpec(entrypoint, outputID)
		if ok && spec.Kind == "tree" {
			spec.MediaTypes = []string{resultfiles.TreeMediaType}
		}
		if !ok || len(spec.MediaTypes) != 1 {
			return exit.Named(exit.Validation, "output_export_media_type_ambiguous",
				"result asset %s must declare exactly one media type", outputID)
		}
		intent.Outputs = append(intent.Outputs, records.OutputExportEntry{
			OutputID: outputID, MediaType: spec.MediaTypes[0],
		})
	}
	out.OutputExport = intent
	return nil
}

func (s *Server) outputExportIntent(out *orchestrator.Submission) (*records.OutputExportIntent, *exit.Error) {
	directory := out.OutputDirectory
	if directory == "" {
		directory = s.layout.PackageOutputs(out.Package)
	}
	if !filepath.IsAbs(directory) || filepath.Clean(directory) != directory || len(directory) > 4096 {
		return nil, exit.Named(exit.Validation, "output_export_directory_malformed",
			"output directory must be one canonical absolute path")
	}
	return &records.OutputExportIntent{Directory: directory}, nil
}

func (s *Server) resolveLocalServing(ctx context.Context, sub Submission,
	out orchestrator.Submission, installID string,
) (orchestrator.Submission, *exit.Error) {
	spec, problem := s.packages.ResolveInstall(installID, sub.Models)
	if problem != nil {
		return out, problem
	}
	if spec.Preparation != nil {
		return s.resolvePendingServing(ctx, sub, out, spec)
	}
	placement := spec.Placement
	if placement.Package != sub.Package {
		return out, exit.Named(exit.Conflict, "install_package_mismatch",
			"install %s serves %s, not %s", installID, placement.Package, sub.Package)
	}
	if len(out.Models) == 0 {
		// The editable install froze its model selection; the rental carries that exact
		// intent (repo, release, lane, manifest) to the pod as its unpublished package placement.
		out.Models = append([]orchestrator.ModelRef(nil), placement.Models...)
	}
	planID, outputs, problem := placementPlan(placement, sub.Function)
	if problem != nil {
		return out, problem
	}
	entrypoint, needsAccelerator, problem := s.packages.Entrypoint(installID, sub.Function)
	if problem != nil {
		return out, problem
	}
	if problem := validateInputs(entrypoint, &out); problem != nil {
		return out, problem
	}
	out.NeedsAccelerator = needsAccelerator
	revision, problem := s.packages.LocalInstallation(installID)
	if problem != nil {
		return out, problem
	}
	out.InstallID, out.PlanID = installID, planID
	out.Release = revision.Release
	out.LocalInstallationID = revision.ID
	if len(out.Outputs) == 0 {
		out.Outputs = outputs
	}
	if problem := s.deriveOutputExport(entrypoint, &out); problem != nil {
		return out, problem
	}
	return out, nil
}

// leaseRefreshed refreshes pkg and, when it is editable, leases the install it now runs
// until the request that references it is recorded: neither the editable watcher nor
// another process's install may reclaim it in between. An install reclaimed before the
// lease was taken is resolved again.
func (s *Server) leaseRefreshed(pkg string) (string, bool, *install.Lease, *exit.Error) {
	for {
		refreshed, editable, _, problem := s.refreshPackage(pkg)
		if problem != nil || !editable {
			return refreshed, editable, nil, problem
		}
		lease, problem := install.LeaseInstall(s.layout, s.store, refreshed)
		if problem == nil || problem.Code != exit.NotFound {
			return refreshed, true, lease, problem
		}
	}
}

// resolvePendingServing retains only code and model selections. The worker's
// completed preparation supplies the binding digest before ordinary dispatch.
func (s *Server) resolvePendingServing(ctx context.Context, sub Submission, out orchestrator.Submission,
	spec orchestrator.WorkerLaunchSpec) (orchestrator.Submission, *exit.Error) {
	placement := spec.Placement
	if placement.Package != sub.Package || spec.Preparation == nil {
		return out, exit.Named(exit.Conflict, "install_package_mismatch", "unprepared install names another package")
	}
	entrypoint, needsAccelerator, problem := s.packages.Entrypoint(placement.InstallID, sub.Function)
	if problem != nil {
		return out, problem
	}
	if len(entrypoint.Models) > 0 && len(out.Models) == 0 {
		return out, exit.Named(exit.Validation, "package_model_binding_required", "%s/%s requires exact model selections", sub.Package, sub.Function)
	}
	if problem := validateInputs(entrypoint, &out); problem != nil {
		return out, problem
	}
	out.InstallID, out.Release = placement.InstallID, placement.Release
	out.NeedsAccelerator = needsAccelerator
	if !spec.Preparation.Published {
		revision, problem := s.packages.LocalInstallation(placement.InstallID)
		if problem != nil {
			return out, problem
		}
		out.LocalInstallationID = revision.ID
	}
	if len(out.Outputs) == 0 {
		out.Outputs = launch.OutputSlots(entrypoint.Result)
	}
	if problem := s.deriveOutputExport(entrypoint, &out); problem != nil {
		return out, problem
	}
	return out, nil
}

// validateInputs checks the payload and every local asset against the entrypoint that
// will run it — the same law for a local install and a rental's frozen PackageInterface.
func validateInputs(entrypoint *launch.Entrypoint, out *orchestrator.Submission) *exit.Error {
	if problem := entrypoint.RequirePublic(); problem != nil {
		return problem
	}
	// Undeclared fields pass through: the Runtime that decodes the request drops them and
	// records its warning, as the CLI did for the ones it dropped before submitting.
	if _, _, e := launch.ValidatePayload(out.Package, entrypoint, out.Payload); e != nil {
		return e
	}
	for index := range out.Assets {
		asset := &out.Assets[index]
		assetSpec, ok := launch.AssetSpecForMedia(entrypoint, asset.FieldPath, asset.MediaType)
		if !ok || assetSpec.MaxBytes <= 0 || asset.Length > assetSpec.MaxBytes {
			return exit.Named(exit.Validation, "input_asset_bound",
				"input asset %s is %d B and its pinned field admits %d B",
				asset.FieldPath, asset.Length, assetSpec.MaxBytes)
		}
		if !assetSpec.AcceptsMediaType(asset.MediaType) {
			return exit.Named(exit.Validation, "input_asset_media_type",
				"input asset %s is %s and its pinned field does not admit that media type",
				asset.FieldPath, asset.MediaType)
		}
		asset.MaxBytes = assetSpec.MaxBytes
	}
	return launch.ValidateAssetCounts(entrypoint, out.Assets)
}

func ignoredWarnings(pkg, function string, ignored []string) []records.Warning {
	if len(ignored) == 0 {
		return nil
	}
	return []records.Warning{launch.IgnoredWarning(pkg+"/"+function, ignored)}
}

func placementPlan(placement orchestrator.DesiredPlacement, function string) (string, []string, *exit.Error) {
	for _, entrypoint := range placement.Entrypoints {
		if entrypoint.Name != function {
			continue
		}
		return entrypoint.Digest, append([]string(nil), entrypoint.Outputs...), nil
	}
	return "", nil, exit.New(exit.NotFound, "%s has no function %q", placement.Package, function).
		WithRemedy("GET /v1/local/packages lists the functions this target serves")
}

func bindAssets(assets []records.AssetBinding) ([]records.AssetBinding, *exit.Error) {
	if len(assets) == 0 {
		return nil, nil
	}
	out := make([]records.AssetBinding, 0, len(assets))
	seen := map[string]bool{}
	for _, asset := range assets {
		if seen[asset.FieldPath] {
			return nil, exit.New(exit.Validation,
				"input asset field %q was supplied more than once", asset.FieldPath)
		}
		seen[asset.FieldPath] = true
		maxBytes := asset.MaxBytes
		if maxBytes <= 0 {
			maxBytes = inputasset.MaxBytes
		}
		bound, e := inputasset.Bind(asset, maxBytes)
		if e != nil {
			return nil, e
		}
		out = append(out, bound)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].FieldPath < out[j].FieldPath })
	return out, nil
}

// ------------------------------------------------------------------------- status

// Lifecycle is the status document. Verbatim from RequestLifecycleResponse plus the
// fields a local client has and a cloud one does not need to presign: the typed result,
// the visible media by OPAQUE id, and the triage handle.
type Lifecycle struct {
	// Hub is the Tensorhub origin the request belongs to.
	Hub              string                `json:"hub,omitempty"`
	MachineExecution *MachineExecutionView `json:"machine_execution,omitempty"`
	Number           int64                 `json:"number"`
	Kind             string                `json:"kind"`
	RequestID        string                `json:"request_id"`
	Status           string                `json:"status"`
	Package          string                `json:"package"`
	Function         string                `json:"function"`
	Attempt          uint64                `json:"attempt"`
	Attempts         int                   `json:"attempts"`
	QueuedMS         int64                 `json:"queued_ms"`
	ExecutionMS      int64                 `json:"execution_ms"`
	ExecutionKnown   bool                  `json:"execution_known"`
	AttemptWallMS    int64                 `json:"attempt_wall_ms,omitempty"`
	ProgressStage    string                `json:"progress_stage,omitempty"`
	StageFraction    *float64              `json:"stage_fraction,omitempty"`
	OverallFraction  *float64              `json:"overall_fraction,omitempty"`
	Position         *int64                `json:"position,omitempty"`
	Total            *int64                `json:"total,omitempty"`
	// RemainingMS is read-only compatibility with older daemons; future-stage time is unknown.
	RemainingMS *int64 `json:"remaining_ms,omitempty"`
	// StageRemainingMS estimates only the current counted stage; later stages are unknown.
	StageRemainingMS *int64   `json:"stage_remaining_ms,omitempty"`
	StepMS           *float64 `json:"step_ms,omitempty"`
	// ProgressUnit names what Position and Total count ("bytes" for a transfer or
	// conversion stage); ProgressRate is that unit per second, as Runtime measured it.
	ProgressUnit string   `json:"progress_unit,omitempty"`
	ProgressRate *float64 `json:"progress_rate,omitempty"`
	// The PREPARATION facts (cl-121). A queued request is not idle — it is acquiring a
	// machine, booting one, or landing model bytes on it — and these say which, with
	// whatever advancement that phase actually has. Absent for anything that has left
	// the queue, and absent field by field for a phase whose producer measured nothing:
	// a missing rate means "not measured", never zero.
	Phase          string                               `json:"phase,omitempty"`
	PhaseMachine   string                               `json:"phase_machine,omitempty"`
	WaitingFor     *orchestrator.WaitingRun             `json:"waiting_for,omitempty"`
	PhaseDetail    string                               `json:"phase_detail,omitempty"`
	PhaseModels    []orchestrator.ModelDownloadProgress `json:"phase_models,omitempty"`
	RentalProgress *orchestrator.RentalProgress         `json:"rental_progress,omitempty"`
	// RentalBoot is the boot of the rental a queued request waits on, as its Hub last said.
	RentalBoot *hub.RentalBoot `json:"rental_boot,omitempty"`
	// ObservationLost is a running run whose machine this computer cannot hear now: its
	// progress is the last it heard, and the run is being attached again.
	ObservationLost *records.ObservationLoss `json:"observation_lost,omitempty"`

	// WaitReason is why a queued request last said it waits (its newest park), verbatim.
	WaitReason       string         `json:"wait_reason,omitempty"`
	PhaseElapsedMS   *int64         `json:"phase_elapsed_ms,omitempty"`
	PhaseMovedBytes  *int64         `json:"phase_moved_bytes,omitempty"`
	PhaseTotalBytes  *int64         `json:"phase_total_bytes,omitempty"`
	PhaseRate        *float64       `json:"phase_rate_bytes_per_second,omitempty"`
	PhaseSampleAgeMS *int64         `json:"phase_sample_age_ms,omitempty"`
	PhaseRemainingMS *int64         `json:"phase_remaining_ms,omitempty"`
	ResponseURL      string         `json:"response_url"`
	Metrics          map[string]any `json:"metrics,omitempty"`
	ErrorType        string         `json:"error_type,omitempty"`
	// ErrorCode is the originating component's stable code (for example a pod's
	// `insufficient_storage`) when the failure did not originate in Creator.
	ErrorCode      string `json:"error_code,omitempty"`
	Error          string `json:"error,omitempty"`
	Retaining      bool   `json:"retaining,omitempty"`
	RetryAvailable bool   `json:"retry_available,omitempty"`
	StoppedEventID int64  `json:"stopped_event_id,omitempty"` // an older daemon's blocked stop
	// CanceledBy is the recorded actor behind a canceled run (cl-108): the explicit
	// `cozy run cancel`, a caller-authored --timeout, `cozy down --all` — never blank
	// for a run this daemon canceled on request.
	CanceledBy string     `json:"canceled_by,omitempty"`
	Result     any        `json:"result,omitempty"`
	Outputs    []MediaRef `json:"outputs"`
	// Output is the run's output items, each at its latest revision (OpenAI Responses' `output`).
	Output []records.OutputItem `json:"output,omitempty"`
	Triage *TriageRef           `json:"triage,omitempty"`
	Rental bool                 `json:"rental,omitempty"`
	// Machine names the selected or executing venue. Queued status does not imply
	// that the machine has accepted an attempt; it may still be preparing this request.
	// After execution, the recorded machine survives rental cleanup.
	Machine string `json:"machine"`
	// Caller affinity names a requested venue, not a remotely queued attempt.
	RequestedRental  string `json:"requested_rental,omitempty"`
	RequestedMachine string `json:"requested_machine,omitempty"`
	// RentalID is the raw immutable rental id (`pr-…`) behind Machine, for machines and
	// --full readers; it never appears in default human output.
	RentalID      string           `json:"rental_id,omitempty"`
	CreatedAt     string           `json:"created_at"`
	QueuePosition *int             `json:"queue_position,omitempty"`
	QueueDepth    *int             `json:"queue_depth,omitempty"`
	OutputExport  *OutputExportRef `json:"output_export,omitempty"`
}

// DisplayMachine also reads the selection facts older daemons expose before an
// attempt exists. An executing or historical venue always takes precedence.
func (l Lifecycle) DisplayMachine() string {
	if l.Machine != "" || l.Status != "queued" {
		return l.Machine
	}
	if l.RequestedMachine != "" {
		return l.RequestedMachine
	}
	return l.PhaseMachine
}

// OutputExportRef is where a run's result files go and whether they are there yet.
// Paths is filled once published: a file is named by its own content digest, which
// nothing knows before the terminal.
type OutputExportRef struct {
	Directory string   `json:"directory"`
	State     string   `json:"state"`
	ErrorCode string   `json:"error_code,omitempty"`
	Error     string   `json:"error,omitempty"`
	Paths     []string `json:"paths"`
}

// MediaRef is how bytes are named in EVERY document this API emits: an opaque id and its
// facts. There is no `path` field, and there is no route that would accept one.
type MediaRef struct {
	OutputID string `json:"output_id"`
	MediaID  string `json:"media_id"`
	URL      string `json:"url"`
	MimeType string `json:"mime_type"`
	Length   int64  `json:"length"`
	Digest   string `json:"digest"`
	// Path is the output's stable file in the run's outputs folder, when it has one.
	Path string `json:"path,omitempty"`
}

// TriageRef is the handle onto a retained bundle. `attempt_key` is the orchestrator's own
// opaque key; `subject_id` is the runtime's. Neither is a path.
type TriageRef struct {
	AttemptKey string `json:"attempt_key"`
	SubjectID  string `json:"subject_id"`
	URL        string `json:"url"`
	Length     int64  `json:"length"`
	Kept       bool   `json:"kept"`
	Fault      string `json:"fault,omitempty"`
}

func (s *Server) getRequest(w http.ResponseWriter, r *http.Request) {
	reference := r.PathValue("id")
	row, e := s.store.RequestByReference(reference)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if row == nil {
		s.refuse(w, r, http.StatusNotFound, "not_found",
			"no request "+reference+" on this host", "")
		return
	}
	var problem *exit.Error
	// observe=false answers from the journal the daemon's own observer keeps, without
	// dialing the machine: a caller about to act on the run needs only its record.
	if r.URL.Query().Get("observe") != "false" {
		problem = s.refreshMachineExecution(r.Context(), *row)
	}
	if current, e := s.store.RequestRow(row.ID); e == nil && current != nil {
		current.Number = row.Number
		row = current
	}
	life := s.lifecycleOf(*row)
	if life.MachineExecution != nil && problem != nil {
		life.MachineExecution.ObservationError = problem.Message
	}
	s.ok(w, r, http.StatusOK, life)
}

func (s *Server) lifecycleOf(row records.Request) Lifecycle {
	life := s.lifecycleFacts(row)
	life.Hub = s.requestHub(row)
	if life.Error == "" && life.Status == "failed" {
		if errType, errCode, errText, problem := s.store.SettledFailure(row.ID); problem == nil {
			life.ErrorType, life.ErrorCode, life.Error = errType, errCode, errText
		} else {
			life.ErrorType, life.Error = problem.ErrName(), problem.Message
		}
	}
	return life
}

func (s *Server) lifecycleFacts(row records.Request) Lifecycle {
	if link, problem := s.store.MachineExecution(row.ID); problem == nil && link != nil {
		state := s.machineJobState(row, link)
		machine := s.machineWord(row, link)
		kind := "invocation"
		if row.IsJob() {
			kind = "job"
		}
		life := Lifecycle{Number: row.Number, Kind: kind, RequestID: row.ID, Status: state.Status,
			Package: row.Package, Function: row.Entrypoint, Attempt: state.Attempt, Attempts: state.Attempts,
			ExecutionMS: state.ExecutionMS, ExecutionKnown: state.ExecutionKnown,
			AttemptWallMS: state.AttemptWallMS, QueuedMS: state.QueuedMS,
			Result: state.Result, Error: state.Error, ErrorType: state.ErrorType, ErrorCode: state.ErrorCode, Outputs: state.Outputs, Output: state.Output,
			Rental: row.Rental, RentalID: row.Worker, Machine: machine, CreatedAt: row.CreatedAt,
			ResponseURL: "/v1/requests/" + row.ID, MachineExecution: state.MachineExecution,
			Retaining: state.Retaining, RetryAvailable: state.RetryAvailable, StoppedEventID: state.StoppedEventID}
		if life.Status == "queued" {
			if phase, ok := s.orchestrator.PhaseOf(row.ID); ok {
				fillPhase(&life, phase)
			} else if phase, ok := s.orchestrator.RentalPhase(link.MachineID); ok {
				fillPhase(&life, phase)
			}
			s.fillWait(&life)
		}
		s.fillLifecycleProgress(&life, row)
		if life.Status == "in_progress" {
			life.ObservationLost, _ = s.store.ObservationLost(row.ID)
		}
		s.fillGPUWait(&life, row)
		life.OutputExport = s.outputExportOf(row.ID)
		if life.Status == "canceled" {
			// A canceled run says WHO (cl-108), whichever machine ran it.
			if actor, _, _, problem := s.store.CancelAttribution(row.ID); problem == nil {
				life.CanceledBy = actor
			}
		}
		if awaiting, problem := s.store.MachinePublicationsAwaitingOwner(row.ID); problem == nil && len(awaiting) > 0 {
			life.Phase, life.PhaseDetail = orchestrator.PhaseOwnerReconciliation, awaiting[0].Publication
		}
		return life
	}
	kind := "invocation"
	if row.IsJob() {
		kind = "job"
	}
	life := Lifecycle{
		Number: row.Number, Kind: kind, RequestID: row.ID,
		Status: s.publicStatusOf(row), Package: row.Package,
		Function: row.Entrypoint, Attempt: uint64(row.Ordinal),
		ResponseURL: "/v1/requests/" + row.ID, CreatedAt: row.CreatedAt,
		Outputs: []MediaRef{}, Rental: row.Rental, RentalID: row.Worker,
	}
	life.Retaining, _ = s.store.RequestRetaining(row)
	life.RetryAvailable = s.store.RetainedRetryAvailable(row)
	if row.RequestedRental != "" {
		life.RequestedRental, life.RequestedMachine = row.RequestedRental, row.Machine
	}
	if life.Status == "queued" {
		if position, depth := s.orchestrator.QueueState(row.ID); position > 0 {
			life.QueuePosition, life.QueueDepth = &position, &depth
		}
		if phase, ok := s.orchestrator.QueuePhase(row.ID); ok {
			fillPhase(&life, phase)
		}
		s.fillWait(&life)
	}
	s.fillLifecycleProgress(&life, row)
	life.OutputExport = s.outputExportOf(row.ID)
	attempts, _ := s.store.Attempts(row.ID)
	life.Attempts = len(attempts)
	life.Machine = s.machineOf(row, len(attempts) > 0)
	life.Machine = life.DisplayMachine()
	if life.Status == "canceled" {
		// A canceled run says WHO (cl-108). The actor rides the durable cancellation
		// events; a queued cancel also has no attempt row, so its cause lives only there.
		if actor, errType, errText, problem := s.store.CancelAttribution(row.ID); problem == nil {
			life.CanceledBy = actor
			if life.Error == "" && errText != "" {
				life.ErrorType, life.Error = errType, errText
			}
		}
	}
	terminalAt, _ := s.store.TerminalEventAt(row.ID)
	life.QueuedMS = queuedMS(row, attempts, terminalAt)
	life.ExecutionMS = executionMS(row, attempts, terminalAt)
	outs, _ := s.store.VisibleOutputs(row.ID)
	for _, o := range outs {
		life.Outputs = append(life.Outputs, MediaRef{
			OutputID: o.OutputID, MediaID: o.MediaID, URL: "/v1/media/" + o.MediaID,
			MimeType: o.MimeType, Length: o.Length, Digest: o.Digest,
		})
	}
	if len(attempts) == 0 {
		return life
	}
	last := attempts[len(attempts)-1]
	if last.TriageSubject != "" {
		life.Triage = &TriageRef{
			AttemptKey: last.AttemptKey, SubjectID: last.TriageSubject,
			URL:    "/v1/local/attempts/" + last.AttemptKey + "/triage",
			Length: last.TriageLength, Kept: last.TriageKept,
		}
	}
	if len(last.TerminalBody) == 0 {
		return life
	}
	doc, err := archive.Read(last.TerminalBody, archive.TerminalBody)
	if err != nil {
		return life
	}
	metrics := doc.Sub("metrics")
	life.Metrics = map[string]any{
		"runtime_ms": metrics.Int("runtime_ms"), "queue_ms": metrics.Int("queue_ms"),
		"handler_ms": metrics.Int("handler_ms"), "device_lease_ms": metrics.Int("device_lease_ms"),
		"finalization_ms":          metrics.Int("finalization_ms"),
		"peak_device_memory_bytes": metrics.Int("peak_device_memory_bytes"),
	}
	if last.TerminalStatus != "SUCCEEDED" {
		life.ErrorType, life.Error = last.TerminalCause, last.SafeMessage
	}
	// The typed result rides the terminal INLINE (cl-001's proof). It is base64 in the
	// canonical document; the contract hands the client the decoded JSON so a browser
	// never decodes a transport encoding to read a result.
	if inline := doc.Sub("result").Str("inline_result"); inline != "" {
		if decoded, err := base64.StdEncoding.DecodeString(inline); err == nil {
			var typed any
			if json.Unmarshal(decoded, &typed) == nil {
				life.Result = typed
			}
		}
	}
	return life
}

// machineOf reads the machine word RECORDED on the request (cl-107): stamped the moment
// the run is bound to a rental, shown during preparation, and kept as
// history after the rental row is gone. The raw rental id never renders here;
// pre-migration history no surviving rental row can name stays blank. A run with no
// rental binding that attempted ran on a worker this host spawned: `local`.
func (s *Server) machineOf(row records.Request, attempted bool) string {
	// Unpinning keeps the old name as history. It is not a current destination.
	if row.Machine != "" && (attempted || row.Worker != "" || row.RequestedRental != "") {
		return row.Machine
	}
	if row.Worker == "" && attempted {
		return machines.Local
	}
	return ""
}

func (s *Server) listRequests(w http.ResponseWriter, r *http.Request) {
	limit := 50
	var before int64
	if raw := r.URL.Query().Get("before"); raw != "" {
		value, err := strconv.ParseInt(raw, 10, 64)
		if err != nil || value < 1 {
			s.refuse(w, r, http.StatusBadRequest, "invalid_before", "before must be a positive run number", "use the oldest run number from the preceding page")
			return
		}
		before = value
	}
	if v := r.URL.Query().Get("limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 && n <= 500 {
			limit = n
		}
	}
	state := ""
	switch strings.TrimSpace(r.URL.Query().Get("status")) {
	case "", "any":
	case "queued":
		state = "queued"
	case "in_progress":
		state = "in_progress"
	case "completed":
		state = "completed"
	case "failed":
		state = "failed"
	case "canceled":
		state = "canceled"
	case "paused", "pausing", "canceling", "finalizing":
		state = strings.TrimSpace(r.URL.Query().Get("status"))
	default:
		s.refuse(w, r, http.StatusBadRequest, "invalid_status",
			"unknown status filter", "any | queued | in_progress | completed | failed | canceled")
		return
	}
	hub, e := s.hubOf(r)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if r.URL.Query().Get("hubs") == "all" {
		hub = ""
	}
	packageName := strings.TrimSpace(r.URL.Query().Get("package"))
	rows, e := s.store.PublicRequestsBefore(state, packageName, hub, s.cfg.HubURL, limit, before)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	// older is how many matching runs this page leaves below it, so a listing that stops
	// here can say what it did not show.
	older := 0
	if len(rows) == limit {
		if older, e = s.store.PublicRequestsOlder(state, packageName, hub, s.cfg.HubURL, rows[len(rows)-1].Number); e != nil {
			s.refuseTyped(w, r, e)
			return
		}
	}
	out := make([]Lifecycle, 0, len(rows))
	for _, row := range rows {
		out = append(out, s.lifecycleOf(row))
	}
	s.ok(w, r, http.StatusOK, map[string]any{"requests": out, "count": len(out), "older": older})
}

// ------------------------------------------------------------------------- cancel

// cancelActor reads WHO is canceling from the request body. Cancellation is an
// attributed act (cl-108): a canceled run must always be able to say who ended it, so
// an unnamed caller is recorded as exactly that rather than as nothing.
func requestActor(r *http.Request) string {
	var body struct {
		Actor string `json:"actor"`
	}
	data, _ := io.ReadAll(io.LimitReader(r.Body, 1<<12))
	if len(data) > 0 {
		_ = json.Unmarshal(data, &body)
	}
	if actor := strings.TrimSpace(body.Actor); actor != "" {
		return actor
	}
	return "an unnamed api client"
}

func (s *Server) cancelRequest(w http.ResponseWriter, r *http.Request) {
	reference := r.PathValue("id")
	row, e := s.store.RequestByReference(reference)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if row == nil {
		s.refuse(w, r, http.StatusNotFound, "not_found", "no request "+reference+" on this host", "")
		return
	}
	actor := requestActor(r)
	if s.machineRequestControl(w, r, *row, "cancel", actor) {
		return
	}
	if status := contractStatus(row.State); status == "completed" || status == "failed" || status == "canceled" {
		s.ok(w, r, http.StatusOK, s.lifecycleOf(*row))
		return
	}
	// No machine executes it: the daemon's own model transfer, which has no attempt.
	if e := s.orchestrator.CancelQueued(row.ID, actor); e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	updated, e := s.store.RequestRow(row.ID)
	if e != nil || updated == nil {
		s.refuse(w, r, http.StatusInternalServerError, "internal",
			"the request was canceled and cannot be read back", "")
		return
	}
	s.ok(w, r, http.StatusOK, s.lifecycleOf(*updated))
}

// All execution venues expose the same work coordinates. Machine observations
// also retain their imported progress, so a new observer need not await a tick.
func (s *Server) fillLifecycleProgress(life *Lifecycle, row records.Request) {
	switch life.Status {
	case "completed":
		complete := 1.0
		life.OverallFraction = &complete
		return
	case "in_progress":
	case "failed", "canceled":
		if life.MachineExecution == nil {
			life.OverallFraction, _ = s.store.TerminalOverallFraction(row.ID, row.Ordinal)
			return
		}
	default:
		return
	}
	if life.MachineExecution == nil {
		return
	}
	samples, problem := s.store.MachineProgressSamples(row.ID, int64(life.Attempt))
	if problem != nil || len(samples) == 0 {
		return
	}
	progress, ok := orchestrator.DecodeProgressSnapshot(samples[0])
	if !ok {
		return
	}
	var previous orchestrator.ProgressSnapshot
	if len(samples) > 1 {
		previous, _ = orchestrator.DecodeProgressSnapshot(samples[1])
	}
	estimate, estimated := progress.EstimateStage(previous)
	life.OverallFraction = progress.OverallFraction
	if life.Status == "in_progress" {
		life.ProgressStage = progress.Stage
		life.StageFraction = progress.StageFraction
		life.Position = progress.Position
		life.Total = progress.Total
		life.StepMS = progress.StepMS
		life.ProgressUnit, life.ProgressRate = progress.Unit, progress.Rate
		if estimated {
			life.StageRemainingMS = &estimate.RemainingMS
		}
	}
}

// fillPhase projects what a queued run is waiting on: the phase, its machine and, for a
// transfer, the bytes moved and the rate.
func fillPhase(life *Lifecycle, phase orchestrator.PhaseObservation) {
	life.Phase, life.PhaseMachine = phase.Name, phase.Machine
	life.PhaseDetail, life.RentalProgress, life.RentalBoot = phase.Detail, phase.Rental, phase.Boot
	life.PhaseModels = phase.Models
	if !phase.At.IsZero() {
		age := phase.SampleAge().Milliseconds()
		life.PhaseSampleAgeMS = &age
	}
	life.WaitingFor = phase.WaitingFor
	if elapsed := phase.Elapsed(); elapsed > 0 {
		ms := elapsed.Milliseconds()
		life.PhaseElapsedMS = &ms
	}
	if phase.HasBytes {
		moved := int64(phase.Moved)
		life.PhaseMovedBytes = &moved
		if phase.Total > 0 {
			total := int64(phase.Total)
			life.PhaseTotalBytes = &total
		}
		if phase.Rate > 0 {
			rate := phase.Rate
			life.PhaseRate = &rate
		}
		if remaining, ok := phase.Remaining(); ok {
			ms := remaining.Milliseconds()
			life.PhaseRemainingMS = &ms
		}
	}
}

// fillWait names why a queued request waits, from its own newest park.
func (s *Server) fillWait(life *Lifecycle) {
	if wait, problem := s.store.CurrentWait(life.RequestID); problem == nil && wait != nil {
		life.WaitReason = wait.Reason
	}
}

// fillGPUWait says an active Runtime execution waits for GPUs, how many, and the first
// other run of this host holding them; when only this run's own calls hold them, how many.
func (s *Server) fillGPUWait(life *Lifecycle, row records.Request) {
	if life.Status != "queued" && life.Status != "in_progress" {
		return
	}
	wait, problem := s.store.MachineGPUWait(row.ID)
	if problem != nil || wait == nil {
		return
	}
	life.Phase, life.PhaseDetail = orchestrator.PhaseGPUWait, GPUWaitDetail(wait.Width, 0, false)
	own := 0
	for _, root := range wait.BlockedBy {
		if root == row.ID {
			own++
			continue
		}
		blocker, problem := s.store.RequestByReference(root)
		if problem == nil && blocker != nil && blocker.ParentRequestID == row.ID {
			own++
		} else if problem == nil && blocker != nil {
			life.WaitingFor = &orchestrator.WaitingRun{Number: blocker.Number, RequestID: blocker.ID}
			return
		}
	}
	if own > 0 && own == len(wait.BlockedBy) {
		held, _ := s.store.MachineGPUsHeld(row.ID)
		life.PhaseDetail = GPUWaitDetail(wait.Width, held, true)
	} else if len(wait.BlockedBy) > 0 {
		life.PhaseDetail += ", behind other work"
	}
}

// GPUWaitDetail is a GPU wait's detail: the devices the call needs and, when only its own
// run holds the GPUs, how many of them that run's other calls are using.
func GPUWaitDetail(width, held int, ownRun bool) string {
	detail := fmt.Sprintf("needs %d", width)
	switch {
	case ownRun && held > 0:
		detail += fmt.Sprintf(", %d in use by this run's other calls", held)
	case ownRun:
		detail += ", in use by this run's other calls"
	}
	return detail
}
