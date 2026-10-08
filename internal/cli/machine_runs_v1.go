package cli

import (
	"bytes"
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
	"github.com/cozy-creator/cozy/internal/runoutputs"
	"github.com/cozy-creator/cozy/internal/scratch"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// A run on a machine that serves cozy.machine.v1 is one Run call: its spec (code, payload,
// file inputs written first with Write, model choices, the run's Hub access) submits it once
// under the request's id, and the same call streams its log to the outcome. Resubmitting is
// attaching: the id is the idempotency, so nothing is frozen, closed or acknowledged. Outputs
// are read with Read into the run's outputs folder as their revisions land.

// errNoDescribe is a machine that cannot name a package's newest release and its interface
// (describe/1): the client reads them at the Hub instead.
var errNoDescribe = exit.Named(exit.Unavailable, "machine.describe_unsupported", "the machine cannot describe a release")

// loopV1 follows one run on its machine until it is settled.
func (m *machineRuns) loopV1(request records.Request) {
	defer m.enforceDeadlineV1(request)()
	lastError, delay, followed := "", time.Second, false
	for m.ctx.Err() == nil {
		current, problem := m.store.RequestRow(request.ID)
		if problem != nil || current == nil {
			return
		}
		link, problem := m.store.MachineExecution(request.ID)
		if problem != nil || link == nil || link.Abandoned || link.Lost {
			return
		}
		accepted, problem := m.store.RunV1(request.ID)
		if problem != nil {
			return
		}
		if accepted && link.Collected {
			return
		}
		if !accepted && (len(link.Receipt) > 0 || len(link.Submission) > 0) {
			// Archived worker submissions may already have run. A native submission would
			// create a second execution; end only this controller's tracking, saying why.
			if !records.Settled(current.State) {
				_, _ = m.store.AbandonMachineExecution(request.ID, "cozy",
					"this run was sent over cozy.worker.v1, which this cozy no longer speaks; whether it ran is unknown")
			}
			return
		}
		sent, problem := m.store.RunV1Marked(request.ID, records.RunV1Sent)
		if problem != nil {
			return
		}
		// Unsent work never starts after cancellation. Sent work still needs its machine's
		// outcome, even when acceptance was lost or a stopped local machine settled it here.
		if !accepted && (link.CancelRequested || records.Settled(current.State) || slices.Contains([]string{"pausing", "paused", "blocked"}, current.State)) {
			if !link.CancelRequested || !sent {
				return
			}
		}
		done, problem := m.stepV1(m.ctx, *current, link, accepted, false)
		if done {
			return
		}
		// stepV1 may have crossed the durable dispatch boundary before losing its reply.
		if marked, readProblem := m.store.RunV1Marked(request.ID, records.RunV1Sent); readProblem != nil {
			return
		} else {
			sent = marked
		}
		if problem != nil && problem.Message != lastError && m.ctx.Err() == nil {
			fmt.Fprintf(m.context.Out, "machine execution %s: %s\n", request.ID, problem.Message)
			lastError = problem.Message
			// Before first dispatch a local refusal can end the request. Once possibly sent,
			// missing credentials/source metadata cannot prove what its machine accepted.
			// A machine older than this cozy refused the work at its door, so it never started,
			// sent or not: the machine takes the Hub's target software and the work is sent once
			// more, or the request fails naming the next step.
			upgrade := problem.ErrName() == "machine.upgrade_required"
			if upgrade && !followed {
				followed = true
				if next := m.followTargetV1(link.MachineID, current.Hub); next == "" {
					lastError = ""
					continue
				} else {
					problem = exit.Named(problem.Code, problem.ErrName(), "%s; %s", problem.Message, next)
				}
			}
			if again, _ := m.store.RunV1(request.ID); !again && (!sent || upgrade) && !link.CancelRequested && permanentRefusal(problem) {
				failed, failure := m.store.FailQueuedRequest(request.ID, records.QueuedFailure(problem))
				if failed {
					return
				}
				if failure == nil {
					continue // a concurrent control changed the request; deliver its intent
				}
				problem = failure
			}
			if !accepted && !link.CancelRequested {
				parked := map[string]any{"reason": problem.Message, "wait": orchestrator.WaitRental}
				if current.Machine != "" {
					parked["waiting_on"] = current.Machine
				}
				_ = m.store.AppendEvent(request.ID, "request.parked", 0, parked)
			}
		}
		// A remote machine holding this run that cannot be reached is asked again only on new
		// evidence: a reader, the rental attaching again, or the daemon's restart. A timer
		// would dial a gone pod forever.
		if problem != nil && (accepted || sent) && !machines.IsLocal(link.MachineID) {
			return
		}
		switch {
		case problem == nil:
			delay = 0 // the stream ended without an outcome: attach again at once
		case problem.Message == lastError:
			delay = min(2*max(delay, time.Second), 5*time.Second)
		default:
			delay = time.Second
		}
		select {
		case <-m.ctx.Done():
			return
		case <-time.After(delay):
		}
	}
}

// permanentRefusal is a refusal resubmitting the same spec cannot change.
func permanentRefusal(problem *exit.Error) bool {
	return problem.Code != exit.Unavailable && problem.Code != exit.Deadline && problem.Code != exit.Canceled
}

// followTargetV1 updates a machine to the Hub's target software and answers "" once it runs
// other software, else the next step its owner takes.
func (m *machineRuns) followTargetV1(machine, origin string) string {
	if machines.IsLocal(machine) {
		changed, kept := m.machines.Host.FollowTarget(m.ctx, client(m.context.forHub(origin)))
		if changed {
			return ""
		}
		return cmp.Or(kept, "this machine runs the files it was installed from") +
			"; `cozy machine install` takes the Hub's target software, or name newer wheels with --runtime-wheel/--tensorfs-wheel"
	}
	name := machine
	if row, _ := m.store.RentalRow(machine); row != nil && row.MachineName != "" {
		name = row.MachineName
	}
	update, problem := m.updates.follow(machine)
	for problem == nil && update.Active() && m.ctx.Err() == nil {
		select {
		case <-m.ctx.Done():
		case <-time.After(time.Second):
		}
		update, problem = m.store.RuntimeUpdate(machine)
		if problem == nil && update == nil {
			problem = exit.New(exit.Conflict, "its update record is gone")
		}
	}
	var result runtimeUpdateResult
	switch {
	case problem != nil:
		return fmt.Sprintf("it could not take the Hub's target software (%s); `cozy rental update %s` retries", problem.Message, name)
	case update.State == "failed":
		return fmt.Sprintf("its update to the Hub's target software failed (%s); `cozy rental update %s` retries, or `cozy rental end %s` and rent a new machine", update.Error, name, name)
	case json.Unmarshal(update.Result, &result) == nil && !result.Unchanged:
		return ""
	}
	return fmt.Sprintf("%s runs Runtime %s (%s); update it to a release that serves this run: `cozy rental update %s --runtime-version <version> --tensorfs-version <version>`",
		name, result.From.Runtime, result.Note, name)
}

// catchUpV1 records what an accepted run's machine holds now, for a reader: its state and its
// log up to the head the machine names, without waiting for more.
func (m *machineRuns) catchUpV1(ctx context.Context, request records.Request) *exit.Error {
	link, problem := m.store.MachineExecution(request.ID)
	if problem != nil || link == nil || link.Collected || link.Abandoned {
		return problem
	}
	accepted, problem := m.store.RunV1(request.ID)
	if problem != nil {
		return problem
	}
	if !accepted && (len(link.Receipt) > 0 || len(link.Submission) > 0) {
		if !records.Settled(request.State) {
			_, problem = m.store.AbandonMachineExecution(request.ID, "cozy",
				"this run was sent over cozy.worker.v1, which this cozy no longer speaks; whether it ran is unknown")
		}
		return problem
	}
	_, problem = m.stepV1(ctx, request, link, true, true)
	return problem
}

// stepV1 places the run if it has no machine, connects, submits or attaches, and records the
// log until the stream ends; done once the outcome is recorded.
// `catchUp` ends it at the log's head as the machine named it when the stream opened.
func (m *machineRuns) stepV1(parent context.Context, request records.Request, link *records.MachineExecution, accepted, catchUp bool) (bool, *exit.Error) {
	ctx, stop := context.WithCancel(parent)
	defer stop()
	if !catchUp {
		m.mu.Lock()
		m.submitting[request.ID] = stop
		m.mu.Unlock()
		defer func() {
			m.mu.Lock()
			delete(m.submitting, request.ID)
			m.mu.Unlock()
		}()
	}
	request, problem := m.place(ctx, request, link)
	if problem != nil {
		return false, problem
	}
	began := time.Now()
	sent, problem := m.store.RunV1Marked(request.ID, records.RunV1Sent)
	if problem != nil {
		return false, problem
	}
	// Only a submission may start this computer's machine; observing attaches to a running one.
	dial := ctx
	if accepted || sent || link.CancelRequested {
		dial = machines.AttachOnly(ctx)
	}
	machine, problem := m.machines.DialV1(dial, link.MachineID, m.runHolder(request, "running"))
	if problem != nil {
		return false, problem
	}
	defer machine.Close()
	var stream grpc.ServerStreamingClient[v1.RunEvent]
	var first *v1.RunEvent
	if !accepted && sent && !link.CancelRequested {
		// Acceptance can outlive its reply. Attach before touching local source or credentials
		// for a new submission: that source may no longer be here, while the run still exists.
		observing, err := machine.Run(ctx, request.ID, uint64(max(link.RemoteCursor, 0)), nil)
		if err == nil {
			first, err = observing.Recv()
		}
		if err == nil {
			stream = observing
		} else if status.Code(err) != codes.NotFound {
			return false, machines.Transport(err)
		}
	}
	var spec *v1.RunSpec
	if stream == nil && !accepted && !link.CancelRequested {
		m.submissionStage(request.ID, "connect", link.MachineID, began)
		if spec, problem = m.specV1(ctx, request, machine); problem != nil {
			return false, problem
		}
	}
	if link.CancelRequested {
		if _, err := machine.Control(ctx, request.ID, v1.Action_ACTION_CANCEL); err != nil {
			return false, machines.Transport(err)
		}
	}
	began = time.Now()
	if spec != nil {
		if send, problem := m.store.MarkRunV1Sent(request.ID); problem != nil || !send {
			return false, problem
		}
	}
	if stream == nil {
		var err error
		stream, err = machine.Run(ctx, request.ID, uint64(max(link.RemoteCursor, 0)), spec)
		if err != nil {
			return m.runRefusalV1(request.ID, err, spec != nil && !sent)
		}
	}
	head, opened, terminal := uint64(0), false, false
	// Output files are read apart from the stream: progress and the outcome never wait on bytes.
	var fetch *fetcherV1
	if !catchUp {
		fetch = m.fetcherV1(ctx, request, machine)
		defer fetch.stop()
	}
	for {
		event := first
		first = nil
		var err error
		if event == nil {
			event, err = stream.Recv()
		}
		if errors.Is(err, io.EOF) {
			return false, nil
		}
		if err != nil {
			return m.runRefusalV1(request.ID, err, spec != nil && !sent && !opened)
		}
		if state := event.GetState(); state != nil && !opened {
			head, opened = state.Sequence, true // the stream's first frame names the log's head
			terminal = records.Settled(state.State)
		}
		if state := event.GetState(); state != nil && !accepted {
			if problem := m.store.AcceptRunV1(request.ID, link.MachineID, state); problem != nil {
				if problem.ErrName() == "machine_execution.acceptance_stale" {
					// Released from this machine and sent on: it must not run here too.
					_, _ = machine.Control(ctx, request.ID, v1.Action_ACTION_CANCEL)
				}
				return false, problem
			}
			accepted = true
			m.submissionStage(request.ID, "submit", "", began)
			if m.fleet != nil && m.fleet.owner != nil {
				m.fleet.owner.ForgetPhase(request.ID) // the machine reports the run from here on
			}
			if latest, problem := m.store.MachineExecution(request.ID); problem == nil && latest != nil && latest.CancelRequested {
				if _, err := machine.Control(ctx, request.ID, v1.Action_ACTION_CANCEL); err != nil {
					return false, machines.Transport(err)
				}
			}
		}
		if outcome := event.GetOutcome(); outcome != nil {
			observed := time.Now().UnixMilli()
			if fetch != nil {
				fetch.finish() // the newest revisions land, their progress shown
			}
			problem := m.collectV1(ctx, request, machine, outcome, event.AtMs, observed)
			// A collection the connection cut is not the run's end here: it is attached again.
			return problem == nil || problem.ErrName() != "machine_execution.transport_unavailable", problem
		}
		var held *records.Product
		if product := event.GetProduct(); product != nil {
			if held = m.productV1(ctx, machine, request, event.Sequence, product); held != nil && fetch != nil {
				fetch.want(*held, product.AppendedFrom)
			}
		}
		if problem := m.store.ObserveRunV1(request.ID, event, held); problem != nil {
			return false, problem
		}
		// A terminal snapshot is followed by its outcome even when our cursor already
		// reaches the head. Read that outcome to retry a previously refused collection.
		if catchUp && !terminal && (head <= uint64(max(link.RemoteCursor, 0)) || event.Sequence >= head) {
			return false, nil
		}
	}
}

// An explicit first-submission rejection can end a request without acceptance. Other
// failures, including a retry after a lost reply, leave the possible remote run unresolved.
func (m *machineRuns) runRefusalV1(id string, err error, first bool) (bool, *exit.Error) {
	problem := machines.Transport(err)
	if first {
		switch status.Code(err) {
		// UNIMPLEMENTED is a machine older than this cozy: the run loop updates it first.
		case codes.InvalidArgument, codes.FailedPrecondition, codes.AlreadyExists, codes.PermissionDenied, codes.Unauthenticated:
			failed, recordProblem := m.store.FailQueuedRequest(id, records.QueuedFailure(problem))
			if recordProblem != nil {
				return false, recordProblem
			}
			return failed, problem
		}
	}
	return false, problem
}

// place gives a run its machine once: a named rental, a rental the fleet selects, an explicit
// endpoint or this computer's machine.
func (m *machineRuns) place(ctx context.Context, request records.Request, link *records.MachineExecution) (records.Request, *exit.Error) {
	if link.MachineID == "" {
		machine := machines.Placement(request)
		if request.Rental {
			if machine == "" {
				machine = request.RequestedRental
			}
			if machine == "" {
				decision, line, problem := m.fleet.acquire(request)
				m.recordPlacement(request, decision, line)
				if problem != nil {
					return request, problem
				}
				if machine = decision.RentalID; machine == "" {
					return request, exit.Unavailablef("waiting to select a machine for execution")
				}
			}
			if pinned, problem := m.store.PinRental(request.ID, machine, nil); problem != nil || !pinned {
				if problem != nil {
					return request, problem
				}
				return request, exit.New(exit.Canceled, "submission stopped before machine selection")
			}
		}
		if problem := m.store.LinkMachineExecution(request.ID, machine); problem != nil {
			return request, problem
		}
		link.MachineID = machine
	}
	if request.Rental {
		current, problem := m.store.RequestRow(request.ID)
		if problem != nil || current == nil {
			return request, problem
		}
		request.Worker, request.Models, request.PlanID = link.MachineID, current.Models, current.PlanID
	}
	return request, nil
}

// specV1 is the run as its machine takes it. Its file inputs and unpublished code are written
// to the machine first (each resumes from what the machine holds); the Hub token and provider
// tokens ride in the spec, which the machine holds in memory for the run's preparation only.
func (m *machineRuns) specV1(ctx context.Context, request records.Request, machine *machines.V1) (*v1.RunSpec, *exit.Error) {
	spec := &v1.RunSpec{Kind: v1.RunKind_RUN_KIND_CALL, Entrypoint: request.Entrypoint, Payload: request.Payload,
		AttentionKernel: request.AttentionKernel}
	if request.LocalInstallationID != "" || strings.HasPrefix(request.Package, "local/") {
		// Only unpublished code names its owner: its org-relative defaults are the owner's.
		spec.Owner = m.runAccount(request)
	}
	revision, problem := m.store.BindingRevision(m.context.forHub(request.Hub).Cfg.HubURL)
	if problem != nil {
		return nil, problem
	}
	spec.BindingRevision = revision
	if problem := m.writeTreesV1(ctx, request, machine, spec); problem != nil {
		return nil, problem
	}
	if request.IsJob() {
		spec.Kind = v1.RunKind_RUN_KIND_JOB
		if request.ModelTransfer != nil && request.ModelTransfer.Destination != "" {
			// The machine publishes the job's weights outputs there itself.
			spec.WeightsDestination = request.ModelTransfer.Destination
		}
	}
	// One grant covers every repository the run may publish into: its weights destination and
	// those the owner consented to (--allow-upload).
	repositories, problem := m.store.RequestPublicationRepositories(request.ID)
	if problem != nil {
		return nil, problem
	}
	if spec.WeightsDestination != "" {
		repositories = append(repositories, spec.WeightsDestination)
	}
	if len(repositories) > 0 {
		if spec.Publication, problem = authorizeV1Publication(ctx, machine, request.Hub, repositories); problem != nil {
			return nil, problem
		}
	}
	began := time.Now()
	if request.LocalInstallationID != "" {
		revision, problem := m.capturedRevision(request)
		if problem != nil {
			return nil, problem
		}
		manifest, sent, err := machinev1.LocalSource(ctx, machine.Machine, revision)
		if err != nil {
			return nil, machines.Transport(err)
		}
		spec.Source = &v1.RunSpec_Local{Local: &v1.LocalSource{Manifest: manifest}}
		if sent {
			// Code the machine did not hold yet; held code reopens its installation as it is.
			m.submissionStage(request.ID, "package", revision.Package, began)
		}
	} else {
		spec.Source = &v1.RunSpec_Release{Release: &v1.Release{Package: request.Package, Release: request.Release}}
	}
	began = time.Now()
	for _, asset := range request.Assets {
		if asset.MediaType == resultfiles.TreeMediaType && asset.Snapshot != nil {
			// A Tree input: each member the run captured, then its manifest, which the input names.
			if problem := writeTreeV1(ctx, machine, asset.Snapshot); problem != nil {
				return nil, problem
			}
		} else {
			// The bytes the run captured at submission: its snapshot's member, else the file itself.
			path := asset.LocalPath
			if asset.Snapshot != nil && asset.Snapshot.Path != "" {
				path = filepath.Join(asset.Snapshot.Path+".files", strings.TrimPrefix(asset.Digest, "sha256:"))
			}
			object, err := machinev1.WriteFile(ctx, machine.Machine, path)
			if err != nil {
				return nil, machines.Transport(err)
			}
			if object.Digest != asset.Digest || int64(object.Length) != asset.Length {
				return nil, exit.Named(exit.Conflict, "input_changed", "%s changed since the run was submitted", path)
			}
		}
		spec.Inputs = append(spec.Inputs, &v1.InputFile{Field: asset.FieldPath, Digest: asset.Digest,
			Length: uint64(asset.Length), MediaType: asset.MediaType, Order: asset.Order})
	}
	if len(request.Assets) > 0 {
		m.submissionStage(request.ID, "inputs", fmt.Sprintf("%d input(s)", len(request.Assets)), began)
	}
	if spec.Models, problem = modelChoicesV1(request, request.Models); problem != nil {
		return nil, problem
	}
	// The selected Hub supplies package/model access independently of rental ownership.
	if request.Hub != "" {
		if access, problem := m.hubAccessV1(ctx, request.Hub, machine, request.LocalInstallationID == "" && request.InstallID == "" || len(request.Models) > 0); problem != nil {
			return nil, problem
		} else {
			spec.Hub = access
		}
	}
	providers := &v1.ProviderAccess{Huggingface: m.resolver.cfg.HuggingFaceToken.Reveal(), Civitai: m.resolver.cfg.CivitaiToken.Reveal()}
	if providers.Huggingface != "" || providers.Civitai != "" {
		spec.Providers = providers
	}
	return spec, nil
}

// modelChoicesV1 are a run's model choices as a v1 spec names them.
func modelChoicesV1(request records.Request, models []records.ModelRef) ([]*v1.ModelChoice, *exit.Error) {
	return orchestrator.ModelChoices(request, models)
}

// warmSetV1 is the machine's warm set with the selection's member added, changed or (`off`)
// removed; the other members are sent back as the machine reported them.
func warmSetV1(current []*v1.WarmItem, selection records.RentalInstallSelection) (*v1.WarmSet, *exit.Error) {
	set := &v1.WarmSet{}
	for _, item := range current {
		if item.GetRelease().GetPackage() == selection.Package && item.GetEntrypoint() == selection.Entrypoint {
			continue
		}
		item.Holds, item.HeldBack = "", ""
		set.Items = append(set.Items, item)
	}
	if selection.Warm == "off" {
		return set, nil
	}
	level := slices.Index(records.WarmLevels, selection.Warm)
	if level < 0 {
		return nil, exit.Usagef("--warm=%s is not one of %s or off", selection.Warm, strings.Join(records.WarmLevels, ", "))
	}
	models, problem := modelChoicesV1(records.Request{Package: selection.Package, Entrypoint: selection.Entrypoint}, selection.Models)
	if problem != nil {
		return nil, problem
	}
	set.Items = append(set.Items, &v1.WarmItem{Source: &v1.WarmItem_Release{Release: &v1.Release{Package: selection.Package, Release: selection.Release}},
		Entrypoint: selection.Entrypoint, Models: models, Level: v1.WarmLevel(level + 1)})
	return set, nil
}

// prewarmV1 makes an installation present as a warm run named by the installation: its code
// installed (no entrypoint), its Hub models downloaded, a provider source made and, with a
// destination, uploaded under a publication authorization granted to this machine.
func (m *machineRuns) prewarmV1(ctx context.Context, row records.RentalInstall, report func(machines.InstallProgress)) (json.RawMessage, *exit.Error) {
	selection := row.Selection
	machine, problem := m.machines.DialV1(ctx, row.RentalID, "installing "+either(selection.Package, "models"))
	if problem != nil {
		return nil, problem
	}
	defer machine.Close()
	if row.WorkerBootID != "" && machine.BootID != row.WorkerBootID {
		return nil, exit.Unavailablef("the rental's worker restarted before preparation; the installation is claimed again on its new boot")
	}
	frame, err := machine.Status(ctx)
	if err != nil {
		return nil, machines.Transport(err)
	}
	capabilities := frame.GetCapabilities()
	if !slices.Contains(capabilities, "warm/1") || selection.Destination != "" && !slices.Contains(capabilities, "upload/1") {
		return nil, exit.Named(exit.Structural, "machine.warm_unsupported",
			"this machine takes no warm runs or model uploads; %s", machines.RuntimeUpdate(row.RentalID))
	}
	if selection.Package != "" && selection.Release == "" && !slices.Contains(capabilities, "describe/1") {
		return nil, errNoDescribe
	}
	if selection.HoldsLocally() && !slices.Contains(capabilities, "local-models/1") {
		return nil, exit.Named(exit.Structural, "machine.local_models_unsupported",
			"this machine holds no local files or local/ models; %s", machines.RuntimeUpdate(row.RentalID))
	}
	origin := selection.Hub
	if origin == "" && machine.Account != nil {
		origin = machine.Account.Base()
	}
	spec := &v1.RunSpec{Kind: v1.RunKind_RUN_KIND_WARM, WeightsDestination: selection.Destination}
	if strings.HasPrefix(selection.Package, "local/") {
		// Only unpublished code names its owner, as a call of it does: the same preparation key.
		if caller, problem := m.resolver.namespaceAt(origin); problem == nil {
			spec.Owner = caller.Account
		}
	}
	// Resolved under the same key as this computer's calls (owner, binding revision), so a
	// call after a warm run reuses its preparation instead of resolving again.
	if spec.BindingRevision, problem = m.store.BindingRevision(m.context.forHub(origin).Cfg.HubURL); problem != nil {
		return nil, problem
	}
	if selection.Warm != "" {
		// A warm set member: the machine's whole set goes back with this one changed.
		if !slices.Contains(capabilities, "warm/2") {
			return nil, exit.Named(exit.Structural, "machine.warm_set_unsupported",
				"this machine keeps no warm set; %s", machines.RuntimeUpdate(row.RentalID))
		}
		if spec.Set, problem = warmSetV1(frame.GetWarm(), selection); problem != nil {
			return nil, problem
		}
		selection.Models = nil
	} else if selection.Package != "" {
		spec.Source = &v1.RunSpec_Release{Release: &v1.Release{Package: selection.Package, Release: selection.Release}}
	}
	for _, model := range selection.Models {
		spec.Models = append(spec.Models, &v1.ModelChoice{Parameter: either(model.Slot, model.Model), Repository: model.Model,
			Release: model.Release, Lane: model.Lane, Manifest: model.Manifest, ManifestLength: uint64(max(model.ManifestLength, 0)),
			Source: model.Source, Profiles: model.Profiles})
	}
	if spec.Hub, problem = m.hubAccessV1(ctx, origin, machine, true); problem != nil {
		return nil, problem
	}
	if selection.Destination != "" && !strings.HasPrefix(selection.Destination, "local/") {
		if spec.Publication, problem = authorizeV1Publication(ctx, machine, origin, []string{selection.Destination}); problem != nil {
			return nil, problem
		}
	}
	if selection.Write != "" {
		report(machines.InstallProgress{Stage: "writing " + filepath.Base(selection.Write)})
		object, err := machinev1.WriteFile(ctx, machine.Machine, selection.Write)
		if err != nil {
			return nil, machines.Transport(err)
		}
		if !strings.HasPrefix(strings.TrimPrefix(selection.Models[0].Source, "object://"), object.Digest+"/") {
			return nil, exit.Named(exit.Conflict, "model_source.local_changed", "%s changed since it was measured", selection.Write)
		}
	}
	if providers := (&v1.ProviderAccess{Huggingface: m.resolver.cfg.HuggingFaceToken.Reveal(), Civitai: m.resolver.cfg.CivitaiToken.Reveal()}); providers.Huggingface != "" || providers.Civitai != "" {
		spec.Providers = providers
	}
	ctx, stop := context.WithCancel(ctx) // the stream ends with this call; its connection stays
	defer stop()
	stream, err := machine.Run(ctx, row.ID, 0, spec)
	if err != nil {
		return nil, machines.Transport(err)
	}
	for {
		event, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return nil, exit.Unavailablef("the machine ended the installation's stream before its outcome")
		}
		if err != nil {
			return nil, machines.Transport(err)
		}
		if progress := event.GetProgress(); progress != nil {
			report(machines.InstallProgress{Stage: progress.Stage, TotalBytes: progress.BytesTotal, TransferredBytes: progress.BytesDone})
		}
		if outcome := event.GetOutcome(); outcome != nil {
			if outcome.Status != "succeeded" {
				reason := outcome.GetReason()
				return nil, exit.Named(exit.Failed, either(reason.GetCode(), "rental_install.failed"), "%s", either(reason.GetMessage(), outcome.Status))
			}
			return json.RawMessage(outcome.Result), nil
		}
	}
}

// Describe installs pkg's newest release on a machine at the machine's own Hub and answers
// it with its interface: an install-only warm run with no release named (describe/1).
func (m *machineRuns) Describe(ctx context.Context, machine, pkg, hub string) (api.ReleaseDescription, *exit.Error) {
	row := records.RentalInstall{ID: records.NewID("describe"), RentalID: machine,
		Selection: records.RentalInstallSelection{Package: pkg, Hub: hub}}
	result, problem := m.prewarmV1(ctx, row, func(machines.InstallProgress) {})
	if problem != nil && problem.ErrName() == "machine.warm_unsupported" {
		problem = errNoDescribe
	}
	var described api.ReleaseDescription
	if problem == nil && (json.Unmarshal(result, &described) != nil || described.Release == "" || len(described.Interface) == 0) {
		problem = errNoDescribe
	}
	return described, problem
}

// writeTreesV1 writes each `--input-tree ref=dir` to the machine as writeTreeV1 does; the
// spec names its manifest as an input of the tree media type under the payload's ref.
func (m *machineRuns) writeTreesV1(ctx context.Context, request records.Request, machine *machines.V1, spec *v1.RunSpec) *exit.Error {
	if request.Trees == "" {
		return nil
	}
	work, problem := scratch.Temp(m.layout.Tmp, "input-trees-")
	if problem != nil {
		return problem
	}
	defer work.Release()
	began := time.Now()
	for index, pair := range strings.Split(request.Trees, ",") {
		ref, directory, _ := strings.Cut(pair, "=")
		tree, problem := inputasset.CaptureTree(filepath.Join(work.Path, strconv.Itoa(index)), ref, directory, inputasset.MaxRootInputBytes)
		if problem != nil {
			return problem
		}
		if problem := writeTreeV1(ctx, machine, tree.Snapshot); problem != nil {
			return problem
		}
		spec.Inputs = append(spec.Inputs, &v1.InputFile{Field: ref, Digest: tree.Digest, Length: uint64(tree.Length), MediaType: resultfiles.TreeMediaType})
	}
	m.submissionStage(request.ID, "trees", request.Trees, began)
	return nil
}

// writeTreeV1 writes a captured tree to the machine: each member file, then the manifest.
func writeTreeV1(ctx context.Context, machine *machines.V1, snapshot *records.ByteInputSnapshot) *exit.Error {
	members, problem := resultfiles.ParseTreeManifest(snapshot.Body, snapshot.ContentBytes)
	if problem != nil {
		return problem
	}
	for _, member := range members {
		object, err := machinev1.WriteFile(ctx, machine.Machine, filepath.Join(snapshot.Path+".files", strings.TrimPrefix(member.Digest, "sha256:")))
		if err != nil {
			return machines.Transport(err)
		}
		if object.Digest != member.Digest {
			return exit.Named(exit.Conflict, "input_changed", "tree member %s changed since the run was submitted", member.Path)
		}
	}
	if _, err := machinev1.WriteBytes(ctx, machine.Machine, snapshot.Body); err != nil {
		return machines.Transport(err)
	}
	return nil
}

// hubAccessV1 is the signed-in account's execution access at origin, bound to the machine's
// leaf: the machine reads its Hub with it during the run's preparation and forgets it. A run
// whose owner is not signed in carries none; the machine then refuses only what needs a Hub.
func (m *machineRuns) hubAccessV1(ctx context.Context, origin string, machine *machines.V1, required bool) (*v1.HubAccess, *exit.Error) {
	account := client(m.context.forHub(origin))
	// The pod already has access to its rental Hub. Another selected source needs the
	// owner's grant for that source; the rental's lifecycle credential never crosses Hubs.
	if machine.Rented && machine.Account != nil && machine.Account.Base() == account.Base() {
		return nil, nil
	}
	if account.CredentialIdentity() == "" {
		if machine.Rented && required {
			return nil, exit.Named(exit.Credential, "hub.execution_access_required",
				"reading %s on this rental requires execution access", account.Base()).
				WithRemedy("sign in with `cozy auth login <email> --tensorhub=%s`", account.Base())
		}
		return nil, nil
	}
	// One grant serves this machine's runs for the first half of its life: a warm run asks the
	// Hub nothing.
	key := origin + "\x00" + account.CredentialIdentity() + "\x00" + string(machine.Leaf)
	if held, ok := m.hubAccess.Load(key); ok && time.Now().Before(held.(heldAccess).renew) {
		return held.(heldAccess).grant, nil
	}
	access, problem := account.AuthorizeExecutionAccess(ctx, machine.Leaf)
	if problem != nil {
		return nil, problem
	}
	reads := access.Environment["TENSORHUB_ORIGIN"]
	if local := loopbackHub(origin); machine.Local && local != "" && loopbackHub(reads) == "" {
		reads = local // this computer's machine reads a Hub on this computer there
	}
	grant := &v1.HubAccess{Origin: reads, Token: access.Token, ExpiresAt: access.ExpiresAt.Unix(), CaDer: access.TrustRoot}
	for _, host := range strings.Split(access.Environment["TENSORHUB_OBJECT_STORAGE_HOSTS"], ",") {
		if host = strings.TrimSpace(host); host != "" {
			grant.ObjectHosts = append(grant.ObjectHosts, host)
		}
	}
	if life := time.Until(access.ExpiresAt); life > 0 {
		m.hubAccess.Store(key, heldAccess{grant: grant, renew: time.Now().Add(life / 2)})
	}
	return grant, nil
}

// heldAccess is a machine's execution access and when to ask for a fresh one.
type heldAccess struct {
	grant *v1.HubAccess
	renew time.Time
}

// loopbackHub is origin as scheme://host[:port] when it names this computer, else "".
func loopbackHub(origin string) string {
	scheme, rest, ok := strings.Cut(origin, "://")
	if !ok || scheme != "http" && scheme != "https" {
		return ""
	}
	host := strings.TrimSuffix(rest, "/")
	name := host
	if i := strings.LastIndex(host, ":"); i >= 0 && !strings.HasSuffix(host, "]") {
		name = host[:i]
	}
	switch strings.Trim(name, "[]") {
	case "localhost", "127.0.0.1", "::1":
		return scheme + "://" + host
	}
	return ""
}

// productV1 is one output revision as this client records it, with the file it lands in;
// nil for a revision this client cannot hold.
func (m *machineRuns) productV1(ctx context.Context, machine *machines.V1, request records.Request, sequence uint64, product *v1.Product) *records.Product {
	if request.Number == 0 {
		if numbered, problem := m.store.RequestByReference(request.ID); problem == nil && numbered != nil {
			request.Number = numbered.Number
		}
	}
	prior, problem := m.store.Products(request.ID)
	if problem != nil {
		return nil
	}
	run := strconv.FormatInt(request.Number, 10)
	item := runoutputs.Item{Output: printable(product.Output, 128, '_'), Index: product.Index, List: product.Index > 0}
	item.ID = run + "/" + item.Output
	if item.List {
		item.ID = fmt.Sprintf("%s/%d", item.ID, item.Index)
	}
	seen := map[string]int{}
	for _, earlier := range prior {
		if _, known := seen[earlier.Item]; !known {
			seen[earlier.Item] = len(seen)
		}
	}
	index, known := seen[item.ID]
	if !known {
		index = len(seen)
	}
	media := printable(product.MediaType, 128, '_')
	held := records.Product{Sequence: sequence, Item: item.ID, OutputIndex: index, Type: runoutputs.TypeOf(media),
		Output: item.Output, Op: records.ProductSet, Rev: uint32(product.Rev), Digest: product.Digest,
		Length: int64(product.Length), DurationUs: product.DurationUs, MediaType: media, Label: printable(product.Label, 256, '_')}
	if item.List {
		held.Op, held.Index = records.ProductAppend, product.Index-1
	}
	if media == resultfiles.TreeMediaType {
		held.ContentBytes = treeContentBytes(ctx, machine, request.ID, held)
	}
	if export, problem := m.store.OutputExportOf(request.ID); problem == nil && export != nil && export.Directory != "" {
		held.Path = filepath.Join(export.Directory, itemFile(run, item, media))
	}
	return &held
}

// treeContentBytes is a tree revision's member bytes, from its manifest (0 when unreadable).
func treeContentBytes(ctx context.Context, machine *machines.V1, run string, product records.Product) int64 {
	var manifest bytes.Buffer
	if _, _, err := machine.ReadOutput(ctx, run, product.Output, outputIndexV1(product), 0, uint64(product.Rev), &manifest); err != nil {
		return 0
	}
	var document struct {
		Entries []struct {
			Blob struct{ Length int64 } `json:"blob"`
		} `json:"entries"`
	}
	if json.Unmarshal(manifest.Bytes(), &document) != nil {
		return 0
	}
	var total int64
	for _, entry := range document.Entries {
		total += entry.Blob.Length
	}
	return total
}

// fetcherV1 keeps a run's outputs folder at each item's newest revision while the run goes on.
// A newer revision of an item stops the read of an older one unless it extends it; the run's
// end waits for the newest revisions and shows how far each is (collectV1 then settles them).
type fetcherV1 struct {
	m       *machineRuns
	request records.Request
	machine *machines.V1
	ctx     context.Context
	end     context.CancelFunc
	done    chan struct{}
	wake    chan struct{}

	mu        sync.Mutex
	order     []string                   // items in the order they first arrived
	newest    map[string]records.Product // each item's newest revision
	from      map[string]int64           // where that revision extends the one before it; 0: it replaces it; -1: unknown
	landed    map[string]string          // the digest each item's file holds
	failed    map[string]string          // a revision that could not be read; the end retries it
	reading   string
	stopRead  context.CancelFunc
	finishing bool
}

func (m *machineRuns) fetcherV1(parent context.Context, request records.Request, machine *machines.V1) *fetcherV1 {
	ctx, end := context.WithCancel(parent)
	f := &fetcherV1{m: m, request: request, machine: machine, ctx: ctx, end: end, done: make(chan struct{}),
		wake: make(chan struct{}, 1), newest: map[string]records.Product{}, from: map[string]int64{},
		landed: map[string]string{}, failed: map[string]string{}}
	// What this client recorded before (an earlier attach) is due too: its log is not sent again.
	if products, problem := m.store.Products(request.ID); problem == nil {
		for _, product := range records.Fold(products) {
			f.want(product, nil)
			f.from[product.Item] = -1
		}
	}
	go f.run()
	return f
}

func (f *fetcherV1) signal() {
	select {
	case f.wake <- struct{}{}:
	default:
	}
}

// want asks for an item's newest revision; appendedFrom says it extends the previous one.
func (f *fetcherV1) want(product records.Product, appendedFrom *uint64) {
	if product.Path == "" {
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if _, known := f.newest[product.Item]; !known {
		f.order = append(f.order, product.Item)
	}
	f.newest[product.Item], f.from[product.Item] = product, 0
	if appendedFrom != nil {
		f.from[product.Item] = int64(*appendedFrom)
	}
	if f.reading == product.Item && appendedFrom == nil && f.stopRead != nil {
		f.stopRead() // its bytes are replaced: what is being read is stale
	}
	f.signal()
}

// next is the first item whose file lacks its newest revision.
func (f *fetcherV1) next() (records.Product, bool) {
	for _, item := range f.order {
		product := f.newest[item]
		if f.landed[item] != product.Digest && f.failed[item] != product.Digest {
			return product, true
		}
	}
	return records.Product{}, false
}

func (f *fetcherV1) run() {
	defer close(f.done)
	for {
		f.mu.Lock()
		product, due := f.next()
		finishing := f.finishing
		if !due {
			f.mu.Unlock()
			if finishing {
				return
			}
			select {
			case <-f.ctx.Done():
				return
			case <-f.wake:
			}
			continue
		}
		read, stop := context.WithCancel(f.ctx)
		f.reading, f.stopRead = product.Item, stop
		from := f.from[product.Item]
		f.mu.Unlock()
		problem := writeOutputV1(read, f.machine, f.request.ID, filepath.Dir(product.Path), product, from, f.progress(product))
		superseded := read.Err() != nil // a newer revision stopped this read
		stop()
		f.mu.Lock()
		f.reading, f.stopRead = "", nil
		switch {
		case problem == nil:
			f.landed[product.Item] = product.Digest
		case f.ctx.Err() != nil:
			f.mu.Unlock()
			return
		case superseded || f.newest[product.Item].Digest != product.Digest:
			// The next turn reads the newer revision.
		default:
			// The run's end reads it again (`collectV1`) and records why when it cannot.
			f.failed[product.Item] = product.Digest
		}
		f.mu.Unlock()
	}
}

// progress reports a read the run's end waits on, at most once a second.
func (f *fetcherV1) progress(product records.Product) func(int64) {
	began, last := time.Now(), time.Time{}
	return func(held int64) {
		f.mu.Lock()
		finishing := f.finishing
		f.mu.Unlock()
		if !finishing || time.Since(last) < time.Second && held < product.Length {
			return
		}
		last = time.Now()
		sample := map[string]any{"stage": "saving " + product.Output, "position": held, "total": product.Length, "unit": "bytes"}
		if elapsed := time.Since(began).Seconds(); elapsed > 0 {
			sample["rate"] = float64(held) / elapsed
		}
		_ = f.m.store.AppendEvent(f.request.ID, "machine.progress", 0, map[string]any{"type": "progress", "payload": sample})
	}
}

// finish waits until every item's file holds its newest revision, or could not.
func (f *fetcherV1) finish() {
	f.mu.Lock()
	f.finishing = true
	f.signal()
	f.mu.Unlock()
	<-f.done
}

func (f *fetcherV1) stop() {
	f.end()
	<-f.done
}

// writeOutputV1 makes the file at product.Path this revision's bytes, replacing it whole once
// they match the digest, never editing it in place. `from` > 0 says the revision extends the
// file's bytes from there, so only the tail is read; -1 says nothing is known, and a shorter
// file is tried as a prefix first. `held` is told how many bytes are in hand as they arrive.
// The bytes read stay beside the file until they are whole, so a read the connection cut
// continues from where it stopped, at the same revision.
func writeOutputV1(ctx context.Context, machine *machines.V1, run, directory string, product records.Product, from int64, held func(int64)) *exit.Error {
	if product.MediaType == resultfiles.TreeMediaType {
		return readTreeV1(ctx, machine, run, directory, product)
	}
	if problem := resultfiles.Preflight(directory); problem != nil {
		return problem
	}
	if digestOf(product.Path) == product.Digest {
		return nil
	}
	if held == nil {
		held = func(int64) {}
	}
	base := filepath.Base(product.Path)
	partial := filepath.Join(directory, "."+base+"."+strings.TrimPrefix(product.Digest, "sha256:")[:16]+".cozy-part")
	unwritable := func(err error) *exit.Error {
		return exit.Named(exit.Unavailable, "output_unwritable", "cannot write into %s: %s", directory, err)
	}
	read := func(prefix int64) *exit.Error {
		file, problem := lockPartial(partial)
		if problem != nil {
			return unwritable(problem)
		}
		if digestOf(product.Path) == product.Digest { // another read of this run placed it
			_ = os.Remove(partial)
			file.Close()
			return nil
		}
		offset, err := file.Seek(0, io.SeekEnd)
		if err == nil && offset > product.Length {
			offset, err = 0, file.Truncate(0)
		}
		if err == nil && offset == 0 && prefix > 0 {
			if prior, openErr := os.Open(product.Path); openErr == nil {
				if offset, err = io.CopyN(file, prior, prefix); err != nil {
					offset, err = 0, file.Truncate(0)
				}
				prior.Close()
			}
		}
		if err != nil {
			file.Close()
			return unwritable(err)
		}
		_, _, err = machine.ReadOutput(ctx, run, product.Output, outputIndexV1(product), uint64(offset), uint64(product.Rev),
			&counted{w: file, n: offset, held: held})
		if closeErr := file.Close(); err == nil {
			err = closeErr
		}
		if err != nil {
			// Bytes of a revision the machine no longer serves are no one's prefix.
			if code := status.Code(err); code == codes.FailedPrecondition || code == codes.OutOfRange {
				_ = os.Remove(partial)
			}
			return machines.Transport(err)
		}
		if digestOf(partial) != product.Digest {
			_ = os.Remove(partial)
			return exit.Named(exit.Conflict, "output_revision_changed", "%s moved past revision %d while it was read", product.Output, product.Rev)
		}
		if err := os.Rename(partial, product.Path); err != nil {
			return exit.Named(exit.Unavailable, "output_unwritable", "cannot place %s: %s", product.Path, err)
		}
		// The item's file is whole: what was read of its older revisions is not needed.
		stale, _ := filepath.Glob(filepath.Join(directory, "."+globEscape(base)+".*.cozy-part"))
		for _, path := range stale {
			_ = os.Remove(path)
		}
		return nil
	}
	info, err := os.Stat(product.Path)
	prefix := int64(0)
	if err == nil && info.Mode().IsRegular() && info.Size() < product.Length && (from < 0 || info.Size() == from) {
		prefix = info.Size()
	}
	if prefix > 0 {
		// Only the tail; bytes that do not match the digest are read whole once.
		if problem := read(prefix); problem == nil || ctx.Err() != nil {
			return problem
		}
	}
	return read(0)
}

// lockPartial opens the partial file at path, waiting while another read of the same
// revision holds it; one that read moved or removed meanwhile is opened again.
func lockPartial(path string) (*os.File, error) {
	for {
		file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
		if err != nil {
			return nil, err
		}
		if err := syscall.Flock(int(file.Fd()), syscall.LOCK_EX); err != nil {
			file.Close()
			return nil, err
		}
		held, err := file.Stat()
		at, atErr := os.Stat(path)
		if err == nil && atErr == nil && os.SameFile(held, at) {
			return file, nil
		}
		file.Close()
		if err != nil {
			return nil, err
		}
		if atErr != nil && !errors.Is(atErr, os.ErrNotExist) {
			return nil, atErr
		}
	}
}

// globEscape quotes a file name for filepath.Glob.
func globEscape(name string) string {
	var quoted strings.Builder
	for _, r := range name {
		if strings.ContainsRune(`*?[\`, r) {
			quoted.WriteByte('\\')
		}
		quoted.WriteRune(r)
	}
	return quoted.String()
}

// counted reports how many bytes a read holds as they arrive.
type counted struct {
	w    io.Writer
	n    int64
	held func(int64)
}

func (c *counted) Write(p []byte) (int, error) {
	n, err := c.w.Write(p)
	c.n += int64(n)
	c.held(c.n)
	return n, err
}

// readTreeV1 makes the directory at product.Path this revision's tree: its manifest, then each
// member file it names, read and verified beside the folder, then placed whole.
func readTreeV1(ctx context.Context, machine *machines.V1, run, directory string, product records.Product) *exit.Error {
	if problem := resultfiles.Preflight(directory); problem != nil {
		return problem
	}
	staging, err := os.MkdirTemp(directory, ".cozy-tree-receiving-")
	if err != nil {
		return exit.Named(exit.Unavailable, "output_unwritable", "cannot write into %s: %s", directory, err)
	}
	defer os.RemoveAll(staging)
	read := func(member, target, digest string) *exit.Error {
		file, err := os.Create(target)
		if err != nil {
			return exit.Named(exit.Unavailable, "output_unwritable", "cannot stage a tree in %s: %s", directory, err)
		}
		hash := sha256.New()
		_, _, err = machine.ReadMember(ctx, run, product.Output, outputIndexV1(product), member, uint64(product.Rev), io.MultiWriter(file, hash))
		if closeErr := file.Close(); err == nil {
			err = closeErr
		}
		if err != nil {
			return machines.Transport(err)
		}
		if "sha256:"+hex.EncodeToString(hash.Sum(nil)) != digest {
			return exit.Named(exit.Conflict, "output_revision_changed", "%s moved past revision %d while it was read", product.Output, product.Rev)
		}
		return nil
	}
	manifest := filepath.Join(staging, "manifest")
	if problem := read("", manifest, product.Digest); problem != nil {
		return problem
	}
	raw, err := os.ReadFile(manifest)
	if err != nil {
		return exit.Internalf("cannot read a tree manifest: %s", err)
	}
	var document struct {
		Entries []struct {
			Blob struct{ Length int64 } `json:"blob"`
		} `json:"entries"`
	}
	_ = json.Unmarshal(raw, &document)
	var contentBytes int64
	for _, entry := range document.Entries {
		contentBytes += entry.Blob.Length
	}
	members, problem := resultfiles.ParseTreeManifest(raw, contentBytes)
	if problem != nil {
		return problem
	}
	if err := os.Mkdir(manifest+".files", 0o700); err != nil {
		return exit.Internalf("cannot stage a tree: %s", err)
	}
	for _, member := range members {
		held := filepath.Join(manifest+".files", strings.TrimPrefix(member.Digest, "sha256:"))
		if _, err := os.Stat(held); err == nil {
			continue // a duplicate file is read once
		}
		if problem := read(member.Path, held, member.Digest); problem != nil {
			return problem
		}
	}
	_, problem = resultfiles.MaterializeTree(manifest, directory, filepath.Base(product.Path), product.Digest, product.Length, contentBytes)
	return problem
}

// outputIndexV1 is the item's index as Read names it: 1-based in a list, 0 for one output.
func outputIndexV1(product records.Product) uint32 {
	if product.Op == records.ProductAppend {
		return product.Index + 1
	}
	return 0
}

func digestOf(path string) string {
	file, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return ""
	}
	return "sha256:" + hex.EncodeToString(hash.Sum(nil))
}

// collectV1 settles the run with its outcome: every output file at its final revision (each
// judged; the export records those delivered and why the others were not), the triage bundle
// beside a failure, then the outcome itself.
func (m *machineRuns) collectV1(ctx context.Context, request records.Request, machine *machines.V1, outcome *v1.Outcome, finished, observed int64) *exit.Error {
	// A destination that refuses its files, or bytes the machine no longer serves: the
	// result stays with the machine, and `cozy run watch` collects it once the cause is gone.
	var failure *exit.Error
	end := records.RunEndV1{Outcome: outcome, FinishedMS: finished, ObservedMS: observed}
	if export, problem := m.store.OutputExportOf(request.ID); problem != nil {
		return problem
	} else if export != nil && export.State != "published" {
		products, problem := m.store.Products(request.ID)
		if problem != nil {
			return problem
		}
		var paths, failed []string
		for _, product := range records.Fold(products) {
			if product.Path == "" {
				continue
			}
			if problem := writeOutputV1(ctx, machine, request.ID, export.Directory, product, -1, nil); problem != nil {
				if problem.ErrName() == "machine_execution.transport_unavailable" {
					return problem // the connection ended: the run is attached again and collected then
				}
				failed = append(failed, product.Output)
				if failure == nil {
					failure = problem
				}
				continue
			}
			paths = append(paths, product.Path)
		}
		if failure != nil {
			named := *failure
			named.Message = strings.Join(failed, ", ") + ": " + failure.Message
			failure = &named
		}
		end.Export, end.Paths = true, paths
	}
	// The triage bundle beside a failure, and what the executor measured (`run show`'s stages,
	// steps and attention) as its measurements, kept as one evidence document.
	var bundle []byte
	if outcome.Triage {
		var read strings.Builder
		if _, _, err := machine.ReadTriage(ctx, request.ID, &read); err != nil {
			fmt.Fprintf(m.context.Out, "machine execution %s: triage bundle not kept: %s\n", request.ID, err)
		} else {
			bundle = []byte(read.String())
		}
	}
	if measured := outcome.GetMeasurements(); json.Valid(measured) {
		evidence := map[string]json.RawMessage{}
		if len(bundle) > 0 && json.Unmarshal(bundle, &evidence) != nil {
			evidence = nil // a bundle that is no JSON object is kept as it is
		}
		if _, held := evidence["measurements"]; evidence != nil && !held {
			evidence["measurements"] = measured
			bundle, _ = json.Marshal(evidence)
		}
	}
	end.Refused, end.Evidence = failure, bundle
	return m.store.RecordRunOutcomeV1(request.ID, end)
}

// controlV1 sends a run's cancel, pause or resume to its machine.
func (m *machineRuns) controlV1(ctx context.Context, request records.Request, action string) *exit.Error {
	value, known := map[string]v1.Action{"cancel": v1.Action_ACTION_CANCEL, "pause": v1.Action_ACTION_PAUSE, "resume": v1.Action_ACTION_RESUME}[action]
	if !known {
		return exit.New(exit.Validation, "unknown machine execution control")
	}
	link, problem := m.store.MachineExecution(request.ID)
	if problem != nil || link == nil {
		return problem
	}
	machine, problem := m.machines.DialV1(machines.AttachOnly(ctx), link.MachineID, m.runHolder(request, action+"ing"))
	if problem != nil {
		return problem
	}
	defer machine.Close()
	state, err := machine.Control(ctx, request.ID, value)
	if err != nil {
		return machines.Transport(err)
	}
	// A running job stops before it rests paused: until then it is pausing.
	if action == "pause" && state.State != "paused" {
		state.State = "pausing"
	}
	if problem := m.store.ObserveRunV1(request.ID, &v1.RunEvent{Event: &v1.RunEvent_State{State: state}}, nil); problem != nil {
		return problem
	}
	return m.Start(request)
}
