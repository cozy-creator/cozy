package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
	"github.com/cozy-creator/cozy/internal/runoutputs"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// A run on a machine that serves cozy.machine.v1 is one Run call: its spec (code, payload,
// file inputs written first with Write, model choices, the run's Hub access) submits it once
// under the request's id, and the same call streams its log to the outcome. Resubmitting is
// attaching: the id is the idempotency, so nothing is frozen, closed or acknowledged. Outputs
// are read with Read into the run's outputs folder as their revisions land.

// errNotV1 is a machine that serves no cozy.machine.v1: its run takes the worker.v1 path.
var errNotV1 = exit.Named(exit.Unavailable, "machine.v1_absent", "the machine serves no cozy.machine.v1")

// loopV1 follows one run on a v1 machine until it is settled. It answers true when the run
// belongs to the worker.v1 path instead (a machine without v1, or a run sent there earlier).
func (m *machineRuns) loopV1(request records.Request) bool {
	lastError, delay := "", time.Second
	for m.ctx.Err() == nil {
		current, problem := m.store.RequestRow(request.ID)
		if problem != nil || current == nil {
			return false
		}
		link, problem := m.store.MachineExecution(request.ID)
		if problem != nil || link == nil || link.Abandoned {
			return false
		}
		accepted, problem := m.store.RunV1(request.ID)
		if problem != nil {
			return false
		}
		if !accepted && (len(link.Receipt) > 0 || len(link.Submission) > 0) {
			return true
		}
		if accepted && link.Collected || !accepted && records.Settled(current.State) {
			return false
		}
		done, problem := m.stepV1(*current, link, accepted)
		if problem == errNotV1 {
			return true
		}
		if done {
			return false
		}
		if problem != nil && problem.Message != lastError && m.ctx.Err() == nil {
			fmt.Fprintf(m.context.Out, "machine execution %s: %s\n", request.ID, problem.Message)
			lastError = problem.Message
			if again, _ := m.store.RunV1(request.ID); !again && permanentRefusal(problem) {
				_, _ = m.store.FailQueuedRequest(request.ID, records.QueuedFailure(problem))
				return false
			}
			if !accepted {
				parked := map[string]any{"reason": problem.Message, "wait": orchestrator.WaitRental}
				if current.Machine != "" {
					parked["waiting_on"] = current.Machine
				}
				_ = m.store.AppendEvent(request.ID, "request.parked", 0, parked)
			}
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
			return false
		case <-time.After(delay):
		}
	}
	return false
}

// permanentRefusal is a refusal resubmitting the same spec cannot change.
func permanentRefusal(problem *exit.Error) bool {
	return problem.Code != exit.Unavailable && problem.Code != exit.Deadline && problem.Code != exit.Canceled
}

// stepV1 places the run if it has no machine, connects, submits or attaches, and records the
// log until the stream ends; done once the outcome is recorded.
func (m *machineRuns) stepV1(request records.Request, link *records.MachineExecution, accepted bool) (bool, *exit.Error) {
	ctx, stop := context.WithCancel(m.ctx)
	m.mu.Lock()
	m.submitting[request.ID] = stop
	m.mu.Unlock()
	defer func() {
		m.mu.Lock()
		delete(m.submitting, request.ID)
		m.mu.Unlock()
		stop()
	}()
	request, problem := m.place(ctx, request, link)
	if problem != nil {
		return false, problem
	}
	began := time.Now()
	machine, problem := m.machines.DialV1(ctx, link.MachineID, m.runHolder(request, "running"))
	if problem != nil {
		return false, problem
	}
	defer machine.Close()
	var spec *v1.RunSpec
	if !accepted {
		if _, err := machine.Status(ctx); status.Code(err) == codes.Unimplemented {
			return false, errNotV1
		} else if err != nil {
			return false, machines.Transport(err)
		}
		m.submissionStage(request.ID, "connect", link.MachineID, began)
		if spec, problem = m.specV1(ctx, request, machine); problem != nil {
			return false, problem
		}
	}
	if accepted && link.CancelRequested {
		if _, err := machine.Control(ctx, request.ID, v1.Action_ACTION_CANCEL); err != nil {
			return false, machines.Transport(err)
		}
	}
	began = time.Now()
	stream, err := machine.Run(ctx, request.ID, uint64(max(link.RemoteCursor, 0)), spec)
	if err != nil {
		return false, machines.Transport(err)
	}
	for {
		event, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return false, nil
		}
		if err != nil {
			return false, machines.Transport(err)
		}
		if state := event.GetState(); state != nil && !accepted {
			if problem := m.store.AcceptRunV1(request.ID, state); problem != nil {
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
			return true, m.collectV1(ctx, request, machine, outcome)
		}
		var held *records.Product
		if product := event.GetProduct(); product != nil {
			held = m.productV1(ctx, request, machine, event.Sequence, product)
		}
		if problem := m.store.ObserveRunV1(request.ID, event, held); problem != nil {
			return false, problem
		}
	}
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
		AttentionKernel: request.AttentionKernel, Owner: m.runAccount(request)}
	if request.Trees != "" {
		return nil, exit.Named(exit.Structural, "input_tree_unsupported", "this machine takes file inputs; input trees are not carried to it yet")
	}
	if request.IsJob() {
		spec.Kind = v1.RunKind_RUN_KIND_JOB
		if request.ModelTransfer != nil {
			spec.WeightsDestination = request.ModelTransfer.Destination
		}
	}
	began := time.Now()
	if request.LocalInstallationID != "" {
		revision, problem := m.capturedRevision(request)
		if problem != nil {
			return nil, problem
		}
		manifest, err := machinev1.LocalSource(ctx, machine.Machine, revision)
		if err != nil {
			return nil, machines.Transport(err)
		}
		spec.Source = &v1.RunSpec_Local{Local: &v1.LocalSource{Manifest: manifest}}
		m.submissionStage(request.ID, "package", revision.Package, began)
	} else {
		spec.Source = &v1.RunSpec_Release{Release: &v1.Release{Package: request.Package, Release: request.Release}}
	}
	began = time.Now()
	for _, asset := range request.Assets {
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
		spec.Inputs = append(spec.Inputs, &v1.InputFile{Field: asset.FieldPath, Digest: asset.Digest,
			Length: uint64(asset.Length), MediaType: asset.MediaType, Order: asset.Order})
	}
	if len(request.Assets) > 0 {
		m.submissionStage(request.ID, "inputs", fmt.Sprintf("%d input(s)", len(request.Assets)), began)
	}
	choices, problem := orchestrator.ModelChoices(request, request.Models)
	if problem != nil {
		return nil, problem
	}
	for _, choice := range choices {
		model := &v1.ModelChoice{Parameter: choice.Parameter, Repository: choice.Repository, Release: choice.Release,
			Lane: choice.Lane, Source: choice.Source, Profiles: choice.Profiles}
		if choice.Manifest != nil {
			model.Manifest, _ = canonical.Spell(choice.Manifest.Digest)
			model.ManifestLength = choice.Manifest.Length
		}
		for _, adapter := range choice.Adapters {
			model.Adapters = append(model.Adapters, &v1.Adapter{Component: adapter.Component, Model: adapter.Model,
				Release: adapter.Release, Lane: adapter.Lane, Manifest: adapter.Manifest, Scale: adapter.Scale})
		}
		spec.Models = append(spec.Models, model)
	}
	if request.Hub != "" {
		if access, problem := m.hubAccessV1(ctx, request.Hub, machine); problem != nil {
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

// hubAccessV1 is the signed-in account's execution access at origin, bound to the machine's
// leaf: the machine reads its Hub with it during the run's preparation and forgets it. A run
// whose owner is not signed in carries none; the machine then refuses only what needs a Hub.
func (m *machineRuns) hubAccessV1(ctx context.Context, origin string, machine *machines.V1) (*v1.HubAccess, *exit.Error) {
	account := client(m.context.forHub(origin))
	if account.CredentialIdentity() == "" {
		return nil, nil
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
	return grant, nil
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

// productV1 brings one output revision's file up to date in the run's outputs folder and
// answers it as this client records it; nil for a revision this client cannot hold.
func (m *machineRuns) productV1(ctx context.Context, request records.Request, machine *machines.V1, sequence uint64, product *v1.Product) *records.Product {
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
	if export, problem := m.store.OutputExportOf(request.ID); problem == nil && export != nil && export.Directory != "" {
		held.Path = filepath.Join(export.Directory, itemFile(run, item, media))
		if problem := writeOutputV1(ctx, machine, request.ID, export.Directory, held); problem != nil {
			// The folder is where people look while the run goes on; the run's end writes every
			// file at its final revision and records any refusal then.
			fmt.Fprintf(m.context.Out, "machine execution %s: %s: %s\n", request.ID, held.Path, problem.Message)
		}
	}
	return &held
}

// writeOutputV1 makes the file at product.Path this revision's bytes: a file holding a prefix
// of it gets the missing tail, anything else is written whole beside it and renamed over.
func writeOutputV1(ctx context.Context, machine *machines.V1, run, directory string, product records.Product) *exit.Error {
	if product.MediaType == resultfiles.TreeMediaType {
		return exit.Named(exit.Structural, "output_tree_unsupported", "tree outputs are not read over cozy.machine.v1 yet")
	}
	if problem := resultfiles.Preflight(directory); problem != nil {
		return problem
	}
	if digestOf(product.Path) == product.Digest {
		return nil
	}
	if info, err := os.Stat(product.Path); err == nil && info.Mode().IsRegular() && info.Size() < product.Length {
		file, err := os.OpenFile(product.Path, os.O_WRONLY|os.O_APPEND, 0)
		if err == nil {
			_, _, err = machine.ReadOutput(ctx, run, product.Output, outputIndexV1(product), uint64(info.Size()), file)
			if closeErr := file.Close(); err == nil {
				err = closeErr
			}
			if err == nil && digestOf(product.Path) == product.Digest {
				return nil
			}
		}
	}
	temporary, err := os.CreateTemp(directory, ".cozy-output-*")
	if err != nil {
		return exit.Named(exit.Unavailable, "output_unwritable", "cannot write into %s: %s", directory, err)
	}
	defer os.Remove(temporary.Name())
	_, _, err = machine.ReadOutput(ctx, run, product.Output, outputIndexV1(product), 0, temporary)
	if closeErr := temporary.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return machines.Transport(err)
	}
	if digestOf(temporary.Name()) != product.Digest {
		return exit.Named(exit.Conflict, "output_revision_changed", "%s moved past revision %d while it was read", product.Output, product.Rev)
	}
	if err := os.Rename(temporary.Name(), product.Path); err != nil {
		return exit.Named(exit.Unavailable, "output_unwritable", "cannot place %s: %s", product.Path, err)
	}
	return nil
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
func (m *machineRuns) collectV1(ctx context.Context, request records.Request, machine *machines.V1, outcome *v1.Outcome) *exit.Error {
	if export, problem := m.store.OutputExportOf(request.ID); problem != nil {
		return problem
	} else if export != nil && export.State != "published" {
		products, problem := m.store.Products(request.ID)
		if problem != nil {
			return problem
		}
		var paths, failed []string
		var failure *exit.Error
		for _, product := range records.Fold(products) {
			if product.Path == "" {
				continue
			}
			if problem := writeOutputV1(ctx, machine, request.ID, export.Directory, product); problem != nil {
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
		if problem := m.store.SettleOutputExport(request.ID, paths, failure); problem != nil {
			return problem
		}
	}
	if outcome.Triage {
		var bundle strings.Builder
		if _, _, err := machine.ReadTriage(ctx, request.ID, &bundle); err != nil {
			fmt.Fprintf(m.context.Out, "machine execution %s: triage bundle not kept: %s\n", request.ID, err)
		} else if problem := m.store.RecordMachineTriage(request.ID, 1, []byte(bundle.String())); problem != nil {
			return problem
		}
	}
	return m.store.RecordRunOutcomeV1(request.ID, outcome)
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
	machine, problem := m.machines.DialV1(ctx, link.MachineID, m.runHolder(request, action+"ing"))
	if problem != nil {
		return problem
	}
	defer machine.Close()
	if _, err := machine.Control(ctx, request.ID, value); err != nil {
		return machines.Transport(err)
	}
	return m.Start(request)
}
