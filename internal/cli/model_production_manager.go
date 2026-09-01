package cli

import (
	"context"
	"io"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/modelproduction"
	"github.com/cozy-creator/cozy/internal/records"
)

// modelProductionManager is the daemon-owned continuation authority. The CLI
// submits intent and observes rows; only this owner resolves, starts, resumes,
// or explicitly cancels accepted work.
type modelProductionManager struct {
	cfg   config.Config
	store *records.Store
	log   io.Writer
	auth  *accountauth.Manager

	mu      sync.Mutex
	running map[string]context.CancelFunc
}

func newModelProductionManager(cfg config.Config, store *records.Store,
	log io.Writer, auth *accountauth.Manager,
) *modelProductionManager {
	if log == nil {
		log = io.Discard
	}
	return &modelProductionManager{cfg: cfg, store: store, log: log, auth: auth,
		running: map[string]context.CancelFunc{}}
}

func (m *modelProductionManager) SubmitModelProduction(_ context.Context,
	instruction modelproduction.Instruction,
) (records.ModelProductionOperation, bool, *exit.Error) {
	if instruction.Destination == "" || instruction.Release == "" || instruction.Source == "" ||
		instruction.Producer == "" || !instruction.Rental {
		return records.ModelProductionOperation{}, false, exit.New(exit.Validation,
			"durable model production requires destination, release, pinned source, producer, and rental approval")
	}
	if _, problem := hub.ParseRef(instruction.Destination); problem != nil {
		return records.ModelProductionOperation{}, false, problem
	}
	if !modelReleasePattern.MatchString(instruction.Release) {
		return records.ModelProductionOperation{}, false, exit.Usagef(
			"model production release %q is not an immutable N.M.P semantic version",
			instruction.Release)
	}
	if _, problem := canonicalProductionSource(&Context{Inv: productionInvocation(instruction)},
		instruction.Source); problem != nil {
		return records.ModelProductionOperation{}, false, problem
	}
	if _, problem := parseProductionCallable(instruction.Producer); problem != nil {
		return records.ModelProductionOperation{}, false, problem
	}
	if strings.HasPrefix(instruction.Source, "local/") {
		return records.ModelProductionOperation{}, false, exit.Usagef(
			"a %s alias cannot be read by a rented worker", instruction.Source).
			WithRemedy("use its addressable Tensorhub release or original pinned foreign source")
	}
	if problem := m.owns(instruction.Destination); problem != nil {
		return records.ModelProductionOperation{}, false, problem
	}
	instructionBytes, err := instruction.Bytes()
	if err != nil {
		return records.ModelProductionOperation{}, false, exit.Internalf(
			"cannot encode model production instruction: %s", err)
	}
	instructionDigest, err := instruction.Digest()
	if err != nil {
		return records.ModelProductionOperation{}, false, exit.Internalf(
			"cannot digest model production instruction: %s", err)
	}
	operation, replay, problem := m.store.BeginModelProductionInstruction(
		instruction.ID(), instructionDigest, instructionBytes)
	if problem != nil {
		return operation, false, problem
	}
	changed := !replay
	if operation.State == "resolving" {
		if operation.CancelRequested {
			_ = m.store.CancelModelProduction(operation.ID, "resolving", "canceled before plan acceptance")
			settled, readProblem := m.store.ModelProduction(operation.ID)
			if readProblem != nil || settled == nil {
				return operation, changed, readProblem
			}
			return *settled, changed, nil
		}
		plan, _, resolveProblem := m.resolve(instruction)
		if resolveProblem != nil {
			return operation, changed, resolveProblem
		}
		planBytes, encodeErr := plan.Bytes()
		if encodeErr != nil {
			return operation, changed, exit.Internalf("cannot encode model production restart plan: %s", encodeErr)
		}
		planDigest, digestErr := plan.Digest()
		if digestErr != nil {
			return operation, changed, exit.Internalf("cannot digest model production restart plan: %s", digestErr)
		}
		operation, _, problem = m.store.AttachModelProductionPlan(operation.ID,
			instructionDigest, instructionBytes, planBytes, planDigest)
		if problem != nil {
			return operation, changed, problem
		}
	}
	m.kick(operation)
	return operation, changed, nil
}

func (m *modelProductionManager) CancelModelProduction(id string) *exit.Error {
	operation, problem := m.store.ModelProduction(id)
	if problem != nil {
		return problem
	}
	if operation == nil {
		return exit.New(exit.NotFound, "model production %s is absent", id)
	}
	if operation.State == "resolving" {
		return m.store.CancelModelProduction(id, "resolving", "canceled before plan acceptance")
	}
	if problem := m.store.RequestModelProductionCancel(id); problem != nil {
		return problem
	}
	m.mu.Lock()
	cancel := m.running[id]
	m.mu.Unlock()
	if cancel != nil {
		cancel()
	} else if operation.State != "completed" && operation.State != "failed" &&
		operation.State != "canceled" {
		m.kick(*operation)
	}
	return nil
}

// Start is called only after the loopback server is listening. It resumes every
// unfinished row, including a process loss between instruction record and plan attach.
func (m *modelProductionManager) Start() {
	go func() {
		rows, problem := m.store.ActiveModelProductions()
		if problem != nil {
			return
		}
		for _, operation := range rows {
			if operation.State == "resolving" {
				instruction, err := modelproduction.ParseInstruction(operation.Plan)
				if err != nil {
					_ = m.store.FailModelProduction(operation.ID, operation.State,
						"model_production.instruction_invalid", err.Error())
					continue
				}
				_, _, _ = m.SubmitModelProduction(context.Background(), instruction)
				continue
			}
			m.kick(operation)
		}
	}()
}

func (m *modelProductionManager) kick(operation records.ModelProductionOperation) {
	if operation.State == "resolving" || operation.State == "completed" ||
		operation.State == "failed" || operation.State == "canceled" {
		return
	}
	m.mu.Lock()
	if _, exists := m.running[operation.ID]; exists {
		m.mu.Unlock()
		return
	}
	runCtx, cancel := context.WithCancel(context.Background())
	m.running[operation.ID] = cancel
	m.mu.Unlock()
	go func() {
		defer func() {
			m.mu.Lock()
			delete(m.running, operation.ID)
			m.mu.Unlock()
		}()
		m.advance(runCtx, operation)
	}()
}

func (m *modelProductionManager) owns(destination string) *exit.Error {
	ref, problem := hub.ParseRef(destination)
	if problem != nil {
		return problem
	}
	ctx := &Context{Inv: &Invocation{}, Out: io.Discard, Err: m.log,
		Cfg: m.cfg, AccountAuth: m.auth}
	_, problem = ownedPublication(ctx, ref)
	return problem
}

func (m *modelProductionManager) advance(runCtx context.Context,
	operation records.ModelProductionOperation,
) {
	plan, err := modelproduction.Parse(operation.Plan)
	if err != nil {
		_ = m.store.FailModelProduction(operation.ID, operation.State,
			"model_production.plan_invalid", err.Error())
		return
	}
	resolveCtx := &Context{Inv: productionInvocation(plan.Instruction), Out: io.Discard,
		Err: m.log, Cfg: m.cfg, AccountAuth: m.auth}
	var source publishSource
	if operation.State != "outputs_preparing" && operation.State != "release_cut" &&
		operation.State != "cleanup_pending" {
		var problem *exit.Error
		source, problem = resolvePublishSource(resolveCtx, plan.Instruction.Source)
		if problem != nil {
			failProduction(m.store, operation.ID, problem)
			return
		}
		if source.Canonical != plan.Source || source.Selection != plan.SourceSelection ||
			source.License != plan.SourceLicense || source.Lane != plan.InputLane ||
			!reflect.DeepEqual(source.Exact, plan.SourceFiles) {
			failProduction(m.store, operation.ID, exit.Named(exit.Conflict,
				"model_production.source_changed",
				"refreshed source capabilities do not match the accepted source inventory"))
			return
		}
	}
	ctx := resolveCtx
	for {
		problem := runRentedModelProduction(ctx, runCtx, plan, source)
		if problem == nil || problem.Name != "model_production.cut_verdict_unknown" {
			return
		}
		select {
		case <-runCtx.Done():
			// Cut itself is not canceled. Replay once more to learn whether it won;
			// the post-cut cancellation fence will then choose cleanup or no visibility.
			runCtx = context.Background()
		case <-time.After(2 * time.Second):
		}
	}
}

func (m *modelProductionManager) resolve(instruction modelproduction.Instruction) (
	modelproduction.Plan, publishSource, *exit.Error,
) {
	ctx := &Context{Inv: productionInvocation(instruction), Out: io.Discard,
		Err: m.log, Cfg: m.cfg, AccountAuth: m.auth}
	source, problem := resolvePublishSource(ctx, instruction.Source)
	if problem != nil {
		return modelproduction.Plan{}, publishSource{}, problem
	}
	producer, problem := resolveProducerPlan(ctx, instruction.Producer)
	if problem != nil {
		return modelproduction.Plan{}, publishSource{}, problem
	}
	if producer == nil {
		return modelproduction.Plan{}, publishSource{}, exit.New(exit.Validation,
			"model production instruction names no reviewed producer")
	}
	plan := modelproduction.Plan{Instruction: instruction,
		Destination: instruction.Destination, Release: instruction.Release,
		Source: source.Canonical, SourceSelection: source.Selection,
		SourceLicense: source.License, InputLane: source.Lane, SourceFiles: source.Exact,
		Producer: producer.Name, ProducerInstallID: producer.InstallID,
		ProducerRelease: producer.Release, ProducerDigest: producer.ReleaseDigest,
		DescriptorDigest: producer.Descriptor.Digest, Production: producer.Production,
		Jobs: producer.Jobs, Resources: producer.Needs}
	return plan, source, nil
}

func productionInvocation(instruction modelproduction.Instruction) *Invocation {
	return &Invocation{Args: []string{instruction.Destination, instruction.Source},
		Bools: bools("--rental", instruction.Rental), Values: values(
			"--release", instruction.Release, "--producer", instruction.Producer,
			"--lane", instruction.InputLane)}
}
