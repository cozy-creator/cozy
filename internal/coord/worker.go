package coord

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

// Binding is cozy-creator's LOCAL PINNED-BINDING RECORD: the resolution of one
// `entrypoint_binding_plan_id` against this machine. The coordinator names a plan by
// digest on the wire and stages the record the worker resolves that digest against.
//
// The record's key set is CLOSED at both ends — cozy-runtime refuses an unknown key —
// and th-004 owns the real EntrypointBindingPlan document. This is the named seam, not a
// pretend hub: when th-004 lands, the id keeps being the digest of a canonical document
// and only the document changes.
type Binding struct {
	Entrypoint string         `json:"entrypoint"`
	Record     map[string]any `json:"record"`
	planID     string
	// Outputs names this entrypoint's RESULT FIELD PATHS (`image`, `preview`,
	// `detail.thumb`). It is not part of the binding record's canonical bytes — it is
	// the LAUNCHER's knowledge of the entrypoint's declared result shape, which cl-006
	// needs so a client is not forced to restate it on every submission. cr-003's
	// descriptor is where it comes from once an installed generation exists.
	Outputs []string `json:"outputs,omitempty"`
}

// PlanID is the binding's identity: sha256 over the canonical bytes of the record
// WITHOUT its own id — a document never contains its own digest.
func (b *Binding) PlanID() (string, *exit.Error) {
	if b.planID != "" {
		return b.planID, nil
	}
	doc := map[string]canonical.Value{}
	for k, v := range b.Record {
		if k == "entrypoint_binding_plan_id" {
			continue
		}
		switch t := v.(type) {
		case string:
			doc[k] = t
		case int:
			doc[k] = int64(t)
		case int64:
			doc[k] = t
		case bool:
			doc[k] = t
		case float64:
			// A record that crossed JSON (the dev-spec door) arrives with float64 where
			// the author wrote an integer. These documents are INTEGER-ONLY, exactly as
			// the canonical writer is, so an integral float is the integer it spells and
			// a fractional one is a refusal rather than a rounding.
			if t != float64(int64(t)) {
				return "", exit.Internalf(
					"binding record field %q is %v; these documents are integer-only", k, t)
			}
			doc[k] = int64(t)
		case []string:
			// The record carries REAL SETS now — the ordered component list and the
			// per-component snapshot map — instead of comma-packed strings that both
			// sides had to agree how to split.
			items := make([]canonical.Value, 0, len(t))
			for _, item := range t {
				items = append(items, item)
			}
			doc[k] = items
		case map[string]string:
			pairs := map[string]canonical.Value{}
			for name, value := range t {
				pairs[name] = value
			}
			doc[k] = pairs
		default:
			return "", exit.Internalf("binding record field %q has no canonical spelling (%T)", k, v)
		}
	}
	doc["format"] = "cozy.local.EntrypointBindingRecord/1"
	data, err := canonical.Write(doc)
	if err != nil {
		return "", exit.Internalf("cannot canonicalize the binding record: %s", err)
	}
	spelled, err := canonical.Spell(canonical.Digest(data))
	if err != nil {
		return "", exit.Internalf("cannot spell the binding plan id: %s", err)
	}
	b.planID = spelled
	return spelled, nil
}

// EndpointSpec is everything the coordinator needs to make one endpoint worker resident.
// The caller (cl-010's `start`, or cl-001's live driver) resolves it from the install
// generation; the coordinator itself resolves nothing about Python.
type EndpointSpec struct {
	Endpoint   string     `json:"endpoint"` // org/name
	ReleaseID  string     `json:"release_id"`
	Generation string     `json:"generation"` // the install generation ("" = an uninstalled dev tree)
	Python     string     `json:"python"`     // the interpreter inside the endpoint's own venv
	Args       []string   `json:"args"`
	Dir        string     `json:"dir"`
	Imposed    []string   `json:"imposed"` // exact env values the launcher imposes, never inherited
	Devices    []string   `json:"devices"` // the device envelope this process may SEE
	Bindings   []*Binding `json:"bindings"`
	GraceSec   float64    `json:"grace_sec"`
	NoWarm     bool       `json:"no_warm"`
	// Jobs is the JOB-mode declaration (cl-004). A worker is in exactly ONE mode — the
	// Directive's own oneof says which — so a spec carries bindings or jobs, never both,
	// and `IsJob` is read from the spec rather than re-derived from what happens to be
	// populated later.
	Jobs []*JobPlan `json:"jobs,omitempty"`
	// Remote attaches an ALREADY-RUNNING worker (a rented pod's TLS leg, cl-015/#445)
	// instead of spawning one: the owner dials Addr with the cert at CACert pinned and
	// presents Token as Claim.proof. Nil = the ordinary local spawn.
	Remote *RemoteSpec `json:"remote,omitempty"`
}

// RemoteSpec is the dial triple for a worker this service did not spawn.
type RemoteSpec struct {
	Addr   string `json:"addr"`
	Token  string `json:"token"`
	CACert string `json:"ca_cert"` // path to the worker's pinned PEM
}

// WorkerID is what the Claim names as the expected worker identity; empty skips the
// worker-side check (a spawned worker's identity is already ours by construction).
func (s EndpointSpec) WorkerID() string { return "" }

// IsJob answers the worker's mode.
func (s EndpointSpec) IsJob() bool { return len(s.Jobs) > 0 }

// InstanceID is the endpoint's local worker SLOT identity, and it is deliberately STABLE
// across supervisor restarts: `instance_id` names one provisioned instance lifetime, and
// terminal replay across a restart is authorized by that identity (02 §2). A slot keeps
// its journal root, which is what lets a restarted supervisor report `recovered_attempts`
// at all. A NEW install generation is a genuinely new instance and gets a new id.
// OutputsFor names one entrypoint's declared result field paths.
func (s EndpointSpec) OutputsFor(entrypoint string) []string {
	for _, b := range s.Bindings {
		if b.Entrypoint == entrypoint {
			return b.Outputs
		}
	}
	return nil
}

func (s EndpointSpec) InstanceID() string {
	slot := "slot/" + s.Endpoint + "/" + s.Generation
	if s.IsJob() {
		// A JOB worker is its own slot: one worker per (endpoint, generation, job
		// function), because a JobDirective names ONE job and a worker is in one mode.
		// It is also what lets a serving worker and a job worker of the same endpoint
		// coexist instead of fighting over one instance id.
		slot += "/job/" + s.Jobs[0].Function
	}
	sum := sha256.Sum256([]byte(slot))
	return "ins-" + hex.EncodeToString(sum[:12])
}

type worker struct {
	instanceID string
	spec       EndpointSpec
	cmd        *exec.Cmd
	logPath    string
	home       string
	planIDs    []string

	// what the worker itself reported; the coordinator echoes, never invents
	exited bool
	// exitCode is the process's own disposition. RECYCLE is not a death (cr-009): a
	// run-once job worker exits with it the moment its terminal is acknowledged, and
	// reading that as "the worker died" turns a completed job into a failed request.
	exitCode int
	pid      int // the process THIS coordinator started; the only one that may register
	// bootstrap is the per-spawn credential the worker verifies at Claim (#463's flip:
	// the OWNER presents it as proof; the worker checks constant-time). Minted for every
	// spawn on every platform — the flipped direction has no SO_PEERCRED to lean on.
	bootstrap secret.Value
	// deploymentID is the owner-minted routing key for the ONE deployment this worker
	// hosts (#446). Routing, never identity.
	deploymentID string
	bootID       string
	generation   uint64
	credits      int
	epoch        uint64
	intake       pb.IntakeState
	ready        map[string]bool
	// jobsAvail is the worker's own last `jobs_available`. It is RESERVED at dispatch
	// and corrected by the next Report: without the reservation one drain pass would
	// hand two queued jobs to the same one-attempt worker on one stale reading, and
	// relying on the worker to refuse the second is not a design.
	jobsAvail int
	revision  uint64
	// The worker's own last reason for not serving, and when it first said so. A worker
	// that reports ERROR has not answered "not yet" — it has answered "I cannot", and a
	// client waiting on it needs that answer rather than a longer wait.
	fault      string
	errorSince time.Time
	// lastReport is when this worker last said ANYTHING. The Report cadence is a
	// protocol fact rather than a choice made here, which is what makes silence
	// measurable: a worker that is loading for twenty minutes still reports every
	// period, so "has not reported" is a stall and never merely "is slow".
	lastReport time.Time
	// spawned starts the silence clock BEFORE the first Report: a worker that never
	// claims at all owes its first Report on the same cadence, and a wait with no clock
	// until the first frame is a wait that cannot end (found by the flip: a pre-flip
	// runtime wheel that cannot host left WaitReady spinning forever).
	spawned time.Time
}

// StartWorker journals the device grant, stages the binding records, and spawns the
// supervisor. The grant is journaled BEFORE the process exists: a process that was never
// granted an envelope cannot appear, and two concurrent starts cannot both consume one.
func (c *Coordinator) StartWorker(spec EndpointSpec) (string, *exit.Error) {
	instanceID := spec.InstanceID()
	// The slot's root is REUSED on purpose: the supervisor journal under it is what a
	// restarted worker replays as `recovered_attempts`. Wiping it would manufacture the
	// absence this protocol refuses to manufacture.
	root := c.opt.Layout.WorkerDir(instanceID)
	workerHome := filepath.Join(root, "home")
	if err := os.MkdirAll(filepath.Join(workerHome, "binding-plans"), 0o755); err != nil {
		return "", exit.Internalf("cannot create the worker home %s: %s", workerHome, err)
	}

	var planIDs []string
	for _, b := range spec.Bindings {
		id, e := b.PlanID()
		if e != nil {
			return "", e
		}
		record := map[string]any{}
		for k, v := range b.Record {
			record[k] = v
		}
		record["entrypoint_binding_plan_id"] = id
		data, err := json.MarshalIndent(record, "", "  ")
		if err != nil {
			return "", exit.Internalf("cannot render the binding record: %s", err)
		}
		name := strings.TrimPrefix(id, "sha256:") + ".json"
		if err := os.WriteFile(filepath.Join(workerHome, "binding-plans", name), data, 0o644); err != nil {
			return "", exit.Internalf("cannot stage the binding record: %s", err)
		}
		planIDs = append(planIDs, id)
	}
	sortStrings(planIDs) // the wire field is sorted lexicographic ascending
	if spec.IsJob() {
		if e := stageJobPlans(workerHome, spec.Jobs); e != nil {
			return "", e
		}
		for _, p := range spec.Jobs {
			planIDs = append(planIDs, p.DescriptorID)
		}
	}

	logPath := filepath.Join(root, "worker.log")
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return "", exit.Internalf("cannot open the worker log %s: %s", logPath, err)
	}

	// The GRANT IS JOURNALED FIRST, before any process exists. An admission that
	// refuses here means nothing was started, which is why the refusal has no cleanup.
	if e := c.opt.Store.SpawnWorker(records.WorkerProcess{
		InstanceID: instanceID,
		Endpoint:   spec.Endpoint,
		Generation: spec.Generation,
		ReleaseID:  spec.ReleaseID,
		WorkerID:   "local",
		Devices:    spec.Devices,
	}); e != nil {
		logFile.Close()
		return "", e
	}

	// `cozy-runtime serve`'s CLOSED flag set — the launch facts nothing can discover for a
	// coordinator. It is the runtime's PUBLIC launch grammar, so this is the only spelling
	// the coordinator knows: `--socket`/`--out` are the path grants the verb translates
	// into the supervisor's own `--hub`/`--root`.
	// THE WORKER BINDS ITS OWN SOCKET (#436): a unix path under its run root (loopback
	// `127.0.0.1:0` on Windows — proc shims pick it), published to run/control.addr for
	// this owner to dial. `--socket` is the runtime serve verb's listen grant.
	listen := filepath.Join(root, "run", "control.sock")
	if bootstrapRequired {
		// Windows has no unix socket worth having: the worker binds an ephemeral
		// loopback port and publishes the real address (cr-020 item 4).
		listen = "127.0.0.1:0"
	}
	if err := os.MkdirAll(filepath.Join(root, "run"), 0o755); err != nil {
		return "", exit.Internalf("cannot create the worker run root: %s", err)
	}
	_ = os.Remove(filepath.Join(root, "run", "control.addr"))
	_ = os.Remove(filepath.Join(root, "run", "control.sock"))
	args := append([]string{}, spec.Args...)
	args = append(args,
		"--socket", listen,
		"--out", filepath.Join(root, "run"),
		"--instance-id", instanceID,
		"--release-id", spec.ReleaseID,
		"--devices", strings.Join(spec.Devices, ","),
		"--grace", strconv.FormatFloat(graceOr(spec.GraceSec), 'f', -1, 64),
	)
	if spec.NoWarm {
		args = append(args, "--no-warm")
	}
	cmd := exec.Command(spec.Python, args...)
	cmd.Dir = spec.Dir
	// The child's whole environment: the allowlist plus the values THIS launcher
	// imposes. COZY_HOME points the worker at its own staged records and nothing else.
	imposed := append([]string{
		"COZY_HOME=" + workerHome,
		"CUDA_VISIBLE_DEVICES=" + strings.Join(spec.Devices, ","),
	}, spec.Imposed...)
	// The launcher mints a PER-SPAWN bootstrap credential and hands it over through the
	// environment — the one channel only this child inherits. The flip made it the
	// controller-authority proof on EVERY platform (#463): this owner presents it as
	// Claim.proof and the worker verifies constant-time; the unix socket's filesystem
	// authority is belt, this is suspenders.
	bootstrap := secret.Mint()
	imposed = append(imposed, secret.EnvEntry("COZY_BOOTSTRAP_CREDENTIAL", bootstrap))
	cmd.Env = c.opt.Cfg.Child(imposed...)
	cmd.Stdout, cmd.Stderr = logFile, logFile
	setProcessGroup(cmd)

	w := &worker{
		instanceID: instanceID, spec: spec, cmd: cmd, logPath: logPath,
		home: workerHome, planIDs: planIDs, ready: map[string]bool{},
		bootstrap:    bootstrap,
		deploymentID: "dpl-" + strings.TrimPrefix(instanceID, "ins-"),
		spawned:      time.Now(),
	}
	// Registered BEFORE the process can dial: a worker that registers faster than its
	// launcher can record it would be refused as an instance nobody spawned.
	c.mu.Lock()
	c.workers[instanceID] = w
	c.mu.Unlock()

	if err := cmd.Start(); err != nil {
		c.mu.Lock()
		delete(c.workers, instanceID)
		c.mu.Unlock()
		_ = c.opt.Store.CloseWorker(instanceID)
		logFile.Close()
		return "", exit.Internalf("cannot start the endpoint worker: %s", err)
	}
	// The group is established at fork on Unix and must be ATTACHED after start on Windows,
	// where the child is created suspended and the job adopts it before it runs. Adoption
	// FAILS CLOSED (#449): a worker that cannot be contained is ended before it executes,
	// and the spawn refuses exactly as if the process had never started.
	if err := adoptProcessGroup(cmd); err != nil {
		_ = cmd.Process.Kill()
		go func() { _ = cmd.Wait(); logFile.Close() }()
		c.mu.Lock()
		delete(c.workers, instanceID)
		c.mu.Unlock()
		_ = c.opt.Store.CloseWorker(instanceID)
		return "", exit.Internalf("cannot contain the endpoint worker: %s", err)
	}
	c.mu.Lock()
	w.pid = cmd.Process.Pid
	c.mu.Unlock()
	if e := c.opt.Store.WorkerStarted(instanceID, cmd.Process.Pid, birthOf(cmd.Process.Pid)); e != nil {
		return "", e
	}
	c.logf("worker %s spawned pid=%d devices=[%s] plans=%d",
		instanceID, cmd.Process.Pid, strings.Join(spec.Devices, ","), len(planIDs))
	// THE OWNER DIALS (#436): one goroutine owns this worker's claim conversation for
	// the life of its process, re-claiming across stream drops.
	go c.attach(w)

	// A worker row outliving its process is exactly the sidecar bug this design refuses:
	// the row holds a device grant, so it dies with the process that held it.
	go func() {
		err := cmd.Wait()
		logFile.Close()
		// The process is reaped: retire its containment handle so a reused pid can
		// never meet a stale job (Windows; a no-op where the group is a kernel fact).
		releaseGroup(cmd.Process.Pid)
		c.mu.Lock()
		current, live := c.workers[instanceID]
		mine := live && current == w
		if mine {
			w.exitCode = cmd.ProcessState.ExitCode()
			// The entry STAYS, marked exited: a caller waiting on readiness needs to
			// learn the worker is gone and where its log is, and a row that vanishes
			// silently is the same lie as a row that outlives its process.
			w.exited = true
			if w.bootID != "" {
				delete(c.sessions, w.bootID)
			}
		}
		c.mu.Unlock()
		if mine {
			_ = c.opt.Store.CloseWorker(instanceID)
			c.logf("worker %s exited (%v); its device grant is released — log %s",
				instanceID, err, logPath)
			// The queue's answer to "does capacity exist" just changed, and so has the
			// fate of anything this worker was RUNNING.
			go c.recoverWorker(w.spec)
		}
	}()
	return instanceID, nil
}

func graceOr(v float64) float64 {
	if v <= 0 {
		return 3
	}
	return v
}

// WaitReady blocks until the worker advertises this plan id as dispatchable — a real
// forward completed, never merely "connected".
// ErrorGrace is how long a worker may stay in INTAKE_STATE_ERROR before the wait gives
// up on it. A shortfall on a shared card is often transient — a neighbouring process gives
// the device back and the next boot succeeds — so the state is not instantly fatal. What
// is fatal is staying there: cl-003 watched a worker whose fill was refused
// `device_shortfall` report ERROR every two seconds for eleven minutes while a `cozy run`
// waited out the full thirty-minute readiness window with nothing on its event stream.
const ErrorGrace = 90 * time.Second

// ReportCadence is the worker's OWN Report period, a protocol fact rather than a number
// chosen here: the supervisor reports durably on this cadence so a coordinator can always
// see a directive it issued but the worker did not apply.
const ReportCadence = 2 * time.Second

// SilentReports is how many of those periods may pass with NOTHING arriving before the
// worker is called silent. A COUNT of missed heartbeats, which is why it is not a guess
// about how long a load takes: a worker resident-loading a 20 GB binding for half an hour
// reports on every one of them, and only a worker that has stopped talking misses them.
const SilentReports = 8

// WaitReady waits on OBSERVATIONS and on nothing else. It used to carry a `timeout` —
// 30 minutes from `dispatch`, 10 from `cozy warm`, two different ceilings on the same
// cold start — and that number was a ceiling on how large a model may be, not a bound on
// anything that had gone wrong. Every way this can actually fail is already visible: the
// worker EXITS, it dwells in ERROR past `ErrorGrace`, or it goes SILENT. A worker that is
// loading is none of those, however long it takes.
func (c *Coordinator) WaitReady(instanceID, planID string) *exit.Error {
	silent := SilentReports * ReportCadence
	for {
		c.mu.Lock()
		w := c.workers[instanceID]
		ok := w != nil && !w.exited && w.intake == pb.IntakeState_INTAKE_STATE_READY && w.ready[planID]
		gone := w == nil || w.exited
		logPath, fault, stuck, code := "", "", time.Duration(0), 0
		quiet := time.Duration(0)
		if w != nil {
			logPath, fault, code = w.logPath, w.fault, w.exitCode
			if !w.errorSince.IsZero() {
				stuck = time.Since(w.errorSince)
			}
			if !w.lastReport.IsZero() {
				quiet = time.Since(w.lastReport)
			} else if !w.spawned.IsZero() {
				quiet = time.Since(w.spawned)
			}
		}
		c.mu.Unlock()
		if ok {
			return nil
		}
		if gone {
			if code == RecycleExit {
				// NOT A DEATH. The worker finished its bounded attempt and recycled; the
				// request this wait was started for is either settled already or back on
				// the queue, and its exit has ALREADY asked the queue for a replacement.
				// Calling this a failure settled a requeued job as `failed` after ONE of
				// its three budgeted attempts (observed live).
				return exit.Named(exit.Unavailable, "worker_recycled",
					"the job worker recycled (exit %d) after its bounded attempt", RecycleExit)
			}
			return exit.New(exit.Failed, "the endpoint worker exited before reporting ready").
				WithRemedy("its log is %s", logPath)
		}
		if stuck > ErrorGrace {
			// The worker's OWN words, under the code its reason projects to. Waiting
			// longer on a worker that has been saying "I cannot" for a minute and a half
			// is not patience, it is a client with no answer.
			return workerError(fault, stuck).WithRemedy("its log is %s", logPath)
		}
		if quiet > silent {
			// STALLED, and said as an observation rather than as an elapsed time: this
			// worker owes a Report every `ReportCadence` and has missed `SilentReports`
			// of them. A slow load is not this — a loading worker keeps reporting.
			return exit.Named(exit.Failed, "worker_silent",
				"the endpoint worker has sent no Report for %s, which is %d missed periods "+
					"of %s: it is stalled, not slow", quiet.Round(time.Second), SilentReports,
				ReportCadence).WithRemedy("its log is %s", logPath)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// RecycleExit is the run-once COMPLETION disposition (cr-009, v1 rc 75). A job worker
// exits with it after its terminal is acknowledged: the process ending is the successful
// end of the work, and it is deliberately distinct from any death.
const RecycleExit = 75

// workerError is the coordinator's PROJECTION over a worker's fault reason. The reasons
// are the runtime's neutral vocabulary; deciding what a user should do about one is this
// side's job, exactly as it is for a terminal's (status, cause).
func workerError(fault string, stuck time.Duration) *exit.Error {
	said := fault
	if said == "" {
		said = "no reason reported"
	}
	if strings.Contains(fault, "shortfall") || strings.Contains(fault, "capacity") {
		return exit.New(exit.Capacity,
			"the endpoint worker cannot make its binding resident on this device (%s, for %s)",
			said, stuck.Round(time.Second))
	}
	return exit.New(exit.Failed,
		"the endpoint worker reported ERROR for %s and never became dispatchable: %s",
		stuck.Round(time.Second), said)
}

func (c *Coordinator) WorkerLog(instanceID string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if w := c.workers[instanceID]; w != nil {
		return w.logPath
	}
	return ""
}

// WorkerFacts is what status renders: the protocol's own identities plus the OS
// process-birth identity. There is no local synonym tuple.
type WorkerFacts struct {
	InstanceID string   `json:"instance_id"`
	Endpoint   string   `json:"endpoint"`
	ReleaseID  string   `json:"release_id"`
	BootID     string   `json:"worker_boot_id"`
	PID        int      `json:"pid"`
	Generation uint64   `json:"executor_generation"`
	Epoch      uint64   `json:"readiness_epoch"`
	Revision   uint64   `json:"applied_revision"`
	Intake     string   `json:"intake_state"`
	Exited     bool     `json:"exited"`
	Devices    []string `json:"devices"`
	Ready      []string `json:"ready_plans"`
	// The two OBSERVATIONS a waiter needs and could not see. `cozy warm` polls this
	// listing and used to give up on a 10-minute clock, because the facts that decide
	// — how long the worker has been silent, and how long it has been saying it cannot
	// serve — lived only inside the coordinator. A slow load is neither of them.
	QuietMS    int64  `json:"quiet_ms"`
	ErrorForMS int64  `json:"error_for_ms"`
	Fault      string `json:"fault"`
}

func (c *Coordinator) Worker(instanceID string) *WorkerFacts {
	c.mu.Lock()
	defer c.mu.Unlock()
	w := c.workers[instanceID]
	if w == nil {
		return nil
	}
	f := factsOf(w)
	return &f
}

// Workers is every worker this service currently owns — the LOCAL extension module's
// listing (cl-006) and `cozy status`'s source.
func (c *Coordinator) Workers() []WorkerFacts {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]WorkerFacts, 0, len(c.workers))
	for _, w := range c.workers {
		out = append(out, factsOf(w))
	}
	return out
}

func factsOf(w *worker) WorkerFacts {
	f := WorkerFacts{
		InstanceID: w.instanceID, Endpoint: w.spec.Endpoint, ReleaseID: w.spec.ReleaseID,
		BootID: w.bootID, Generation: w.generation, Epoch: w.epoch,
		Revision: w.revision, Intake: pb.IntakeState_name[int32(w.intake)],
		Exited: w.exited, Devices: w.spec.Devices, Ready: []string{},
		Fault: w.fault,
	}
	if !w.lastReport.IsZero() {
		f.QuietMS = time.Since(w.lastReport).Milliseconds()
	}
	if !w.errorSince.IsZero() {
		f.ErrorForMS = time.Since(w.errorSince).Milliseconds()
	}
	if w.cmd != nil && w.cmd.Process != nil {
		f.PID = w.cmd.Process.Pid
	}
	for id, ok := range w.ready {
		if ok {
			f.Ready = append(f.Ready, id)
		}
	}
	sortStrings(f.Ready)
	return f
}

// StopGrace is how long a worker has between SIGTERM and SIGKILL. It is a CANCELLATION
// BUDGET, not a wait for work to finish: the supervisor's own cooperative-cancel budget
// plus the terminal transaction that follows it, which is what a worker still needs to do
// after it is told to stop. It is deliberately ONE number — this operation used to be
// spelled 20 s in the stall watchdog, 20 s for a stale worker, 20 s for one that failed to
// become ready, and 30 s on the API route, with nothing anywhere saying why the same
// SIGTERM deserved three budgets.
const StopGrace = 30 * time.Second

// StopWorker drains and stops the WHOLE worker process group — the only yield mechanism
// there is. An attempt is never killed to improve queue latency, and no suspend path
// exists anywhere in this package. The wait for the process to go is `alive(pid)`, the
// kernel's own answer, polled until `StopGrace` is spent.
func (c *Coordinator) StopWorker(instanceID string, grace time.Duration) {
	c.mu.Lock()
	w := c.workers[instanceID]
	c.mu.Unlock()
	if w == nil {
		return
	}
	if w.cmd.Process != nil && !w.exited {
		// The cooperative tier: SIGTERM to the group, CTRL_BREAK to the job's console
		// group on Windows. A failure here is loud but not an escalation by itself —
		// the bounded wait below is what separates asking from insisting.
		if err := killGroup(w.cmd.Process.Pid, syscall.SIGTERM); err != nil {
			c.logf("worker %s: the cooperative stop could not be delivered (%s); "+
				"the forced tier follows the grace window", instanceID, err)
		}
		deadline := time.Now().Add(grace)
		for time.Now().Before(deadline) {
			if !alive(w.cmd.Process.Pid) {
				break
			}
			time.Sleep(20 * time.Millisecond)
		}
		if alive(w.cmd.Process.Pid) {
			_ = killGroup(w.cmd.Process.Pid, syscall.SIGKILL)
		}
	}
	_ = c.opt.Store.CloseWorker(instanceID)
	c.mu.Lock()
	w.exited = true
	delete(c.workers, instanceID)
	if w.bootID != "" {
		delete(c.sessions, w.bootID)
	}
	c.mu.Unlock()
	c.logf("worker %s stopped; its device grant is released", instanceID)
	go c.reviveQueue()
}

// Reconcile runs at boot, before anything is served. Rows describing processes from a
// previous life are checked against their OS BIRTH identity: a matching birth is a real
// orphan and is killed (it holds a device grant and a socket this service no longer
// knows); a mismatch is a REUSED PID and is never signalled — only its row is closed.
func (c *Coordinator) Reconcile() (killed, forgotten int, e *exit.Error) {
	rows, e := c.opt.Store.LiveWorkers()
	if e != nil {
		return 0, 0, e
	}
	for _, row := range rows {
		if row.PID > 0 && birthOf(row.PID) == row.Birth && row.Birth != "" {
			_ = killGroup(row.PID, syscall.SIGKILL)
			killed++
			c.logf("orphan worker %s (pid %d, birth %s) killed on reconcile",
				row.InstanceID, row.PID, row.Birth)
		} else {
			forgotten++
			c.logf("worker row %s forgotten: pid %d is gone or reused (birth %q != %q)",
				row.InstanceID, row.PID, birthOf(row.PID), row.Birth)
		}
		if e := c.opt.Store.CloseWorker(row.InstanceID); e != nil {
			return killed, forgotten, e
		}
	}
	// THE DISPATCH QUEUE IS MEMORY, AND THE AUTHORITY IS NOT. A request recorded as owed
	// work before the crash has no attempt and no queue entry after it — it simply stopped
	// existing as far as scheduling was concerned, while its row went on saying `queued`
	// forever. Found live by cl-004's crash arm: a job submitted moments before the
	// coordinator was `kill -9`ed never ran again. Rebuilding the queue from the authority
	// is the only place the two can be made to agree, and it happens before anything is
	// served. Order is the authority's own (created_at), so FIFO survives a crash too.
	owed, e := c.opt.Store.Owed()
	if e != nil {
		return killed, forgotten, e
	}
	for _, req := range owed {
		c.enqueue(req.ID)
		c.logf("%s was owed work before the restart; it is back on the dispatch queue", req.ID)
	}
	if len(owed) > 0 {
		go c.reviveQueue()
	}
	// AND THE ATTEMPTS THAT OWE A TERMINAL. Killing an orphan is only half of a restart:
	// the attempts it held are unsettled, and the ONE thing that can settle them is the
	// supervisor's own journal replayed by a worker in the SAME SLOT (02 §6.2). Nothing
	// else in this process will ask for that slot — the requests are not queued, they have
	// ordinals — so a job dispatched moments before a `kill -9` of the coordinator hung
	// forever. Found live by cl-004's crash arm; cl-006's own crash section had been
	// POSTing /v1/local/workers by hand to work around it.
	unsettled, e := c.opt.Store.Unsettled()
	if e != nil {
		return killed, forgotten, e
	}
	for _, req := range unsettled {
		c.logf("%s holds an attempt with no terminal; making its slot resident so the "+
			"supervisor journal replays", req.ID)
		c.selectOrStart(req)
	}
	return killed, forgotten, nil
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}
