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
	Entrypoint string
	Record     map[string]any
	planID     string
}

// PlanID is the binding's identity: sha256 over the canonical bytes of the record
// WITHOUT its own id — a document never contains its own digest.
func (b *Binding) PlanID() (string, *exit.Error) {
	if b.planID != "" {
		return b.planID, nil
	}
	doc := map[string]canonical.Value{}
	for k, v := range b.Record {
		if k == "plan_id" {
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
	Endpoint   string // org/name
	ReleaseID  string
	Generation string // the install generation this worker serves ("" = an uninstalled dev tree)
	Python     string // the interpreter inside the endpoint's own venv
	Args       []string
	Dir        string
	Imposed    []string // exact env values the launcher imposes, never inherited
	Devices    []string // the device envelope this process may SEE
	Bindings   []*Binding
	GraceSec   float64
	NoWarm     bool
}

// InstanceID is the endpoint's local worker SLOT identity, and it is deliberately STABLE
// across supervisor restarts: `instance_id` names one provisioned instance lifetime, and
// terminal replay across a restart is authorized by that identity (02 §2). A slot keeps
// its journal root, which is what lets a restarted supervisor report `recovered_attempts`
// at all. A NEW install generation is a genuinely new instance and gets a new id.
func (s EndpointSpec) InstanceID() string {
	sum := sha256.Sum256([]byte("slot/" + s.Endpoint + "/" + s.Generation))
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
	exited      bool
	pid         int // the process THIS coordinator started; the only one that may register
	sessionID   string
	incarnation uint64
	epoch       uint64
	intake      pb.IntakeState
	ready       map[string]bool
	revision    uint64
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
		record["plan_id"] = id
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

	args := append([]string{}, spec.Args...)
	args = append(args,
		"--hub", "unix:"+c.opt.Socket,
		"--root", filepath.Join(root, "run"),
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
	cmd.Env = c.opt.Cfg.Child(append([]string{
		"COZY_HOME=" + workerHome,
		"CUDA_VISIBLE_DEVICES=" + strings.Join(spec.Devices, ","),
	}, spec.Imposed...)...)
	cmd.Stdout, cmd.Stderr = logFile, logFile
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	w := &worker{
		instanceID: instanceID, spec: spec, cmd: cmd, logPath: logPath,
		home: workerHome, planIDs: planIDs, ready: map[string]bool{},
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
	c.mu.Lock()
	w.pid = cmd.Process.Pid
	c.mu.Unlock()
	if e := c.opt.Store.WorkerStarted(instanceID, cmd.Process.Pid, birthOf(cmd.Process.Pid)); e != nil {
		return "", e
	}
	c.logf("worker %s spawned pid=%d devices=[%s] plans=%d",
		instanceID, cmd.Process.Pid, strings.Join(spec.Devices, ","), len(planIDs))

	// A worker row outliving its process is exactly the sidecar bug this design refuses:
	// the row holds a device grant, so it dies with the process that held it.
	go func() {
		err := cmd.Wait()
		logFile.Close()
		c.mu.Lock()
		current, live := c.workers[instanceID]
		mine := live && current == w
		if mine {
			// The entry STAYS, marked exited: a caller waiting on readiness needs to
			// learn the worker is gone and where its log is, and a row that vanishes
			// silently is the same lie as a row that outlives its process.
			w.exited = true
			if w.sessionID != "" {
				delete(c.sessions, w.sessionID)
			}
		}
		c.mu.Unlock()
		if mine {
			_ = c.opt.Store.CloseWorker(instanceID)
			c.logf("worker %s exited (%v); its device grant is released — log %s",
				instanceID, err, logPath)
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
func (c *Coordinator) WaitReady(instanceID, planID string, timeout time.Duration) *exit.Error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		c.mu.Lock()
		w := c.workers[instanceID]
		ok := w != nil && !w.exited && w.intake == pb.IntakeState_INTAKE_STATE_READY && w.ready[planID]
		gone := w == nil || w.exited
		logPath := ""
		if w != nil {
			logPath = w.logPath
		}
		c.mu.Unlock()
		if ok {
			return nil
		}
		if gone {
			return exit.New(exit.Failed, "the endpoint worker exited before reporting ready").
				WithRemedy("its log is %s", logPath)
		}
		time.Sleep(20 * time.Millisecond)
	}
	return exit.New(exit.Deadline, "the endpoint worker did not report %s ready in %s",
		planID, timeout).WithRemedy("its log is %s", c.WorkerLog(instanceID))
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
	InstanceID  string
	SessionID   string
	PID         int
	Incarnation uint64
	Epoch       uint64
	Revision    uint64
	Intake      string
	Devices     []string
	Ready       []string
}

func (c *Coordinator) Worker(instanceID string) *WorkerFacts {
	c.mu.Lock()
	defer c.mu.Unlock()
	w := c.workers[instanceID]
	if w == nil {
		return nil
	}
	f := &WorkerFacts{
		InstanceID: w.instanceID, SessionID: w.sessionID, PID: w.cmd.Process.Pid,
		Incarnation: w.incarnation, Epoch: w.epoch, Revision: w.revision,
		Intake: pb.IntakeState_name[int32(w.intake)], Devices: w.spec.Devices,
	}
	for id, ok := range w.ready {
		if ok {
			f.Ready = append(f.Ready, id)
		}
	}
	sortStrings(f.Ready)
	return f
}

// StopWorker drains and stops the WHOLE worker process group — the only yield mechanism
// there is. An attempt is never killed to improve queue latency, and no suspend path
// exists anywhere in this package.
func (c *Coordinator) StopWorker(instanceID string, grace time.Duration) {
	c.mu.Lock()
	w := c.workers[instanceID]
	c.mu.Unlock()
	if w == nil {
		return
	}
	if w.cmd.Process != nil && !w.exited {
		_ = syscall.Kill(-w.cmd.Process.Pid, syscall.SIGTERM)
		deadline := time.Now().Add(grace)
		for time.Now().Before(deadline) {
			if !alive(w.cmd.Process.Pid) {
				break
			}
			time.Sleep(20 * time.Millisecond)
		}
		if alive(w.cmd.Process.Pid) {
			_ = syscall.Kill(-w.cmd.Process.Pid, syscall.SIGKILL)
		}
	}
	_ = c.opt.Store.CloseWorker(instanceID)
	c.mu.Lock()
	w.exited = true
	delete(c.workers, instanceID)
	if w.sessionID != "" {
		delete(c.sessions, w.sessionID)
	}
	c.mu.Unlock()
	c.logf("worker %s stopped; its device grant is released", instanceID)
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
			_ = syscall.Kill(-row.PID, syscall.SIGKILL)
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
	return killed, forgotten, nil
}

func alive(pid int) bool { return syscall.Kill(pid, 0) == nil }

// birthOf is the OS process-birth identity: /proc/<pid>/stat field 22, the kernel's own
// start time in clock ticks. A reused pid has a different birth, which is why a pid
// alone is never enough to adopt a worker as warm.
func birthOf(pid int) string {
	data, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return ""
	}
	// The comm field may contain spaces; everything after the last ')' is positional.
	tail := string(data)
	if i := strings.LastIndex(tail, ")"); i >= 0 {
		tail = tail[i+1:]
	}
	fields := strings.Fields(tail)
	if len(fields) < 20 {
		return ""
	}
	return fields[19] // (22) starttime, offset by the two fields consumed above
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}
