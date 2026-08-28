// cozy-live is cl-001's live verification driver. It is NOT a test suite (tracker
// README #160): every section starts the real system — this repository's real
// local orchestrator and record owner, the REAL cozy-runtime worker and its disposable
// CUDA executor,
// and cr-005's real SDXL CAS — and prints what it observed.
//
// The orchestrator it drives is the same `internal/orchestrator` package `cozy up` runs. What
// this binary adds is the ORCHESTRATION worker-live.py's harness added on the other
// side: staging, timing, kills, and one adversarial worker (`fakeworker`) that speaks
// raw protocol bytes so the refusal arms have someone to refuse.
//
//	nice -n 19 ./cozy-live canonical
//	nice -n 19 ./cozy-live attempt   --runtime ~/cozy_v2/cozy-runtime --bench ~/cozy_v2/tensorfs-bench
//	nice -n 19 ./cozy-live recovered --runtime ... --bench ...
//	nice -n 19 ./cozy-live arms      --runtime ... --bench ...
package main

import (
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator/internal/records"
)

var (
	pass, fail int
	flags      = map[string]string{}
)

func head(title string) { fmt.Printf("\n=== %s\n", title) }

func check(label string, ok bool, detail string) bool {
	if ok {
		pass++
		fmt.Printf("  ok   %s", label)
	} else {
		fail++
		fmt.Printf("  FAIL %s", label)
	}
	if detail != "" {
		fmt.Printf(" — %s", detail)
	}
	fmt.Println()
	return ok
}

func flag(name, fallback string) string {
	if v, ok := flags[name]; ok && v != "" {
		return v
	}
	return fallback
}

func must(what string, err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "cozy-live: %s: %v\n", what, err)
		exitAfterCleanup(1)
	}
}

// Every child this driver spawns (`cozy up`, pod workers, cozy-media, in-process
// orchestrators) registers its teardown here, because `must` exits the process and a
// deferred stop never runs on that path. Cleanups run once, newest first, on: normal
// return, a failing check count, `must`, a panic, or SIGINT/SIGTERM.
var (
	cleanupMu sync.Mutex
	cleanups  []func()
)

func onExit(fn func()) {
	cleanupMu.Lock()
	defer cleanupMu.Unlock()
	cleanups = append(cleanups, fn)
}

func runCleanups() {
	cleanupMu.Lock()
	list := cleanups
	cleanups = nil
	cleanupMu.Unlock()
	for i := len(list) - 1; i >= 0; i-- {
		list[i]()
	}
}

func exitAfterCleanup(code int) {
	runCleanups()
	os.Exit(code)
}

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: cozy-live "+
			"<canonical|attempt|recovered|arms|stall|stallactive|tlspin|api|apiarms|apicrash|verbs|journey|rent|planids|"+
			"descriptor|pipeline|m4arms|dropack|jobarms|jobs|jobcrash|artifacts|artifactcrash|workflows|workflowcrash|mediagc|fixtures|fakeworker> [--flag value]")
		os.Exit(2)
	}
	section := args[0]
	for i := 1; i < len(args); i++ {
		if !strings.HasPrefix(args[i], "--") {
			continue
		}
		name := strings.TrimPrefix(args[i], "--")
		if i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
			flags[name] = args[i+1]
			i++
		} else {
			flags[name] = "true"
		}
	}

	if section == "fakeworker" {
		os.Exit(fakeWorker())
	}
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-signals
		fmt.Fprintf(os.Stderr, "cozy-live: %v: tearing down\n", sig)
		exitAfterCleanup(130)
	}()
	defer func() {
		if r := recover(); r != nil {
			runCleanups()
			panic(r)
		}
		runCleanups()
	}()

	switch section {
	case "canonical":
		sectionCanonical()
	case "descriptor":
		sectionDescriptor()
	case "attempt":
		sectionAttempt()
	case "recovered":
		sectionRecovered()
	case "arms":
		sectionArms()
	case "apiarms":
		sectionAPIArms()
	case "api":
		sectionAPI()
	case "apicrash":
		sectionAPICrash()
	case "verbs":
		sectionVerbs()
	case "journey":
		sectionJourney()
	case "planids":
		sectionPlanIDs()
		return
	case "rent":
		sectionRent()
	case "pipeline":
		sectionPipeline()
	case "m4arms":
		sectionM4Arms()
	case "dropack":
		sectionDropAck()
	case "mediagc":
		sectionMediaGC()
	case "artifacts":
		sectionArtifacts()
	case "artifactcrash":
		sectionArtifactCrash()
	case "jobarms":
		sectionJobArms()
	case "jobs":
		sectionJobs()
	case "jobcrash":
		sectionJobCrash()
	case "fixtures":
		sectionFixtures()
	case "workflows":
		sectionWorkflows()
	case "workflowcrash":
		sectionWorkflowCrash()
	case "stall":
		sectionStall()
	case "stallactive":
		sectionStallActive()
	case "tlspin":
		sectionTLSPin()
	default:
		fmt.Fprintf(os.Stderr, "cozy-live: unknown section %q\n", section)
		os.Exit(2)
	}

	fmt.Printf("\n%d checks, %d failed\n", pass+fail, fail)
	if fail > 0 {
		exitAfterCleanup(1)
	}
}

// --------------------------------------------------------------------------- hosting

type live struct {
	root   string
	cfg    config.Config
	l      home.Layout
	store  *records.Store
	c      *orchestrator.Orchestrator
	log    *os.File
	closed bool
}

// hostCoordinator brings up the REAL LocalCoordinator on a fresh root. Everything a
// `cozy up` process does, minus the loopback transport this section has no use for.
func hostCoordinator(name string, fresh bool) *live {
	root := flag("home", filepath.Join(os.TempDir(), "cozy-live", name))
	if fresh {
		must("clearing the live root", os.RemoveAll(root))
	}
	must("creating the live root", os.MkdirAll(root, 0o755))
	must("setting COZY_HOME for the one env reader", os.Setenv("COZY_HOME", root))

	cfg, e := config.Load()
	if e != nil {
		must("config", e)
	}
	// Load freezes on first call; a driver that reuses the process must re-derive the
	// layout from the root it just made rather than a stale frozen home.
	cfg.Home = root
	l, e := home.Open(cfg.Home)
	if e != nil {
		must("layout", e)
	}
	st, e := records.Open(l.DB)
	if e != nil {
		must("records", e)
	}
	logPath := filepath.Join(root, "orchestrator.log")
	logFile, err := os.Create(logPath)
	must("orchestrator log", err)

	c, e := orchestrator.Open(orchestrator.Options{
		Cfg: cfg, Layout: l, Store: st, Yield: "smart", Log: logFile,
		EnvironmentSpecDigest: "sha256:" + strings.Repeat("11", 32),
		ConfigDigest:          "sha256:" + strings.Repeat("22", 32),
		MaxOutputMiB:          8,
	})
	if e != nil {
		must("orchestrator", e)
	}
	if _, _, e := c.Reconcile(); e != nil {
		must("reconcile", e)
	}
	go func() { _ = c.Serve() }()
	lv := &live{root: root, cfg: cfg, l: l, store: st, c: c, log: logFile}
	onExit(lv.close)
	return lv
}

func (lv *live) close() {
	if lv.closed {
		return
	}
	lv.closed = true
	lv.c.Close(20 * time.Second)
	lv.store.Close()
	lv.log.Close()
}

// rssMiB is this process's resident set — the orchestrator's own footprint, since the
// orchestrator runs in this process.
func rssMiB() float64 {
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "VmRSS:") {
			var kb float64
			fmt.Sscanf(strings.TrimPrefix(line, "VmRSS:"), "%f", &kb)
			return kb / 1024
		}
	}
	return 0
}

func ms(d time.Duration) string { return fmt.Sprintf("%.1f ms", float64(d.Microseconds())/1000) }

// orchestratorSubmission is the driver's alias for the orchestrator's Submission, so the
// section files read as prose rather than as a struct literal repeated five times.
type orchestratorSubmission = orchestrator.Submission

func submission(planID string, body []byte) orchestratorSubmission {
	return submissionKey(planID, body, "idem-"+fmt.Sprint(time.Now().UnixNano()))
}

func submissionKey(planID string, body []byte, idem string) orchestratorSubmission {
	return orchestratorSubmission{
		IdemKey: idem, Endpoint: "cozy/sdxl-unet", Entrypoint: "denoise",
		PlanID: planID, Payload: body,
		// One destination per RESULT FIELD PATH. `denoise` returns `image`.
		Outputs: []string{"image"},
	}
}

// errOf lets a *exit.Error be handed to must() without a typed-nil trap.
func errOf(e *exit.Error) error {
	if e == nil {
		return nil
	}
	return e
}

func briefly(e *exit.Error) string {
	if e == nil {
		return "accepted"
	}
	return e.Message
}
