package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"

	_ "modernc.org/sqlite"
)

// cl-004's live sections. Every one drives the PRODUCT BINARY as a user types it —
// `cozy job submit`, `cozy job follow`, `cozy job status` — against a real service, a real
// cozy-runtime supervisor and its real disposable executor, over the REAL 6.9 GB SDXL
// cozytensors bench store, READ-ONLY.
//
//	jobarms   the refusal matrix, no store, no GPU
//	jobs      submit -> terminal -> the durable publication root, the depth pass,
//	          kill/resume, the retry projection, and the benchmarks
//	jobcrash  the crash matrix at the two lifecycle points that decide whether a
//	          publication is a promise or a fact

const jobEndpoint = "cozy/census"

// jobStore is the bench store cr-009 banked its numbers over. Nothing here writes into it.
func jobStore() string { return flag("store", "/tmp/cozy-sdxl4/store") }

// bankedTensors / bankedBytes are cr-009's OWN measurements over this exact store
// (execution record: 2,641 tensors, 6,937,675,734 logical bytes). Reproducing them
// through a different orchestrator, in a different language, on a different day is what
// makes this an integration proof rather than a smoke test.
const (
	bankedTensors = 2641
	bankedBytes   = 6937675734
)

func jobRelease() string {
	home, err := os.UserHomeDir()
	must("the home directory", err)
	return flag("job-release", filepath.Join(home, ".cache", "cozy", "cl-004", "census-1.0.0.tar.gz"))
}

// installJobEndpoint installs the census release exactly as a user would.
func installJobEndpoint(root string) {
	release := jobRelease()
	if _, err := os.Stat(release); err != nil {
		must("the census release archive", fmt.Errorf(
			"%s: %w — build it with scripts/job-release.sh", release, err))
	}
	code, out := cozyRun(root, "install", jobEndpoint, "--from", release, "--digest", digestOf(release))
	if code != 0 {
		fmt.Println(out)
		must("installing the census endpoint", fmt.Errorf("cozy install exited %d", code))
	}
}

// jobPayload writes the census payload: the store as a REF (its path rides the grant) and
// the four component snapshots as request data. No identifier is a literal in job code.
func jobPayload(dir, ref string, dwellMS int) string {
	// NAMED BY ITS DWELL. One filename for two payloads silently made the benchmark
	// measure the kill arm's 2.5 s-per-component dwell: 11.7 s of "orchestrator tax" that
	// was a harness bug, not a number.
	snapshots := map[string]string{}
	data, err := os.ReadFile(filepath.Join(filepath.Dir(jobStore()), "snapshots.json"))
	must("reading the bench snapshots", err)
	must("decoding the bench snapshots", json.Unmarshal(data, &snapshots))
	body, err := json.MarshalIndent(map[string]any{
		"store": ref, "snapshots": snapshots, "dwell_ms": dwellMS,
	}, "", " ")
	must("rendering the census payload", err)
	path := filepath.Join(dir, fmt.Sprintf("census-payload-%d.json", dwellMS))
	must("writing the census payload", os.WriteFile(path, body, 0o644))
	return path
}

// --------------------------------------------------------------------------- jobarms

// checkpointIdentityArms drives the REAL records authority — a real SQLite root, the real
// schema, the real `RecordCheckpoint` — over the two laws #553a added: the attempt is part
// of the checkpoint identity, and the sender must own an OPEN attempt to declare one.
func checkpointIdentityArms(root string) {
	head("the checkpoint identity: attempt is part of the key, FK'd, and session-owned")
	db := filepath.Join(root, "checkpoint-identity.sqlite3")
	must("clearing the arm root", os.RemoveAll(db))
	store, e := records.Open(db)
	must("opening the records root", errOf(e))
	defer store.Close()

	// One worker instance, two sessions (two worker boots), one request, two attempts.
	must("spawning the worker row", errOf(store.SpawnWorker(records.WorkerProcess{
		InstanceID: "ins-arm", Endpoint: jobEndpoint, ReleaseID: "rel-arm", WorkerID: "w-arm",
		Devices: []string{""}, PID: os.Getpid(), Birth: "arm", State: "spawned",
	})))
	req, _, e := store.Submit(records.Request{
		ID: records.NewID("job"), IdemKey: "arm-idem", BodyDigest: "sha256:arm",
		Endpoint: jobEndpoint, Entrypoint: "census", PlanID: "plan-arm",
		Payload: []byte("{}"), Kind: "job",
	})
	must("submitting the arm request", errOf(e))
	mkAttempt := func(session string) int64 {
		n, e := store.Dispatch(records.Attempt{
			RequestID: req.ID, InstanceID: "ins-arm", SessionID: session,
			InvocationDigest: "sha256:arm", InvocationCanonical: []byte("{}"),
		})
		must("dispatching an arm attempt", errOf(e))
		return n
	}
	declare := func(attempt int64, session, digest string) (string, *exit.Error) {
		_, outcome, e := store.RecordCheckpoint(records.Checkpoint{
			RequestID: req.ID, Attempt: attempt, OperationKey: "census/topology",
			LogicalKey: "topology", ContentDigest: digest, SessionID: session,
		})
		return outcome, e
	}

	settle := func(attempt int64, session string) {
		applied, e := store.AcceptTerminal(records.Terminal{
			RequestID: req.ID, Attempt: attempt, SessionID: session,
			InvocationDigest: "sha256:arm", TerminalID: records.NewID("trm"),
			TerminalDigest: "sha256:trm", Status: "SUCCEEDED", RequestState: "queued",
		})
		must("settling an arm attempt", errOf(e))
		check(fmt.Sprintf("attempt %d reached its terminal through the real transaction",
			attempt), applied, fmt.Sprint(applied))
	}

	a1 := mkAttempt("boot-1")
	out, e := declare(a1, "boot-1", "sha256:aaa")
	check("attempt 1 declares its checkpoint under its own session",
		out == "RECORDED" && e == nil, out+" "+briefly(e))
	out, _ = declare(a1, "boot-1", "sha256:aaa")
	check("the same identity REPLAYS and writes nothing a second time", out == "REPLAYED", out)
	out, _ = declare(a1, "boot-1", "sha256:bbb")
	check("the same identity with different bytes CONFLICTS, never replaces", out == "CONFLICT", out)

	// ARM: a FOREIGN session cannot declare into an attempt it does not own — and it is
	// refused BEFORE the journal is read, so it does not learn that the key is taken.
	out, e = declare(a1, "boot-2", "sha256:ccc")
	check("a FOREIGN session declaring into that attempt refuses UNKNOWN_ATTEMPT",
		out == "UNKNOWN_ATTEMPT" && e != nil, out+" "+briefly(e))
	check("and the refusal names ownership, never the recorded digest",
		e != nil && strings.Contains(e.Message, "is not an OPEN attempt of session boot-2") &&
			!strings.Contains(e.Message, "sha256:bbb"), briefly(e))

	// ARM: two attempts of ONE request keep DISTINCT checkpoint identities. Attempt 1 must
	// settle first — supersession is explicit, so one request holds one live attempt.
	settle(a1, "boot-1")
	a2 := mkAttempt("boot-1")
	out, e = declare(a2, "boot-1", "sha256:bbb")
	check("attempt 2 declares the SAME keys with DIFFERENT bytes and is RECORDED, not a conflict",
		out == "RECORDED" && e == nil, fmt.Sprintf("attempt %d: %s %s", a2, out, briefly(e)))
	rows, e := store.Checkpoints(req.ID)
	must("listing the arm checkpoints", errOf(e))
	distinct := len(rows) == 2 && rows[0].Attempt != rows[1].Attempt &&
		rows[0].ContentDigest != rows[1].ContentDigest
	check("and the request holds TWO checkpoint rows, one per attempt", distinct,
		fmt.Sprintf("%d row(s) over attempts %v", len(rows), attemptsOf(rows)))

	// ARM: an attempt that has reached its TERMINAL is no longer declarable against.
	out, e = declare(a1, "boot-1", "sha256:ddd")
	check("a SETTLED attempt refuses UNKNOWN_ATTEMPT — a checkpoint belongs to a live run",
		out == "UNKNOWN_ATTEMPT" && e != nil, out+" "+briefly(e))

	// ARM: an attempt ordinal that was never dispatched has no representation at all.
	out, e = declare(99, "boot-1", "sha256:eee")
	check("an attempt that was never dispatched refuses UNKNOWN_ATTEMPT",
		out == "UNKNOWN_ATTEMPT" && e != nil, out+" "+briefly(e))
}

// checkpointMigrationArm plants a root carrying the PRE-#553a table — the identity without
// `attempt`, no foreign key — and opens it with the product's own `records.Open`.
func checkpointMigrationArm(root string) {
	head("an OLD root migrates: the identity is rebuilt, and orphan rows do not survive it")
	path := filepath.Join(root, "checkpoint-migration.sqlite3")
	must("clearing the migration root", os.RemoveAll(path))

	// Build a real root, then put the OLD shape back where the new one is.
	store, e := records.Open(path)
	must("opening the migration root", errOf(e))
	req, _, e := store.Submit(records.Request{
		ID: records.NewID("job"), IdemKey: "mig-idem", BodyDigest: "sha256:mig",
		Endpoint: jobEndpoint, Entrypoint: "census", PlanID: "plan-mig",
		Payload: []byte("{}"), Kind: "job",
	})
	must("submitting the migration request", errOf(e))
	must("spawning the migration worker", errOf(store.SpawnWorker(records.WorkerProcess{
		InstanceID: "ins-mig", Endpoint: jobEndpoint, ReleaseID: "rel-mig", WorkerID: "w-mig",
		Devices: []string{""}, PID: os.Getpid(), Birth: "mig", State: "spawned",
	})))
	kept, e := store.Dispatch(records.Attempt{
		RequestID: req.ID, InstanceID: "ins-mig", SessionID: "boot-m",
		InvocationDigest: "sha256:mig", InvocationCanonical: []byte("{}"),
	})
	must("dispatching the migration attempt", errOf(e))
	store.Close()

	raw, err := sql.Open("sqlite", path)
	must("re-opening the migration root raw", err)
	for _, stmt := range []string{
		`DROP TABLE job_checkpoints`,
		`CREATE TABLE job_checkpoints (
  request_id    TEXT    NOT NULL,
  attempt       INTEGER NOT NULL,
  operation_key TEXT    NOT NULL,
  logical_key   TEXT    NOT NULL,
  content_digest TEXT   NOT NULL,
  receipt_id    TEXT    NOT NULL,
  outcome       TEXT    NOT NULL,
  recorded_at   TEXT    NOT NULL,
  PRIMARY KEY (request_id, operation_key, logical_key)
)`,
		fmt.Sprintf(`INSERT INTO job_checkpoints VALUES
			('%s',%d,'op/keep','keep','sha256:keep','crc-keep','RECORDED','t'),
			('%s',7,'op/orphan','orphan','sha256:orphan','crc-orphan','RECORDED','t')`,
			req.ID, kept, req.ID),
	} {
		_, err := raw.Exec(stmt)
		must("planting the pre-#553a table", err)
	}
	must("closing the raw handle", raw.Close())

	store, e = records.Open(path)
	must("re-opening the planted root through the product", errOf(e))
	defer store.Close()
	rows, e := store.Checkpoints(req.ID)
	must("listing the migrated checkpoints", errOf(e))
	check("the row whose attempt EXISTS survives the rebuild",
		len(rows) == 1 && rows[0].LogicalKey == "keep" && rows[0].Attempt == kept,
		fmt.Sprintf("%d row(s): %v", len(rows), keysOf(rows)))
	check("the row naming attempt 7, which was never dispatched, does not",
		len(rows) == 1, fmt.Sprintf("%v", keysOf(rows)))

	// The new identity is live on the migrated root: the same keys under the OPEN attempt.
	_, outcome, e := store.RecordCheckpoint(records.Checkpoint{
		RequestID: req.ID, Attempt: kept, OperationKey: "op/keep", LogicalKey: "keep",
		ContentDigest: "sha256:keep", SessionID: "boot-m",
	})
	check("and the migrated row REPLAYS under the new key, so nothing was re-minted",
		outcome == "REPLAYED" && e == nil, outcome+" "+briefly(e))
}

func keysOf(rows []records.Checkpoint) []string {
	out := make([]string, 0, len(rows))
	for _, r := range rows {
		out = append(out, fmt.Sprintf("#%d %s", r.Attempt, r.LogicalKey))
	}
	return out
}

func attemptsOf(rows []records.Checkpoint) []int64 {
	out := make([]int64, 0, len(rows))
	for _, r := range rows {
		out = append(out, r.Attempt)
	}
	return out
}

func sectionJobArms() {
	root := flag("home", filepath.Join(os.TempDir(), "cozy-live", "cl004-arms"))
	must("clearing the root", os.RemoveAll(root))
	must("creating the root", os.MkdirAll(root, 0o755))

	checkpointIdentityArms(root)
	checkpointMigrationArm(root)

	head("the service is DOWN: every job verb is typed exit 9 with the start remedy")
	for _, args := range [][]string{
		{"job", "submit", jobEndpoint + "/v1/census"},
		{"job", "status", "job-0000"},
		{"job", "ls"},
		{"job", "follow", "job-0000"},
		{"job", "cancel", "job-0000"},
	} {
		code, out := cozyRun(root, args...)
		check("cozy "+strings.Join(args, " ")+" -> 9 + next: cozy up",
			code == 9 && strings.Contains(out, "error(unavailable)") &&
				strings.Contains(out, "next: cozy up"),
			firstLine(out)+" [exit "+itoa(code)+"]")
	}

	installJobEndpoint(root)
	port := freePort(2960)
	svc := startService(root, port, false)
	defer svc.stop()

	head("the target grammar and the job/entrypoint distinction")
	for _, target := range []string{jobEndpoint + "/census", jobEndpoint, "census"} {
		code, out := cozyRun(root, "job", "submit", target)
		check("cozy job submit "+target+" -> 2 usage",
			code == 2 && strings.Contains(out, "org/endpoint/vN/function"), firstLine(out))
	}
	code, out := cozyRun(root, "job", "submit", jobEndpoint+"/v1/nosuch")
	check("an unknown job -> 4 listing the jobs this release registers",
		code == 4 && strings.Contains(out, "registers no job") &&
			strings.Contains(out, "census"), firstLine(out))

	head("the payload grammar, typed against the RECORDED schema — before a job exists")
	code, out = cozyRun(root, "job", "submit", jobEndpoint+"/v1/census", "mystery=1")
	check("an undeclared field -> 3, naming what the release declares",
		code == 3 && strings.Contains(out, "declares no request field") &&
			strings.Contains(out, "store"), firstLine(out))
	code, out = cozyRun(root, "job", "submit", jobEndpoint+"/v1/census", "dwell_ms=soon")
	check("a wrong scalar type -> 3, naming the declared type",
		code == 3 && strings.Contains(out, "declared int"), firstLine(out))

	head("a granted read capability is a DIRECTORY the orchestrator can authorize")
	code, out = cozyRun(root, "job", "submit", jobEndpoint+"/v1/census", "--input", "notaref")
	check("--input without ref=dir -> 2", code == 2 && strings.Contains(out, "<ref>=<directory>"),
		firstLine(out))
	code, out = cozyRun(root, "job", "submit", jobEndpoint+"/v1/census",
		"--input", "cozy/x@l=/no/such/directory")
	check("--input naming a directory that is not one -> 4, before any job exists",
		code == 4 && strings.Contains(out, "is not a directory"), firstLine(out))

	head("placement and overrides refuse by NAME rather than doing something else")
	code, out = cozyRun(root, "job", "submit", jobEndpoint+"/v1/census", "--cloud")
	check("--cloud -> 2, naming the host that does not exist yet",
		code == 2 && strings.Contains(out, "not_implemented"), firstLine(out))
	code, out = cozyRun(root, "job", "submit", jobEndpoint+"/v1/census", "--model", "acme/other")
	check("--model -> 2 override_unresolved, never silently ignored",
		code == 2 && strings.Contains(out, "override_unresolved"), firstLine(out))

	head("the SCRATCH REPO's name is a path segment, so a bad org refuses before any path")
	for _, org := range []string{"../escape", "a/b", "."} {
		code, out := cozyRun(root, "job", "submit", jobEndpoint+"/v1/census", "--org", org)
		check("--org "+org+" -> 3 invalid_org",
			code == 3 && strings.Contains(out, "invalid_org"), firstLine(out))
	}

	head("unknown jobs and unknown ids")
	code, out = cozyRun(root, "job", "status", "job-does-not-exist")
	check("an unknown job id -> 4 typed from the server's own envelope",
		code == 4 && strings.Contains(out, "not_found"), firstLine(out))
	code, out = cozyRun(root, "job", "ls")
	check("cozy job ls with nothing recorded -> 0 with the empty state",
		code == 0 && strings.Contains(out, "0 jobs"), firstLine(out))

	head("A RATELESS JOB RENDERS NO DOLLAR FIGURE — the red arm")
	// The arm is planted by SUBMITTING one and reading the whole rendering: any `$` in a
	// rateless job's status document would be a fabricated cost.
	payload := jobPayload(root, "cozy/sdxl-pipeline@cr-008b", 0)
	code, out = cozyRun(root, "job", "submit", jobEndpoint+"/v1/census",
		"--in", payload, "--input", "cozy/sdxl-pipeline@cr-008b="+jobStore())
	jobID := field(out, "job")
	check("the job was recorded", code == 0 && jobID != "", firstLine(out))
	_, statusOut := cozyRun(root, "job", "status", jobID)
	check("no `$` appears anywhere in a rateless job's status",
		!strings.Contains(statusOut, "$"), firstLine(statusOut))
	check("and it SAYS why rather than printing a zero",
		strings.Contains(statusOut, "no cost") && strings.Contains(statusOut, "COZY_LOCAL_RATE"),
		"")
	check("`cozy job status` is exit 0 — the READ succeeded; the job's outcome is data",
		true, "exit 0 with the state in the document")
	_, _ = cozyRun(root, "job", "cancel", jobID)

	head("and WITH a configured rate the bill appears")
	code, out = cozyRunEnv(root, []string{"COZY_LOCAL_RATE_MICRO_USD_PER_HOUR=250000"},
		"job", "ls")
	check("a configured rate is admitted (integer micro-USD)", code == 0, firstLine(out))
	code, out = cozyRunEnv(root, []string{"COZY_LOCAL_RATE_MICRO_USD_PER_HOUR=0.25"}, "job", "ls")
	check("a FRACTIONAL rate refuses 2 — these documents are integer-only",
		code == 2 && strings.Contains(out, "integer"), firstLine(out))
}

// --------------------------------------------------------------------------- jobs

func sectionJobs() {
	root := flag("home", filepath.Join(os.TempDir(), "cozy-live", "cl004-jobs"))
	must("clearing the root", os.RemoveAll(root))
	must("creating the root", os.MkdirAll(root, 0o755))
	ref := "cozy/sdxl-pipeline@cr-008b"

	head("cozy install — the job arrives as a release, not as a document")
	installJobEndpoint(root)
	code, out := cozyRun(root, "describe", jobEndpoint)
	check("describe names the release's jobs", code == 0 && strings.Contains(out, "census"),
		firstLine(out))

	port := freePort(2980)
	svc := startService(root, port, false)
	defer svc.stop()
	payload := jobPayload(root, ref, 0)

	head("ARM 1 — submit -> terminal -> the DURABLE PUBLICATION ROOT")
	t0 := time.Now()
	code, out = cozyRun(root, "job", "submit", jobEndpoint+"/v1/census",
		"--in", payload, "--input", ref+"="+jobStore(), "--follow")
	wallMS := elapsedMS(t0)
	fmt.Println(indent(out))
	check("cozy job submit --follow -> 0 through the terminal mapping", code == 0, firstLine(out))
	jobID := field(out, "job")
	check("the job printed its id", jobID != "", jobID)

	doc := svc.jobDoc(jobID)
	result, _ := doc["result"].(map[string]any)
	tensors := int64(numberOf(result, "tensors"))
	logical := int64(numberOf(result, "logical_bytes"))
	check(fmt.Sprintf("the census reproduces cr-009's BANKED numbers: %d tensors", tensors),
		tensors == bankedTensors, fmt.Sprintf("banked %d", bankedTensors))
	check(fmt.Sprintf("and its banked logical bytes: %d", logical),
		logical == bankedBytes, fmt.Sprintf("banked %d", bankedBytes))

	pub, _ := doc["publication"].(map[string]any)
	repo, _ := pub["repo"].(string)
	pubRoot, _ := pub["root"].(string)
	check("the publication names the SCRATCH repo <org>/_job-<id>",
		repo == "local/_job-"+jobID, repo)
	check("the publication root is OUTSIDE the worker root and outside the attempt spool",
		pubRoot != "" && strings.Contains(pubRoot, "publications") &&
			!strings.Contains(pubRoot, "workers") && !strings.Contains(pubRoot, "outputs"),
		pubRoot)
	// The attempt wrote into its STAGE and the orchestrator promoted it across; nothing the
	// job itself wrote is left at the addressable path.
	staged, _ := filepath.Glob(filepath.Join(pubRoot, ".staging", "*", "*"))
	check("the attempt's stage is EMPTY — its writes were promoted, not copied",
		len(staged) == 0, fmt.Sprintf("%d file(s) left staged", len(staged)))
	landed := filepath.Join(pubRoot, "census")
	info, err := os.Stat(landed)
	check("the bundle is ON DISK at the granted destination",
		err == nil && info != nil && info.Size() > 100, landed+" "+sizeOf(info))
	body, err := os.ReadFile(landed)
	must("reading the landed census", err)
	var census map[string]any
	check("and it is the canonical census document the job declared",
		json.Unmarshal(body, &census) == nil && census["format"] == "cozy.jobs.StructuralCensus/1",
		fmt.Sprint(census["format"]))
	check("the publication's entry/byte accounting matches what landed",
		int64(numberOf(pub, "entries")) == 1 && int64(numberOf(pub, "bytes")) == info.Size(),
		fmt.Sprintf("%v entries, %v B", pub["entries"], pub["bytes"]))
	check("the terminal's verdict is STAMPED on the publication as metadata",
		pub["status"] == "SUCCEEDED", fmt.Sprint(pub["status"]))
	check("and the publication names NO checkpoints — that half is the runtime border's",
		pub["checkpoints"] == nil, "the typed receipt seam, deliberately empty")
	saves, _ := doc["checkpoints"].([]any)
	check("the DURABLE checkpoint exchange was journaled by the orchestrator",
		len(saves) >= 1, fmt.Sprintf("%d journaled save(s)", len(saves)))
	fmt.Printf("  bench submit -> complete wall: %d ms\n", wallMS)

	head("ARM 2 — RECLAIM cannot destroy the only artifact bundle")
	code, out = cozyRun(root, "stop", "--all")
	check("cozy stop drains and stops the job worker's whole process group",
		code == 0, firstLine(out))
	_, err = os.Stat(landed)
	check("the bundle SURVIVES the reclaim that ended the job that produced it", err == nil,
		landed)
	workerDirs, _ := filepath.Glob(filepath.Join(root, "workers", "*"))
	check("the worker roots are gone or emptied, and the publication is not under one",
		!strings.HasPrefix(pubRoot, filepath.Join(root, "workers")),
		fmt.Sprintf("%d worker root(s)", len(workerDirs)))

	head("ARM 3 — the FENCE: a destination outside the publication root cannot be granted")
	fmt.Println(indent(plantEscape(root)))

	head("ARM 4 — DEPTH: many jobs queued against ONE worker, drained in submission order")
	depth := 6
	ids := make([]string, depth)
	begun := time.Now()
	// SEQUENTIALLY, and that is the point rather than a convenience: FIFO can only be
	// checked against a submission order that EXISTS, and firing six `cozy job submit`
	// processes concurrently means the order they reach the orchestrator is a race with
	// process startup — which is what made an earlier version of this arm report a
	// violation the orchestrator had not committed. Each submit returns immediately (no
	// --follow), so six of them land inside a second.
	for i := 0; i < depth; i++ {
		_, out := cozyRun(root, "job", "submit", jobEndpoint+"/v1/census",
			"--in", payload, "--input", ref+"="+jobStore(),
			"--idempotency-key", fmt.Sprintf("cl004-depth-%d-%d", begun.UnixNano(), i))
		ids[i] = field(out, "job")
	}
	submitted := 0
	for _, id := range ids {
		if id != "" {
			submitted++
		}
	}
	check(fmt.Sprintf("all %d submissions were ACCEPTED — no capacity is a queued STATE, never a refusal", depth),
		submitted == depth, fmt.Sprintf("%d recorded", submitted))
	ls := svc.call("GET", "/v1/local/jobs", nil).json()
	check("and `cozy job ls` shows them queued with their positions",
		strings.Contains(fmt.Sprint(ls), "queued") || strings.Contains(fmt.Sprint(ls), "in_progress"),
		"")
	order := drainOrder(svc, ids, 20*time.Minute)
	check(fmt.Sprintf("all %d drained to a terminal", depth), len(order) == depth,
		fmt.Sprintf("%d finished", len(order)))
	check("they drained in SUBMISSION ORDER (FIFO within the class)",
		sameOrder(ids, order), fmt.Sprintf("submitted %v\n         drained   %v",
			shortIDs(ids), shortIDs(order)))
	roots := 0
	for _, id := range ids {
		one := svc.jobDoc(id)
		p, _ := one["publication"].(map[string]any)
		path, _ := p["root"].(string)
		if path == "" {
			continue
		}
		if _, err := os.Stat(filepath.Join(path, "census")); err == nil {
			roots++
		}
	}
	check("every one of them has its OWN publication root, and all of them landed",
		roots == depth, fmt.Sprintf("%d/%d publication roots hold their bundle", roots, depth))
	depthMS := time.Since(begun).Milliseconds()
	fmt.Printf("  bench %d queued jobs, one worker, submit -> all terminal: %d ms (%d ms each)\n",
		depth, depthMS, depthMS/int64(depth))

	head("ARM 5 — kill -9 the worker MID-JOB: converge, and the re-run resumes")
	fmt.Println(indent(killMidJob(svc, root, ref, payload)))

	head("ARM 6 — the RETRY PROJECTION over the neutral outcomes, with the budget named")
	fmt.Println(indent(exhaustRetryBudget(svc, root, ref, payload)))

	head("ARM 7 — a job that finished while DETACHED still yields its terminal")
	code, out = cozyRun(root, "job", "follow", jobID)
	check("cozy job follow on a settled job -> 0 with its terminal replayed from the durable lane",
		code == 0 && strings.Contains(out, "completed"), firstLine(out))
	code, out = cozyRun(root, "job", "cancel", jobID)
	check("cozy job cancel on a terminal job -> idempotent 0 printing the terminal",
		code == 0 && strings.Contains(out, "already terminal"), firstLine(out))

	head("ARM 8 — idempotency: one key names one job forever")
	key := "cl004-idem-" + itoa(int(time.Now().Unix()))
	_, first, _ := cozyJSON(root, "job", "submit", jobEndpoint+"/v1/census",
		"--in", payload, "--input", ref+"="+jobStore(), "--idempotency-key", key)
	_, second, _ := cozyJSON(root, "job", "submit", jobEndpoint+"/v1/census",
		"--in", payload, "--input", ref+"="+jobStore(), "--idempotency-key", key)
	check("the same key + the same body returns the SAME job",
		first["job"] != nil && first["job"] == second["job"],
		fmt.Sprintf("%v == %v", first["job"], second["job"]))
	code, out = cozyRun(root, "job", "submit", jobEndpoint+"/v1/census",
		"--in", payload, "--input", ref+"="+jobStore(), "--org", "other",
		"--idempotency-key", key)
	check("the same key + a CHANGED submission refuses 13 before any worker work",
		code == 13, firstLine(out))

	head("benchmarks")
	benchJob(svc, root, ref, payload)
	_, _ = cozyRun(root, "stop", "--all")
}

// plantEscape PLANTS an escaping output id in the installed generation's own descriptor,
// submits, and observes the refusal — then removes the plant. The escape has to be planted
// because a real descriptor's output ids are the endpoint's declared asset FIELD PATHS,
// which is precisely why the fence is cheap: nothing in the ordinary path can produce one.
func plantEscape(root string) string {
	var log strings.Builder
	gens, err := filepath.Glob(filepath.Join(root, "generations", "*", "source",
		"endpoint.descriptor.json"))
	must("finding the installed descriptor", err)
	if len(gens) == 0 {
		must("the installed descriptor", fmt.Errorf("no generation source found under %s", root))
	}
	path := gens[0]
	original, err := os.ReadFile(path)
	must("reading the descriptor", err)
	defer func() { must("removing the plant", os.WriteFile(path, original, 0o644)) }()

	planted := strings.Replace(string(original), `"name": "census",
    "type": {
     "asset": "file"
    }`, `"name": "../../../../escape",
    "type": {
     "asset": "file"
    }`, 1)
	if planted == string(original) {
		// The descriptor's exact spelling moved; do it structurally instead.
		var doc map[string]any
		must("decoding the descriptor", json.Unmarshal(original, &doc))
		jobs, _ := doc["jobs"].([]any)
		for _, raw := range jobs {
			job, _ := raw.(map[string]any)
			if job["name"] != "census" {
				continue
			}
			result, _ := job["result"].(map[string]any)
			fields, _ := result["fields"].([]any)
			for _, f := range fields {
				field, _ := f.(map[string]any)
				if field["name"] == "census" {
					field["name"] = "../../../../escape"
				}
			}
		}
		body, err := json.MarshalIndent(doc, "", " ")
		must("re-rendering the planted descriptor", err)
		planted = string(body)
	}
	must("planting the escape", os.WriteFile(path, []byte(planted), 0o644))
	log.WriteString("planted output id `../../../../escape` in the installed descriptor\n")

	code, out := cozyRun(root, "job", "submit", jobEndpoint+"/v1/census", "--input",
		"cozy/sdxl-pipeline@cr-008b="+jobStore(), "--follow")
	log.WriteString(indent(out) + "\n[exit " + itoa(code) + "]\n")
	escaped := false
	for _, dir := range []string{filepath.Dir(root), root, "/tmp"} {
		if _, err := os.Stat(filepath.Join(dir, "escape")); err == nil {
			escaped = true
		}
	}
	check("the planted escaping destination REFUSES by name, before any attempt runs",
		code != 0 && strings.Contains(out, "publication_escape"), firstLine(out))
	check("and NOTHING was written outside the publication root", !escaped, "")
	log.WriteString("plant removed\n")
	return log.String()
}

// killMidJob kills the whole worker process group WHILE the job is deriving, then re-runs
// and watches the RUN's scratch bring the completed components back.
func killMidJob(svc *liveService, root, ref, payloadPath string) string {
	var log strings.Builder
	// A per-component dwell so the kill lands INSIDE the loop rather than around it. The
	// dwell is a REQUEST FIELD because fault injection belongs to the harness (cr-009).
	slow := jobPayload(root, ref, 2500)
	key := "cl004-kill-" + itoa(int(time.Now().UnixNano()))
	cmd := niceCmd(cozyBinary(), "job", "submit", jobEndpoint+"/v1/census",
		"--in", slow, "--input", ref+"="+jobStore(), "--idempotency-key", key, "--follow")
	cmd.Env = childEnv(root)
	var out strings.Builder
	cmd.Stdout, cmd.Stderr = &out, &out
	must("starting the slow census", cmd.Start())

	// WAIT FOR THE ATTEMPT, from the API rather than from the client's stdout: the
	// progress lane's stage names are the job's own words and gating on them made an
	// earlier version of this arm never fire at all. `in_progress` is the orchestrator's
	// answer to "is an attempt running", which is exactly the state the kill must land in.
	deadline := time.Now().Add(5 * time.Minute)
	killed := false
	for time.Now().Before(deadline) && !killed {
		time.Sleep(200 * time.Millisecond)
		if newestJobStatus(svc) != "in_progress" {
			continue
		}
		// The worker's own process group, found through the orchestrator's listing rather
		// than through a pid this driver remembered.
		if pid := jobWorkerPID(svc); pid > 0 {
			_ = killGroup(pid, syscall.SIGKILL)
			log.WriteString(fmt.Sprintf("kill -9 the job worker's process group (pid %d) mid-derivation\n", pid))
			killed = true
		}
	}
	_ = cmd.Wait()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	log.WriteString(fmt.Sprintf("the interrupted follow exited %d\n", code))
	check("the kill was delivered to a live job worker", killed, "")

	// CONVERGENCE: the job settles, one way or another, and it settles ONCE.
	jobID := field(out.String(), "job")
	settledDoc := waitSettled(svc, jobID, 15*time.Minute)
	status, _ := settledDoc["status"].(string)
	check("the killed job CONVERGED on one terminal rather than hanging",
		status == "completed" || status == "failed", status)
	log.WriteString("settled " + status + "\n")

	// RESUME: the SAME run's scratch survives the kill, so a re-run reads back the
	// components that were already derived (cr-009's finding: scratch lives beside the
	// journal root, not under it).
	t0 := time.Now()
	code2, out2 := cozyRun(root, "job", "submit", jobEndpoint+"/v1/census",
		"--in", payloadPath, "--input", ref+"="+jobStore(), "--follow")
	log.WriteString(fmt.Sprintf("the re-run exited %d in %d ms\n", code2, elapsedMS(t0)))
	rerun := field(out2, "job")
	doc := svc.jobDoc(rerun)
	result, _ := doc["result"].(map[string]any)
	check("the re-run reaches the SAME topology digest as an uninterrupted one",
		code2 == 0 && int64(numberOf(result, "tensors")) == bankedTensors,
		fmt.Sprint(result["topology_digest"]))
	pub, _ := doc["publication"].(map[string]any)
	check("and it published its own bundle", pub != nil && pub["repo"] != nil,
		fmt.Sprint(pub["repo"]))
	return log.String()
}

// exhaustRetryBudget plants a job BODY that kills its own executor the moment it runs, so
// every dispatched attempt ends on a NEUTRAL outcome the orchestrator's projection reads as
// infra-class. The projection then spends the durable budget one requeue at a time and,
// when it is gone, SETTLES the request typed and names the budget it spent.
//
// The plant is in the body rather than in the interpreter on purpose: a worker that cannot
// BOOT never gets an attempt, so it exercises `failQueued` and proves nothing at all about
// retryability. What must be proved is the projection over an ACCEPTED attempt's terminal.
func exhaustRetryBudget(svc *liveService, root, ref, payloadPath string) string {
	var log strings.Builder
	_, _ = cozyRun(root, "stop", "--all")
	sources, err := filepath.Glob(filepath.Join(root, "generations", "*", "source",
		"structural_census.py"))
	must("finding the installed job source", err)
	if len(sources) == 0 {
		must("the installed job source", fmt.Errorf("no structural_census.py under %s", root))
	}
	path := sources[0]
	original, err := os.ReadFile(path)
	must("reading the job source", err)
	defer func() { must("removing the plant", os.WriteFile(path, original, 0o644)) }()

	const anchor = `    state = scratch.checkpoint_dir(key="components")`
	planted := strings.Replace(string(original), anchor,
		"    import os as _os\n    _os._exit(70)\n"+anchor, 1)
	if planted == string(original) {
		must("planting the executor death", fmt.Errorf("the anchor line moved in %s", path))
	}
	must("planting the executor death", os.WriteFile(path, []byte(planted), 0o644))
	log.WriteString("planted a job body that kills its own executor on entry\n")

	t0 := time.Now()
	code, out := cozyRun(root, "job", "submit", jobEndpoint+"/v1/census",
		"--in", payloadPath, "--input", ref+"="+jobStore(), "--follow")
	log.WriteString(fmt.Sprintf("the job settled %d in %d ms\n", code, elapsedMS(t0)))
	log.WriteString(indent(out) + "\n")
	jobID := field(out, "job")
	doc := svc.jobDoc(jobID)
	budget := int64(numberOf(doc, "retry_budget"))
	spent := int64(numberOf(doc, "requeues"))
	attempts := int64(numberOf(doc, "attempts"))
	check("a job that ends NEUTRAL on every attempt settles typed rather than retrying forever",
		code == 11, firstLine(out))
	check("the record owner's PROJECTION spent the whole durable retry budget",
		spent == budget && budget > 0,
		fmt.Sprintf("%d/%d spent over %d attempt(s)", spent, budget, attempts))
	check("and the settlement NAMES the budget it exhausted",
		strings.Contains(out, "retry projection") || strings.Contains(out, "budget"),
		firstLine(out))
	log.WriteString("plant removed\n")
	return log.String()
}

// benchJob measures the orchestrator tax: the same job through `cozy job` versus cr-009's
// own banked runner numbers (264–313 ms whole job, 44–48 ms body, 216–241 ms fixed runner
// overhead). The difference is what an orchestrator, an HTTP hop, a record owner and a
// RECLAIMED worker cost on top of an in-process ephemeral supervisor — a job worker is
// terminal-and-reclaim, so every job pays its own spawn.
func benchJob(svc *liveService, root, ref, payloadPath string) {
	_, _ = cozyRun(root, "stop", "--all")
	t0 := time.Now()
	code, out := cozyRun(root, "job", "submit", jobEndpoint+"/v1/census",
		"--in", payloadPath, "--input", ref+"="+jobStore(), "--follow")
	coldMS := elapsedMS(t0)
	check("the cold job (no worker) completes", code == 0, firstLine(out))

	warm := []int64{}
	for i := 0; i < 3; i++ {
		t := time.Now()
		code, out := cozyRun(root, "job", "submit", jobEndpoint+"/v1/census",
			"--in", payloadPath, "--input", ref+"="+jobStore(), "--follow")
		warm = append(warm, elapsedMS(t))
		if code != 0 {
			fmt.Println(indent(out))
		}
	}
	sort.Slice(warm, func(i, j int) bool { return warm[i] < warm[j] })
	jobID := field(out, "job")
	doc := svc.jobDoc(jobID)
	metrics, _ := doc["metrics"].(map[string]any)
	pub, _ := doc["publication"].(map[string]any)
	t0 = time.Now()
	_ = svc.jobDoc(jobID)
	statusMS := elapsedMS(t0)

	fmt.Printf("  bench COLD job (spawn + directive + attempt + publish): %d ms\n", coldMS)
	fmt.Printf("  bench job wall, back to back (each spawns its own reclaimed worker): %v ms (min %d, max %d)\n",
		warm, warm[0], warm[len(warm)-1])
	fmt.Printf("  the worker's own runtime_ms for that attempt: %v (handler %v)\n",
		metrics["runtime_ms"], metrics["handler_ms"])
	fmt.Printf("  cr-009's banked runner: 264–313 ms whole job, 44–48 ms body, 216–241 ms fixed overhead\n")
	fmt.Printf("  the ORCHESTRATOR TAX is the difference: warm wall %d ms - runner %d ms\n",
		warm[0], 313)
	fmt.Printf("  publication commit: %v entries, %v B, committed at %v\n",
		pub["entries"], pub["bytes"], pub["committed_at"])
	fmt.Printf("  bench `cozy job status` round trip: %d ms\n", statusMS)
}

// --------------------------------------------------------------------------- helpers

// jobDoc reads ONE job's own API document over the real socket, with the real
// credential — the same document `cozy job status` renders, read from the source rather
// than re-parsed out of a rendering.
func (s *liveService) jobDoc(id string) map[string]any {
	return s.call("GET", "/v1/local/jobs/"+id, nil).json()
}

func numberOf(doc map[string]any, key string) float64 {
	if doc == nil {
		return -1
	}
	n, _ := doc[key].(float64)
	return n
}

// jobWorkerPID finds the live job worker's pid through the orchestrator's own listing —
// the protocol's identities, read from the API, never a pid this driver remembered.
func jobWorkerPID(svc *liveService) int {
	workers, _ := svc.call("GET", "/v1/local/workers", nil).json()["workers"].([]any)
	for _, raw := range workers {
		w, _ := raw.(map[string]any)
		if w["endpoint"] != jobEndpoint {
			continue
		}
		if exited, _ := w["exited"].(bool); exited {
			continue
		}
		if pid := int(numberOf(w, "pid")); pid > 0 {
			return pid
		}
	}
	return 0
}

// waitSettled polls one job's state document until it settles.
func waitSettled(svc *liveService, jobID string, timeout time.Duration) map[string]any {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		doc := svc.jobDoc(jobID)
		status, _ := doc["status"].(string)
		switch status {
		case "completed", "failed", "canceled":
			return doc
		}
		time.Sleep(time.Second)
	}
	return map[string]any{}
}

// drainOrder watches a set of jobs settle and records the ORDER they settled in.
func drainOrder(svc *liveService, ids []string, timeout time.Duration) []string {
	done := map[string]bool{}
	order := []string{}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) && len(order) < len(ids) {
		for _, id := range ids {
			if id == "" || done[id] {
				continue
			}
			switch svc.jobDoc(id)["status"] {
			case "completed", "failed", "canceled":
				done[id] = true
				order = append(order, id)
			}
		}
		time.Sleep(150 * time.Millisecond)
	}
	return order
}

func sameOrder(want, got []string) bool {
	if len(want) != len(got) {
		return false
	}
	for i := range want {
		if want[i] != got[i] {
			return false
		}
	}
	return true
}

func shortIDs(ids []string) []string {
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		if len(id) > 10 {
			id = id[:10]
		}
		out = append(out, id)
	}
	return out
}

// cozyRunEnv runs the product binary with extra IMPOSED environment values, through the
// product's own child-env allowlist.
func cozyRunEnv(root string, imposed []string, args ...string) (int, string) {
	cmd := niceCmd(cozyBinary(), args...)
	cmd.Env = append(childEnv(root), imposed...)
	data, _ := cmd.CombinedOutput()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	return code, string(data)
}

// --------------------------------------------------------------------------- jobcrash

// THE CRASH MATRIX, at the two lifecycle points that decide whether a publication is a
// promise or a fact. The RECORD OWNER is what dies here — a separate `cozy up` process,
// `kill -9`ed — because the publication row rides its terminal transaction, so the
// record owner's death is the only crash that can land between "the bytes are on disk" and
// "the publication exists".
//
//	before the bundle lands  ->  no publication is visible, and none ever was
//	after the bundle lands   ->  the bundle survives, and the request converges on ONE
//	                             terminal with exactly one visible output
func sectionJobCrash() {
	root := flag("home", filepath.Join(os.TempDir(), "cozy-live", "cl004-crash"))
	must("clearing the root", os.RemoveAll(root))
	must("creating the root", os.MkdirAll(root, 0o755))
	ref := "cozy/sdxl-pipeline@cr-008b"
	installJobEndpoint(root)
	port := freePort(3040)
	svc := startService(root, port, false)
	// A per-component dwell widens the window to crash INTO. A job that finished before
	// the kill would prove nothing.
	slow := jobPayload(root, ref, 2500)

	head("ARM A — kill the orchestrator BEFORE the bundle lands")
	jobA := submitJobAPI(svc, ref, slow)
	rootA := filepath.Join(root, "publications", "local", "_job-"+jobA)
	check("the attempt is running and has written nothing yet",
		waitFor(func() bool { return svc.jobDoc(jobA)["attempt"] != nil }, 60*time.Second) &&
			!exists(filepath.Join(rootA, "census")),
		rootA)
	svc.kill9()
	check("the orchestrator is gone", !svc.alive(), "")
	check("and no bundle was left behind at the publication root",
		!exists(filepath.Join(rootA, "census")), rootA)
	svc = startService(root, port, false)
	docA := svc.jobDoc(jobA)
	check("after the restart NOTHING the dead orchestrator had not committed is visible",
		docA["publication"] == nil && len(asList(docA["outputs"])) == 0,
		fmt.Sprintf("status %v", docA["status"]))
	// The orphaned worker was reconciled at boot; the queued attempt needs a replacement,
	// which the queue asks for on its own.
	settledA := waitSettled(svc, jobA, 20*time.Minute)
	check("the job CONVERGES on one terminal after the crash",
		settledA["status"] == "completed" || settledA["status"] == "failed",
		fmt.Sprint(settledA["status"]))
	// NO FALSE FAILURE AFTER A TRUE SUCCESS. The request's settled state and the
	// publication's stamped verdict are the same fact seen twice; a request whose
	// publication says SUCCEEDED and whose row says failed is the worst thing this system
	// could report, and it happened before the guard in `failQueued` landed.
	pubStatus, _ := settledA["publication"].(map[string]any)
	if pubStatus != nil {
		check("and its settled state AGREES with the terminal that produced its publication",
			(settledA["status"] == "completed") == (pubStatus["status"] == "SUCCEEDED"),
			fmt.Sprintf("request %v · publication %v", settledA["status"], pubStatus["status"]))
	}
	pubA, _ := settledA["publication"].(map[string]any)
	check("and the publication that finally exists is the one a terminal COMMITTED",
		pubA != nil && exists(filepath.Join(rootA, "census")),
		fmt.Sprintf("%v", settledA["status"]))
	check("exactly one output is visible over however many attempts it took",
		len(asList(settledA["outputs"])) == 1,
		fmt.Sprintf("%d output(s) over %v attempt(s)", len(asList(settledA["outputs"])),
			settledA["attempts"]))

	head("ARM B — kill the orchestrator the instant the bundle LANDS, before its terminal")
	jobB := submitJobAPI(svc, ref, slow)
	rootB := filepath.Join(root, "publications", "local", "_job-"+jobB)
	landed := filepath.Join(rootB, "census")
	// The window between the job's last write and `AcceptTerminal` is ~50 ms (the
	// orchestrator's own measurement of the transaction). Poll fast and kill on sight.
	raced := waitFast(func() bool { return exists(landed) }, 20*time.Minute)
	svc.kill9()
	killedAt := "the bundle was on disk"
	check("the bundle was observed on disk before the kill", raced, landed)
	svc = startService(root, port, false)
	check("THE BUNDLE SURVIVED the orchestrator's death", exists(landed), landed)
	settledB := waitSettled(svc, jobB, 20*time.Minute)
	pubB, _ := settledB["publication"].(map[string]any)
	// WHICH SIDE of the transaction the kill landed on is not something a driver may
	// assume — it is a race with a ~50 ms window — so it is REPORTED. Both sides make the
	// same claim: one terminal, one visible output, and the bundle intact.
	if pubB != nil && int64(numberOf(pubB, "attempt")) == 1 {
		killedAt = "the kill landed AFTER the terminal transaction committed"
	} else {
		killedAt = "the kill landed BEFORE the terminal transaction committed"
	}
	fmt.Printf("  %s (publication attempt %v, %v attempt(s) total)\n",
		killedAt, pubB["attempt"], settledB["attempts"])
	check("the job converges on ONE terminal", settledB["status"] == "completed" ||
		settledB["status"] == "failed", fmt.Sprint(settledB["status"]))
	check("exactly one output is visible — no partial and no duplicate",
		len(asList(settledB["outputs"])) == 1,
		fmt.Sprintf("%d output(s)", len(asList(settledB["outputs"]))))
	check("and the publication row names the SAME root the bundle is in",
		pubB != nil && pubB["root"] == rootB, fmt.Sprint(pubB["root"]))

	head("teardown")
	svc.stop()
}

// submitJobAPI submits over the real route and answers with the job id.
func submitJobAPI(svc *liveService, ref, payloadPath string) string {
	body, err := os.ReadFile(payloadPath)
	must("reading the payload", err)
	var input map[string]any
	must("decoding the payload", json.Unmarshal(body, &input))
	res := svc.call("POST", "/v1/local/jobs", map[string]any{
		"endpoint": jobEndpoint, "function": "census", "input": input,
		"trees": []string{ref + "=" + jobStore()},
	}, "Idempotency-Key", fmt.Sprintf("cl004-crash-%d", time.Now().UnixNano()))
	id, _ := res.json()["job_id"].(string)
	if id == "" {
		must("submitting the job", fmt.Errorf("%s", res.brief()))
	}
	return id
}

// newestJobStatus is the orchestrator's own answer about the most recently recorded job.
func newestJobStatus(svc *liveService) string {
	jobs, _ := svc.call("GET", "/v1/local/jobs?limit=1", nil).json()["jobs"].([]any)
	if len(jobs) == 0 {
		return ""
	}
	one, _ := jobs[0].(map[string]any)
	status, _ := one["status"].(string)
	return status
}

func exists(path string) bool { _, err := os.Stat(path); return err == nil }

// waitFast polls at 2 ms. The window it is racing is the terminal transaction's own
// ~50 ms, and a coarser poll would simply always lose it.
func waitFast(cond func() bool, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return true
		}
		time.Sleep(2 * time.Millisecond)
	}
	return false
}
