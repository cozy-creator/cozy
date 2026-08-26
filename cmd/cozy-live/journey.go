package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// cl-010's two sections. Both drive the PRODUCT BINARY exactly as a user types it —
// `cozy up`, `cozy run …`, `cozy logs …` — against a real service in its own process.
// Nothing here calls into internal/orchestrator or internal/api: if a verb works, it worked
// through the local client API over a socket.
//
//	journey  the full user path on the real card: up -> install -> describe -> fit ->
//	         start -> run (real SDXL, SSE rendered, image saved) -> logs -> re-run ->
//	         cancel -> stop -> down, with the numbers banked
//	verbs    the refusal matrix, no GPU: service down, wrong credential, unknown
//	         endpoint, majorless target, undeclared field, wrong scalar, a path as an
//	         attempt id, --cloud, an unresolved override, one key two bodies

const endpointRef = "cozy/sdxl-unet"

func sectionVerbs() {
	root := flag("home", filepath.Join(os.TempDir(), "cozy-live", "cl010-verbs"))
	must("clearing the root", os.RemoveAll(root))
	must("creating the root", os.MkdirAll(root, 0o755))

	head("the service is DOWN: every server-backed verb is typed exit 9 with the start remedy")
	for _, args := range [][]string{
		{"run", endpointRef + "/v1/denoise", "steps=2"},
		{"start", endpointRef},
		{"stop", "--all"},
		{"doctor"},
		{"logs", "att-0000"},
	} {
		code, out := cozyRun(root, args...)
		check("cozy "+strings.Join(args, " ")+" -> 9 + next: cozy up",
			code == 9 && strings.Contains(out, "error(unavailable)") && strings.Contains(out, "next: cozy up"),
			firstLine(out)+" [exit "+itoa(code)+"]")
	}

	head("an endpoint this host does not serve")
	installEndpoint(root)
	// The service comes up here: every arm below is about what a VERB refuses, and the
	// exit-9 gate above would otherwise answer for all of them.
	port := freePort(2880)
	svc := startService(root, port, false)
	defer svc.stop()
	code, out := cozyRun(root, "describe", "nobody/nothing")
	check("describe of an uninstalled endpoint -> 4",
		code == 4 && strings.Contains(out, "not installed"), firstLine(out))

	head("the target grammar: the semver-major is a REQUIRED path segment")
	for _, target := range []string{endpointRef + "/denoise", endpointRef, "denoise", endpointRef + "/vx/denoise"} {
		code, out := cozyRun(root, "run", target)
		check("cozy run "+target+" -> 2 usage",
			code == 2 && strings.Contains(out, "org/endpoint/vN/function"), firstLine(out))
	}

	head("the payload grammar, typed against the RECORDED schema — before a request exists")
	code, out = cozyRun(root, "run", endpointRef+"/v1/denoise", "mystery=1")
	check("an undeclared field -> 3, naming what the release declares",
		code == 3 && strings.Contains(out, "declares no request field") &&
			strings.Contains(out, "steps"), firstLine(out))
	code, out = cozyRun(root, "run", endpointRef+"/v1/denoise", "steps=lots")
	check("a wrong scalar type -> 3, naming the declared type",
		code == 3 && strings.Contains(out, "declared int"), firstLine(out))
	code, out = cozyRun(root, "run", endpointRef+"/v1/nosuch", "steps=2")
	check("an unknown function -> 4, listing the ones this release registers",
		code == 4 && strings.Contains(out, "registers no function"), firstLine(out))
	_, status, raw := cozyJSON(root, "status")
	check("and NO request was recorded by any of them — the refusal is client-side",
		status["requests"] == float64(0), fmt.Sprint(status["requests"])+" "+firstLine(raw))

	head("placement and overrides refuse by NAME rather than doing something else")
	code, out = cozyRun(root, "run", endpointRef+"/v1/denoise", "--cloud")
	check("--cloud -> 2, naming the host that does not exist yet",
		code == 2 && strings.Contains(out, "not_implemented"), firstLine(out))
	code, out = cozyRun(root, "run", endpointRef+"/v1/denoise", "--model", "acme/other")
	check("--model -> 2 override_unresolved (cl-005), never silently ignored",
		code == 2 && strings.Contains(out, "override_unresolved"), firstLine(out))
	// Two runtime flags this host advertises and has no WIRE FIELD for. A flag that
	// parses and does nothing is the same bug as an ignored override.
	code, out = cozyRun(root, "run", endpointRef+"/v1/denoise", "--seed", "7")
	check("--seed -> 2, naming the payload field that WOULD carry it",
		code == 2 && strings.Contains(out, "seed=7"), firstLine(out))
	code, out = cozyRun(root, "run", endpointRef+"/v1/denoise", "--offline")
	check("--offline -> 2, naming the standalone door that has it",
		code == 2 && strings.Contains(out, "not_implemented"), firstLine(out))
	code, out = cozyRun(root, "run", endpointRef+"/v1/denoise", "--timeout", "soon")
	check("--timeout with a non-duration -> 2", code == 2 && strings.Contains(out, "duration"),
		firstLine(out))

	head("a PATH is not an attempt id")
	for _, subject := range []string{"../../etc/passwd", "/etc/passwd", "./triage/x.json"} {
		code, out := cozyRun(root, "logs", subject)
		check("cozy logs "+subject+" -> 2, before any read",
			code == 2 && strings.Contains(out, "is a path"), firstLine(out))
	}
	code, out = cozyRun(root, "logs", "att-does-not-exist")
	check("an unknown attempt key -> 4 typed from the server's own envelope",
		code == 4 && strings.Contains(out, "not_found"), firstLine(out))

	head("the 0600 credential is what the CLI presents, and a wrong one is exit 5")
	credential := filepath.Join(root, "client.cred")
	original, err := os.ReadFile(credential)
	must("reading the client credential", err)
	info, err := os.Stat(credential)
	must("stat", err)
	check("the credential file is mode 0600", info.Mode().Perm() == 0o600,
		fmt.Sprintf("%#o", info.Mode().Perm()))
	must("planting a wrong credential", os.WriteFile(credential,
		[]byte("0000000000000000000000000000000000000000000000000000000000000000\n"), 0o600))
	code, out = cozyRun(root, "doctor")
	check("a WRONG credential -> 5, the server's own unauthenticated envelope",
		code == 5 && strings.Contains(out, "unauthenticated"), firstLine(out))
	must("widening the credential mode", os.Chmod(credential, 0o644))
	code, out = cozyRun(root, "doctor")
	check("a credential that became world-readable -> 5, refused before it is used",
		code == 5 && strings.Contains(out, "0600"), firstLine(out))
	must("restoring the credential", os.WriteFile(credential, original, 0o600))
	must("restoring its mode", os.Chmod(credential, 0o600))
	code, out = cozyRun(root, "doctor")
	check("restored: doctor answers 0 again", code == 0, firstLine(out))

	head("stop is idempotent when nothing runs")
	code, out = cozyRun(root, "stop", "--all")
	check("cozy stop --all with no worker -> 0", code == 0 && strings.Contains(out, "idempotent"),
		firstLine(out))
}

func sectionJourney() {
	idle := requireFreeGPU()
	root := flag("home", filepath.Join(os.TempDir(), "cozy-live", "cl010-journey"))
	must("clearing the root", os.RemoveAll(root))
	must("creating the root", os.MkdirAll(root, 0o755))

	head("cozy install — the endpoint arrives as a release, not as a document")
	t0 := time.Now()
	installEndpoint(root)
	installMS := elapsedMS(t0)
	code, out := cozyRun(root, "ls")
	check("cozy ls names the installed generation", code == 0 && strings.Contains(out, endpointRef),
		firstLine(out))
	fmt.Printf("  bench install (archive -> venv -> descriptor -> pin): %d ms\n", installMS)

	head("cozy describe — the recorded surface, no service and no subprocess")
	t0 = time.Now()
	code, out = cozyRun(root, "describe", endpointRef)
	describeMS := elapsedMS(t0)
	check("describe lists the release's functions", code == 0 && strings.Contains(out, "denoise"),
		firstLine(out))
	code, out = cozyRun(root, "describe", endpointRef+"/denoise")
	check("describe <fn> prints the request schema and the declared output paths",
		code == 0 && strings.Contains(out, "steps") && strings.Contains(out, "image"),
		firstLine(out))
	fmt.Printf("  bench describe: %d ms\n", describeMS)

	head("cozy up -d — one service, idempotent")
	port := freePort(2900)
	t0 = time.Now()
	svc := startService(root, port, false)
	upMS := elapsedMS(t0)
	defer svc.stop()
	check("the API answers on "+svc.addr, svc.alive(), fmt.Sprintf("pid %d", svc.cmd.Process.Pid))
	code, out = cozyRun(root, "status")
	check("cozy status sees the service, the endpoint and zero workers",
		code == 0 && strings.Contains(out, "up") && strings.Contains(out, endpointRef),
		firstLine(out))
	fmt.Printf("  bench cozy up -> API answering: %d ms\n", upMS)

	head("cozy fit — the runtime's own verdict, rendered")
	t0 = time.Now()
	code, out = cozyRun(root, "fit", endpointRef+"/denoise")
	fitMS := elapsedMS(t0)
	check("fit prints a verdict for denoise", strings.Contains(out, "denoise"), firstLine(out))
	fmt.Printf("  fit exit %d in %d ms\n", code, fitMS)
	fmt.Println(indent(out))

	head("cozy start — the prewarm verb, and it waits for a DISPATCHABLE plan")
	t0 = time.Now()
	code, out = cozyRun(root, "start", endpointRef)
	warmMS := elapsedMS(t0)
	check("cozy start -> 0 with the worker READY", code == 0 && strings.Contains(out, "ready"),
		firstLine(out))
	fmt.Println(indent(out))
	t0 = time.Now()
	code, out = cozyRun(root, "start", endpointRef)
	idemMS := elapsedMS(t0)
	check("a second cozy start is idempotent 0 and starts nothing",
		code == 0 && strings.Contains(out, "resident"), firstLine(out))
	fmt.Printf("  bench cold start -> READY: %d ms · idempotent re-start: %d ms\n", warmMS, idemMS)

	head("cozy run — one real SDXL request, SSE rendered, the image saved")
	outDir := filepath.Join(root, "out")
	t0 = time.Now()
	code, out = cozyRun(root, "run", endpointRef+"/v1/denoise", "steps=2", "latent=64",
		"--out", outDir)
	warmRunMS := elapsedMS(t0)
	fmt.Println(indent(out))
	check("cozy run -> 0", code == 0, firstLine(out))
	check("the SSE progress was RENDERED as it happened",
		strings.Contains(out, "progress") || strings.Contains(out, "accepted"), "")
	check("the typed result came back inline", strings.Contains(out, "result"), "")
	// Named by the DECLARED FIELD PATH plus the type the MANIFEST declares. cl-010 landed
	// against a runtime that hard-coded `application/octet-stream` on every OutputEntry,
	// so the file arrived as `image` with no extension and the run degraded loudly about
	// it; the pinned peer declares the real type, so the seam is closed and the name is
	// complete. The client still invents nothing — the suffix comes from the mime.
	saved := filepath.Join(outDir, "image.png")
	info, err := os.Stat(saved)
	check("the image is on disk under its DECLARED field path, named by its DECLARED type",
		err == nil && info != nil && info.Size() > 1000, saved+" "+sizeOf(info))
	check("and the run does NOT degrade about a missing media type — the manifest has one",
		!strings.Contains(out, "declares no media type"), "")
	check("the PNG the endpoint encoded is what landed",
		err == nil && isPNG(saved), "")
	attempt := field(out, "attempt_key")
	check("the run printed its attempt key", attempt != "", attempt)
	fmt.Printf("  bench warm run wall: %d ms\n", warmRunMS)

	head("cozy logs <attempt> — the retained bundle, verified, explained")
	code, out = cozyRun(root, "logs", attempt)
	check("cozy logs <attempt> -> 0 and renders the server's explain projection",
		code == 0 && strings.Contains(out, "terminal"), firstLine(out))
	fmt.Println(indent(out))

	head("idempotency: one key names one request forever")
	key := "cl010-idem-" + itoa(int(time.Now().Unix()))
	code1, doc1, _ := cozyJSON(root, "run", endpointRef+"/v1/denoise", "steps=2",
		"--idempotency-key", key)
	code2, doc2, out2 := cozyJSON(root, "run", endpointRef+"/v1/denoise", "steps=2",
		"--idempotency-key", key)
	check("the same key + the same body returns the SAME request",
		code1 == 0 && code2 == 0 && doc1["request"] == doc2["request"] && doc1["request"] != nil,
		fmt.Sprintf("%v == %v", doc1["request"], doc2["request"]))
	_ = out2
	code3, out3 := cozyRun(root, "run", endpointRef+"/v1/denoise", "steps=3",
		"--idempotency-key", key)
	check("the same key + a CHANGED body refuses 13 before any worker work",
		code3 == 13, firstLine(out3))
	code4, out4 := cozyRun(root, "run", endpointRef+"/v1/pair", "steps=2",
		"--idempotency-key", key)
	check("the same key naming a different FUNCTION refuses 13 too", code4 == 13, firstLine(out4))

	head("--timeout is a REQUEST DEADLINE the client enforces by cancelling")
	t0 = time.Now()
	code, out = cozyRun(root, "run", endpointRef+"/v1/stubborn", "steps=512", "latent=80",
		"--timeout", "3s")
	timeoutMS := elapsedMS(t0)
	check("a run past its --timeout exits 10, not 12: the deadline is what happened",
		code == 10 && strings.Contains(out, "deadline"), firstLine(out))
	check("and it cancelled through the orchestrator rather than walking away",
		strings.Contains(out, "expired"), "")
	fmt.Printf("  the deadline settled in %d ms\n", timeoutMS)

	head("cancel mid-attempt: the request is cancelled, the attempt's own terminal settles it")
	cancelled := cancelMidRun(root, port)
	check("a cancelled run exits 12 with the canceled terminal", cancelled, "")

	head("COLD and WARM traverse the same states")
	code, out = cozyRun(root, "stop", endpointRef)
	check("cozy stop drains the worker and BLOCKS until the process group is gone",
		code == 0 && strings.Contains(out, "true"), firstLine(out))
	t0 = time.Now()
	code, out = cozyRun(root, "run", endpointRef+"/v1/denoise", "steps=2", "--stream")
	coldRunMS := elapsedMS(t0)
	check("a COLD run (no worker) still exits 0 through the same path",
		code == 0, firstLine(out))
	cold := eventTypes(out)
	check("and its event sequence carries the same lifecycle states as the warm one",
		strings.Contains(cold, "request.submitted") && strings.Contains(cold, "request.dispatched") &&
			strings.Contains(cold, "request.accepted") && strings.Contains(cold, "request.completed"),
		cold)
	fmt.Printf("  bench cold run wall (spawn + 4.782 GiB fill + 2 steps): %d ms\n", coldRunMS)

	head("cozy stop --all, then the card")
	code, out = cozyRun(root, "stop", "--all")
	check("stop --all -> 0", code == 0, firstLine(out))
	used := gpuReleased(idle, 60*time.Second)
	check("the GPU is back to its baseline", used <= idle+40,
		fmt.Sprintf("%d MiB (idle was %d)", used, idle))
}

// cancelMidRun starts a LONG attempt and interrupts the client the way a person does:
// SIGINT to the `cozy run` process. The client must not die — it cancels through the
// orchestrator and keeps watching, because the attempt's own journaled terminal is what
// settles the request.
func cancelMidRun(root string, port int) bool {
	cmd := niceCmd(cozyBinary(), "run", endpointRef+"/v1/stubborn", "steps=512", "latent=80")
	cmd.Env = childEnv(root)
	var out strings.Builder
	cmd.Stdout, cmd.Stderr = &out, &out
	must("starting the long run", cmd.Start())
	// Wait for the attempt to be RUNNING before interrupting: cancelling a queued request
	// proves nothing about cancellation.
	deadline := time.Now().Add(300 * time.Second)
	running := false
	for time.Now().Before(deadline) && !running {
		time.Sleep(500 * time.Millisecond)
		running = strings.Contains(out.String(), "progress") || strings.Contains(out.String(), "accepted")
	}
	if !running {
		fmt.Println(indent(out.String()))
		return false
	}
	_ = cmd.Process.Signal(interruptSignal())
	err := cmd.Wait()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	fmt.Println(indent(out.String()))
	fmt.Printf("  the interrupted run exited %d (%v)\n", code, err)
	return code == 12 && strings.Contains(out.String(), "cancel")
}

// ------------------------------------------------------------------------- helpers

func firstLine(s string) string {
	for _, line := range strings.Split(strings.TrimSpace(s), "\n") {
		if strings.TrimSpace(line) != "" {
			return strings.TrimSpace(line)
		}
	}
	return ""
}

func indent(s string) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	for i, line := range lines {
		lines[i] = "    " + line
	}
	return strings.Join(lines, "\n")
}

// field reads one `key: value` line out of the compact record rendering.
func field(out, key string) string {
	for _, line := range strings.Split(out, "\n") {
		if name, value, ok := strings.Cut(strings.TrimSpace(line), ":"); ok &&
			strings.TrimSpace(name) == key {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

// eventTypes collects the `type` of every NDJSON frame `--stream` emitted.
func eventTypes(out string) string {
	types := []string{}
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, `{"type":"`) {
			continue
		}
		rest := strings.TrimPrefix(line, `{"type":"`)
		if i := strings.Index(rest, `"`); i > 0 {
			types = append(types, rest[:i])
		}
	}
	return strings.Join(types, " ")
}

func elapsedMS(from time.Time) int64 { return time.Since(from).Milliseconds() }

func sizeOf(info os.FileInfo) string {
	if info == nil {
		return "absent"
	}
	return fmt.Sprintf("%d B", info.Size())
}

func itoa(n int) string { return fmt.Sprint(n) }

// isPNG reads the file's own first bytes. The driver may look at what the product wrote;
// the product may not, which is why the extension is missing rather than sniffed.
func isPNG(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	var magic [8]byte
	if _, err := f.Read(magic[:]); err != nil {
		return false
	}
	return string(magic[1:4]) == "PNG"
}
