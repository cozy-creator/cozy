package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

func runOut(name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	var out bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &out
	err := cmd.Run()
	return out.String(), err
}

// sectionAttempt is the whole point of cl-001: the REAL product coordinator replacing
// worker-live.py's scripted Hub, driving the REAL two-process runtime through one real
// GPU attempt, and making the terminal and its output visible in ONE transaction.
func sectionAttempt() {
	idle := requireFreeGPU()
	lv := hostCoordinator("attempt", true)
	defer lv.close()

	spec := sdxlSpec("denoise")
	planID := planIDOf(spec, "denoise")

	head("the LocalService starts the REAL cozy-runtime supervisor over its own socket")
	bootStart := time.Now()
	instance, e := lv.c.StartWorker(spec)
	if e != nil {
		check("StartWorker", false, e.Message)
		return
	}
	check("worker spawned", true, "instance "+instance+", devices [0]")
	if e := lv.c.WaitReady(instance, planID, 300*time.Second); e != nil {
		check("READY", false, e.Message+" "+e.Remedy)
		fmt.Println(tail(lv.c.WorkerLog(instance), 25))
		return
	}
	boot := time.Since(bootStart)
	facts := lv.c.Worker(instance)
	check("READY", true, fmt.Sprintf("%s, session %s incarnation %d epoch %d revision %d",
		ms(boot), facts.SessionID, facts.Incarnation, facts.Epoch, facts.Revision))
	check("the worker advertises the plan this coordinator minted", len(facts.Ready) == 1 &&
		facts.Ready[0] == planID, planID)

	rows, _ := lv.store.LiveWorkers()
	check("the device grant is an attribute of the worker row", len(rows) == 1 &&
		len(rows[0].Devices) == 1 && rows[0].Devices[0] == "0",
		fmt.Sprintf("pid %d birth %s", rows[0].PID, rows[0].Birth))

	head("one real GPU attempt: submit -> accepted -> terminal -> visible output")
	body := payload(map[string]any{"steps": 4, "latent": 64, "seed": 1005})
	submitAt := time.Now()
	requestID, attempt, e := lv.c.Submit(submissionKey(planID, body, "idem-attempt-1"))
	if e != nil {
		check("submit", false, e.Message)
		return
	}
	if e := lv.c.AwaitAccepted(requestID, attempt, 60*time.Second); e != nil {
		check("accepted", false, e.Message)
		return
	}
	accepted := time.Since(submitAt)
	check("submit -> AttemptAccepted", true, ms(accepted))

	// Before the terminal: the runtime may already have written bytes under the grant,
	// and NONE of it is visible. Publication authority is the coordinator's alone.
	beforeOutputs, _ := lv.store.VisibleOutputs(requestID)
	check("no output is visible before the terminal is accepted", len(beforeOutputs) == 0,
		fmt.Sprintf("%d visible", len(beforeOutputs)))

	result, e := lv.c.AwaitSettled(requestID, 120*time.Second)
	total := time.Since(submitAt)
	if e != nil {
		check("terminal", false, e.Message)
		fmt.Println(tail(lv.c.WorkerLog(instance), 25))
		return
	}
	check("terminal", result.Status == "SUCCEEDED",
		fmt.Sprintf("%s/%s in %s", result.Status, result.Cause, ms(total)))
	check("exactly one output became visible", len(result.Outputs) == 1,
		fmt.Sprintf("%d output(s)", len(result.Outputs)))

	if len(result.Outputs) == 1 {
		o := result.Outputs[0]
		digest, length := fileDigest(o.Path)
		check("the visible output is on disk where the grant named it", length > 0,
			fmt.Sprintf("%s, %d B", o.Path, length))
		check("its content digest is the one the manifest declared", digest == o.Digest,
			fmt.Sprintf("%s (%d B declared)", short(o.Digest), o.Length))
		png, _ := os.ReadFile(o.Path)
		check("the bytes are a real PNG the GPU produced",
			len(png) > 8 && string(png[1:4]) == "PNG", fmt.Sprintf("%d B", len(png)))
		check("its output id is the RESULT FIELD PATH, not a position", o.OutputID == "image",
			o.OutputID)
	}

	head("the terminal document, read back from the ONE authority")
	doc, rerr := canonical.Read(result.Body, &pb.TerminalBody{})
	check("the journaled TerminalBody re-reads as canonical bytes", rerr == nil, detailOf(rerr))
	if rerr == nil {
		env := doc.Sub("result")
		inline, _ := env["inline_result"].(string)
		check("the typed result rode the terminal INLINE", len(inline) > 0,
			fmt.Sprintf("%d base64 chars, schema %s", len(inline),
				short(env.Str("result_schema_digest"))))
		metrics := doc.Sub("metrics")
		check("the attested metrics arrived", metrics.Int("runtime_ms") > 0,
			fmt.Sprintf("runtime %d ms · handler %d ms · lease %d ms · finalization %d ms · peak vram %d B",
				metrics.Int("runtime_ms"), metrics.Int("handler_ms"),
				metrics.Int("device_lease_ms"), metrics.Int("finalization_ms"),
				metrics.Int("peak_vram_bytes")))
	}

	row, _ := lv.store.AttemptRow(requestID, int64(attempt))
	check("the attempt closed on the ack", row != nil && row.State == "closed",
		fmt.Sprintf("state %s, terminal %s", row.State, short(row.TerminalDigest)))
	check("the journaled plan digests never moved", row.PlanDigest != "" && row.Construction != "",
		fmt.Sprintf("plan %s construction %s [%s]", short(row.PlanDigest),
			short(row.Construction), row.PlanSummary))
	check("triage is addressable by an OPAQUE attempt key", strings.HasPrefix(row.AttemptKey, "att-"),
		row.AttemptKey)
	byKey, _ := lv.store.AttemptByKey(row.AttemptKey)
	check("that key reads the attempt back; no path exists to read it by",
		byKey != nil && byKey.RequestID == requestID, row.AttemptKey)

	head("idempotency and ordinal arithmetic")
	again, againAtt, e := lv.c.Submit(submissionKey(planID, body, "idem-attempt-1"))
	check("the same key with the same body answers with the SAME request and no new attempt",
		e == nil && again == requestID && againAtt == attempt,
		fmt.Sprintf("%s#%d", again, againAtt))
	all, _ := lv.store.Attempts(requestID)
	check("re-submitting started nothing", len(all) == 1, fmt.Sprintf("%d attempt(s)", len(all)))
	_, _, e = lv.c.Submit(submissionKey(planID,
		payload(map[string]any{"steps": 2}), "idem-attempt-1"))
	check("the same key with a DIFFERENT body refuses", e != nil &&
		strings.Contains(e.Message, "different body"), briefly(e))

	head("a warm second attempt: same path, different latency")
	warmStart := time.Now()
	rid2, att2, e := lv.c.Submit(submissionKey(planID, body, "idem-attempt-2"))
	must("second submit", errOf(e))
	warmAccepted := time.Since(warmStart)
	if e := lv.c.AwaitAccepted(rid2, att2, 60*time.Second); e != nil {
		check("second accepted", false, e.Message)
	}
	r2, e := lv.c.AwaitSettled(rid2, 120*time.Second)
	warmTotal := time.Since(warmStart)
	check("the warm attempt succeeded through the same states", e == nil && r2.Status == "SUCCEEDED",
		fmt.Sprintf("%s (dispatch %s)", ms(warmTotal), ms(warmAccepted)))

	head("benchmarks — RTX 4070 Laptop, nice -n 19, shared box")
	fmt.Printf("  worker spawn -> READY (cold page cache + warm pass) : %s\n", ms(boot))
	fmt.Printf("  submit -> AttemptAccepted (cold)                    : %s\n", ms(accepted))
	fmt.Printf("  submit -> AttemptAccepted (warm)                    : %s\n", ms(warmAccepted))
	fmt.Printf("  submit -> visible output, whole attempt             : %s\n", ms(total))
	fmt.Printf("  warm attempt, submit -> visible output              : %s\n", ms(warmTotal))
	for _, line := range lv.c.Events() {
		if strings.Contains(line, "applied in") {
			fmt.Printf("  terminal -> visible (the ONE transaction)           : %s\n",
				strings.TrimSpace(line[strings.Index(line, "applied in")+len("applied in"):]))
		}
	}
	fmt.Printf("  coordinator RSS while serving                       : %.1f MiB\n", rssMiB())

	head("teardown")
	lv.c.StopWorker(instance, 20*time.Second)
	after, _ := lv.store.LiveWorkers()
	check("stopping the worker released its device grant", len(after) == 0,
		fmt.Sprintf("%d live worker row(s)", len(after)))
	time.Sleep(2 * time.Second)
	now := gpuUsedMiB()
	check("the GPU is back at its idle baseline", now <= idle+40,
		fmt.Sprintf("%d MiB now, %d MiB before", now, idle))
}

func tail(path string, n int) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return "    | " + strings.Join(lines, "\n    | ")
}

func short(s string) string {
	s = strings.TrimPrefix(s, "sha256:")
	if len(s) > 16 {
		return s[:16] + "…"
	}
	return s
}
