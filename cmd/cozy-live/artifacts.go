package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator/internal/records"
)

// cl-023's live section: the DURABLE ARTIFACT TRANSACTION, driven end to end against the
// real `internal/orchestrator` and the adversarial fake worker of `fakeartifact.go`.
//
// The claim under test is one sentence. For each declared artifact output the record owner
// records exactly ONE durable final intent — adopt the returned receipt into this job's own
// scratch root, or abandon the output with or without a committed receipt — asks the
// runtime to execute it, and records the exact completion before the outcome is
// acknowledged. Creator never mints a receipt, never accepts a job-authored snapshot id or
// path, and never promotes the scratch root publicly.
//
//	artifacts      the happy path, both abandon paths, and the two refusal matrices
//	artifactcrash  kill -9 at each edge of that sequence (artifactcrash.go)
//
// Neither needs a store, a GPU, or a real runtime peer.

// artifactOwnerScope mirrors `orchestrator.recordOwnerID`. It is spelled here rather than
// exported because the DRIVER recomputing the owner's own derivation independently is the
// scratch-root fence: if the two ever disagree, the derived-root check below fails.
const artifactOwnerScope = "cozy-local-client"

// derivedScratchRoot recomputes Creator's private root identity from the same three facts
// Creator uses. A root the WORKER supplied, or a path, or the public `<org>/_job-*` repo
// name would all fail this comparison.
func derivedScratchRoot(requestID, slot string) string {
	data, err := canonical.Write(map[string]canonical.Value{
		"format":                "cozy.runtime.ArtifactScratchRootIdentity/1",
		"owner_authority_scope": artifactOwnerScope,
		"request_id":            requestID,
		"output_slot":           slot,
	})
	must("recomputing the artifact scratch root identity", err)
	digest, err := canonical.Spell(canonical.Digest(data))
	must("spelling the artifact scratch root identity", err)
	return "artifact-scratch-" + digest[len("sha256:"):]
}

// artifactSubmission declares an ARTIFACT-ONLY job: rev5's OutputBinding has no kind, so
// the artifact subset is stated explicitly beside the InvocationSpec and the whole output
// set is that subset.
func artifactSubmission(planID, endpoint, idem string, slots ...string) orchestratorSubmission {
	arts := make([]orchestrator.ArtifactOutput, 0, len(slots))
	for _, slot := range slots {
		arts = append(arts, orchestrator.ArtifactOutput{
			OutputID: slot, MimeType: orchestrator.ArtifactSnapshotMime, MaxBytes: 1 << 20,
		})
	}
	return orchestratorSubmission{
		IdemKey: idem, Endpoint: endpoint, Entrypoint: "fake", PlanID: planID,
		Payload: payload(map[string]any{"artifact": true}),
		Outputs: append([]string(nil), slots...), ArtifactOutputs: arts,
		Kind: "job", Org: "local",
	}
}

// artifactWorker starts one fake worker on its own device envelope and returns the plan id
// dispatch will name.
func artifactWorker(lv *live, name, device, arm string) (string, bool) {
	spec := fakeSpec(name, device, "--arm", arm, "--cozy-home", lv.root)
	instance, _, e := lv.c.EnsureWorker(spec)
	if !check("the "+arm+" peer registers over the committed contract", e == nil, briefly(e)) {
		return "", false
	}
	planID := planIDOf(spec, "fake")
	if e := lv.c.EnsurePlacementReady(instance, planID); e != nil {
		check("the "+arm+" peer is dispatchable", false, e.Message)
		return "", false
	}
	return planID, true
}

func sectionArtifacts() {
	lv := hostCoordinator("cl023-artifacts", true)
	defer lv.close()

	artifactDeclarationArms(lv)
	artifactHappyPath(lv)
	artifactAbandonPaths(lv)
	artifactReceiptRedArms(lv)
	artifactResultRedArms(lv)
	artifactDurableLaws(lv.root)

	head("teardown")
	lv.c.Close(15 * time.Second)
	rows, _ := lv.store.LiveWorkers()
	check("every device grant is released", len(rows) == 0,
		fmt.Sprintf("%d live row(s)", len(rows)))
}

// --------------------------------------------------------------- the declaration fence

// artifactDeclarationArms refuses a bad artifact contract BEFORE a request exists, which
// is what makes "the declaration, never the arriving receipt, decides what an artifact
// slot is" a structural fact rather than a convention.
func artifactDeclarationArms(lv *live) {
	head("the artifact-output DECLARATION is refused before any request is recorded")
	snapshot := orchestrator.ArtifactSnapshotMime
	arm := func(label, wantName string, mutate func(*orchestratorSubmission)) {
		sub := artifactSubmission("plan-none", "fake/none", "cl023-decl-"+label, "model")
		mutate(&sub)
		_, _, e := lv.c.Submit(sub)
		check(label+" -> "+wantName, e != nil && e.ErrName() == wantName,
			fmt.Sprintf("%s: %s", nameOf(e), briefly(e)))
	}
	arm("artifact outputs on a SERVING submission", "artifact_output_not_job",
		func(s *orchestratorSubmission) { s.Kind = "" })
	arm("a slot that is not the snapshot MIME", "artifact_output_contract",
		func(s *orchestratorSubmission) { s.ArtifactOutputs[0].MimeType = "image/png" })
	arm("a slot with no new-byte cap", "artifact_output_contract",
		func(s *orchestratorSubmission) { s.ArtifactOutputs[0].MaxBytes = 0 })
	arm("a repeated slot id", "artifact_output_identity", func(s *orchestratorSubmission) {
		s.ArtifactOutputs = append(s.ArtifactOutputs, s.ArtifactOutputs[0])
	})
	arm("an artifact slot that is not one of the job's outputs", "mixed_job_output_kinds",
		func(s *orchestratorSubmission) { s.Outputs = []string{"image"} })
	arm("a job output that is not in the artifact set", "mixed_job_output_kinds",
		func(s *orchestratorSubmission) { s.Outputs = append(s.Outputs, "image") })
	arm("more slots than the protocol's receipt cap", "artifact_output_count_cap",
		func(s *orchestratorSubmission) {
			s.Outputs, s.ArtifactOutputs = nil, nil
			for i := 0; i < 17; i++ {
				id := fmt.Sprintf("slot-%02d", i)
				s.Outputs = append(s.Outputs, id)
				s.ArtifactOutputs = append(s.ArtifactOutputs, orchestrator.ArtifactOutput{
					OutputID: id, MimeType: snapshot, MaxBytes: 1 << 20,
				})
			}
		})
}

func nameOf(e *exit.Error) string {
	if e == nil {
		return "accepted"
	}
	return e.ErrName()
}

// --------------------------------------------------------------------- the happy path

func artifactHappyPath(lv *live) {
	head("ADOPT: every declared output commits, and NOTHING becomes public")
	planID, ok := artifactWorker(lv, "artifact", "0", "artifact")
	if !ok {
		return
	}
	sub := artifactSubmission(planID, "fake/artifact", "cl023-adopt", "model", "tokenizer")
	requestID, attempt, e := lv.c.Submit(sub)
	if !check("a two-slot artifact job is dispatched", e == nil, briefly(e)) {
		return
	}
	_, e = lv.c.Await(requestID, attempt, 60*time.Second)
	check("the attempt closes on its journaled outcome", e == nil, briefly(e))

	receipts, _ := lv.store.ArtifactReceiptsOf(requestID, int64(attempt))
	check("both exact receipts are durable — they commit WITH the outcome, before the ack",
		len(receipts) == 2, fmt.Sprintf("%d receipt(s)", len(receipts)))
	exact := true
	for _, r := range receipts {
		spelled, err := canonical.Spell(canonical.Digest(r.ReceiptBytes))
		if err != nil || spelled != r.ReceiptDigest || r.OwnerScope != artifactOwnerScope ||
			r.RequestID != requestID || r.TransactionID == "" {
			exact = false
		}
	}
	check("each stored digest hashes exactly the stored bytes, under this owner and this job",
		exact && len(receipts) == 2, receiptBrief(receipts))

	finals, _ := lv.store.ArtifactFinalizationsOf(requestID)
	check("ONE durable final intent per declared output slot", len(finals) == 2,
		finalizationBrief(finals))
	roots := map[string]bool{}
	adopted, derived, opaque := 0, true, true
	for _, f := range finals {
		if f.Disposition == "ADOPT" && f.ResultOutcome == "ADOPTED" {
			adopted++
		}
		if f.ScratchRootID != derivedScratchRoot(requestID, f.OutputSlot) {
			derived = false
		}
		// The root is a SEMANTIC id. A path separator or the public `<org>/_job-*` repo
		// name appearing here would be the substitution this fence exists to refuse.
		if strings.ContainsAny(f.ScratchRootID, `/\`) || strings.Contains(f.ScratchRootID, "_job-") {
			opaque = false
		}
		roots[f.ScratchRootID] = true
	}
	check("every intent is ADOPT and every result is the exact ADOPTED completion",
		adopted == 2, finalizationBrief(finals))
	check("the scratch root is CREATOR'S OWN derivation, recomputed independently here",
		derived && len(finals) == 2, rootBrief(finals))
	check("and it is an opaque semantic id — never a path, never the public scratch repo",
		opaque && len(roots) == 2, fmt.Sprintf("%d distinct root(s)", len(roots)))
	evidence := true
	for _, f := range finals {
		if f.ResultReceiptDigest != f.ReceiptDigest || f.ResultReceiptDigest == "" {
			evidence = false
		}
	}
	check("each ADOPTED result returned the EXACT receipt the intent named", evidence,
		finalizationBrief(finals))

	line, held := waitEvent(lv, "remains unacked behind 2 artifact finalization(s)", 5*time.Second)
	check("the outcome ack was WITHHELD until both finalizations were answered", held, trimLog(line))
	line, recorded := waitEvent(lv, "durably recorded 2 artifact receipt(s)", 5*time.Second)
	check("and the receipts were durable before acknowledgement", recorded, trimLog(line))

	row, _ := lv.store.AttemptRow(requestID, int64(attempt))
	check("the attempt is CLOSED on one outcome, after the finalizations",
		row != nil && row.State == "closed" && row.TerminalStatus == "SUCCEEDED",
		fmt.Sprintf("%s/%s", row.TerminalStatus, row.State))

	head("public visibility is a LATER, EXPLICIT act — the red arm")
	outs, _ := lv.store.VisibleOutputs(requestID)
	pub, _ := lv.store.PublicationOf(requestID)
	check("an adopted artifact job publishes NO output row and NO publication",
		len(outs) == 0 && pub == nil,
		fmt.Sprintf("%d output(s), publication %v", len(outs), pub != nil))
	entries := publicRepoEntries(lv, requestID)
	check("and the public scratch repository holds no addressable entry",
		len(entries) == 0, fmt.Sprintf("%v", entries))
}

// publicRepoEntries lists everything ADDRESSABLE in this job's public scratch repository.
// `.staging` is outside the output-id namespace and is the attempt's own write area, so it
// is not a publication.
func publicRepoEntries(lv *live, requestID string) []string {
	rows, err := os.ReadDir(lv.l.PublicationRoot("local", requestID))
	if err != nil {
		return nil
	}
	out := []string{}
	for _, row := range rows {
		if !strings.HasPrefix(row.Name(), ".") {
			out = append(out, row.Name())
		}
	}
	return out
}

// --------------------------------------------------------------------- abandon paths

func artifactAbandonPaths(lv *live) {
	head("ABANDON: a committed artifact whose job then failed is released WITH its receipt")
	planID, ok := artifactWorker(lv, "artifactabandon", "1", "artifactabandon")
	if ok {
		sub := artifactSubmission(planID, "fake/artifactabandon", "cl023-abandon", "model")
		requestID, attempt, e := lv.c.Submit(sub)
		check("the failing artifact job is dispatched", e == nil, briefly(e))
		_, e = lv.c.Await(requestID, attempt, 60*time.Second)
		check("it reaches a FINAL failure, not a requeue", e != nil, briefly(e))
		finals, _ := lv.store.ArtifactFinalizationsOf(requestID)
		one := len(finals) == 1
		check("the intent is ABANDON, carries the committed receipt, and names NO root",
			one && finals[0].Disposition == "ABANDON" && finals[0].ReceiptDigest != "" &&
				finals[0].ScratchRootID == "", finalizationBrief(finals))
		check("and the ABANDONED result returned that same receipt as evidence",
			one && finals[0].ResultOutcome == "ABANDONED" &&
				finals[0].ResultReceiptDigest == finals[0].ReceiptDigest, finalizationBrief(finals))
	}

	head("ABANDON_UNCOMMITTED: a final failure with no receipt closes its open slot")
	planID, ok = artifactWorker(lv, "artifactuncommitted", "2", "artifactuncommitted")
	if !ok {
		return
	}
	sub := artifactSubmission(planID, "fake/artifactuncommitted", "cl023-uncommitted", "model")
	requestID, attempt, e := lv.c.Submit(sub)
	check("the receipt-free failing job is dispatched", e == nil, briefly(e))
	_, e = lv.c.Await(requestID, attempt, 60*time.Second)
	check("it reaches a FINAL failure", e != nil, briefly(e))
	receipts, _ := lv.store.ArtifactReceiptsOf(requestID, int64(attempt))
	finals, _ := lv.store.ArtifactFinalizationsOf(requestID)
	one := len(finals) == 1
	check("no receipt was recorded, because none was ever committed", len(receipts) == 0,
		fmt.Sprintf("%d receipt(s)", len(receipts)))
	check("the intent is ABANDON_UNCOMMITTED with neither a receipt nor a root",
		one && finals[0].Disposition == "ABANDON_UNCOMMITTED" && finals[0].ReceiptDigest == "" &&
			finals[0].ScratchRootID == "", finalizationBrief(finals))
	check("and it completes ABANDONED, carrying no invented evidence",
		one && finals[0].ResultOutcome == "ABANDONED" && finals[0].ResultReceiptDigest == "",
		finalizationBrief(finals))
}

// ------------------------------------------------------------------ the refusal matrices

// artifactReceiptRedArms drives eleven inadmissible receipt outcomes against ONE open
// attempt, then the admissible one. Every fence in the receipt plane gets an observed red
// arm, planted by the real peer and refused by the real record owner.
func artifactReceiptRedArms(lv *live) {
	head("the RECEIPT refusal matrix: eleven inadmissible outcomes, then the admissible one")
	planID, ok := artifactWorker(lv, "artifactred", "3", "artifactred")
	if !ok {
		return
	}
	sub := artifactSubmission(planID, "fake/artifactred", "cl023-red", "model")
	requestID, attempt, e := lv.c.Submit(sub)
	if !check("the adversarial artifact job is dispatched", e == nil, briefly(e)) {
		return
	}
	_, e = lv.c.Await(requestID, attempt, 90*time.Second)
	check("the attempt closes on its ONE admissible outcome", e == nil, briefly(e))

	for _, arm := range []struct{ label, substr string }{
		{"a receipt digest that does not hash its bytes", "artifact_receipt_digest_mismatch"},
		{"a nested tensorfs digest that does not hash the nested bytes",
			"artifact_receipt_nested_digest_mismatch"},
		{"a BARE SNAPSHOT ID where the nested receipt belongs", "artifact_receipt_incomplete"},
		{"a local PATH planted in the closed receipt document", `unknown field "path"`},
		{"a JOB-AUTHORED authority scope", `"job/self-declared"`},
		{"a receipt minted for ANOTHER JOB", "job-somebody-else"},
		{"a receipt naming another InvocationSpec",
			strings.TrimPrefix(spellString("another invocation"), "sha256:")[:16]},
		{"an UNDECLARED output slot", "is not in the persisted artifact-output subset"},
		{"receipts that are not in strict slot order", "artifact_receipt_order"},
		{"a SUCCEEDED artifact job returning no receipt", "required slot(s)"},
		{"an artifact-only job also returning an output manifest", "output-manifest entries"},
	} {
		line, seen := waitEvent(lv, arm.substr, 3*time.Second)
		check("REFUSED: "+arm.label, seen, trimLog(line))
	}

	receipts, _ := lv.store.ArtifactReceiptsOf(requestID, int64(attempt))
	finals, _ := lv.store.ArtifactFinalizationsOf(requestID)
	check("exactly ONE receipt and ONE adopted intent survived eleven refusals",
		len(receipts) == 1 && len(finals) == 1 && finals[0].ResultOutcome == "ADOPTED",
		fmt.Sprintf("%d receipt(s), %s", len(receipts), finalizationBrief(finals)))
}

// artifactResultRedArms drives the finalize-result plane: six inadmissible answers to one
// persisted intent, the exact answer, a byte-identical replay of it, and a CHANGED answer
// after the transaction completed.
func artifactResultRedArms(lv *live) {
	head("the FINALIZE-RESULT refusal matrix, over ONE first-wins intent")
	planID, ok := artifactWorker(lv, "artifactresultred", "4", "artifactresultred")
	if !ok {
		return
	}
	sub := artifactSubmission(planID, "fake/artifactresultred", "cl023-resultred", "model")
	requestID, attempt, e := lv.c.Submit(sub)
	if !check("the adversarial finalize job is dispatched", e == nil, briefly(e)) {
		return
	}
	_, e = lv.c.Await(requestID, attempt, 90*time.Second)
	check("the attempt closes on the ONE exact result", e == nil, briefly(e))

	for _, arm := range []struct{ label, substr string }{
		{"a result digest that does not hash its bytes", "result digest does not hash"},
		{"a result for a slot with NO persisted intent", "no persisted first-wins intent"},
		{"an envelope naming a slot the document does not", "envelope/document/owner divergence"},
		{"a result CONTRADICTING the persisted intent", "contradicts persisted ADOPT intent"},
		{"ADOPTED returning a DIFFERENT receipt", "differs from persisted"},
		{"ADOPTED omitting the exact adopted receipt", "inadmissible result document"},
	} {
		line, seen := waitEvent(lv, arm.substr, 5*time.Second)
		check("REFUSED: "+arm.label, seen, trimLog(line))
	}
	line, seen := waitEvent(lv, requestID+"/model ADOPTED persisted before outcome ack", 5*time.Second)
	check("the EXACT result is journaled before the outcome is acknowledged", seen, trimLog(line))
	// A CHANGED result cannot even reach the journal: every field an ArtifactFinalizeResult
	// could vary is fenced by the intent, so the second attempt to change one is refused at
	// the receipt fence — twice observed, once before the completion and once after it.
	// The journal's own "already completed" conflict is armed against the real store below.
	changed := awaitCount(lv, requestID, "REFUSED: returned receipt", 2, 15*time.Second)
	check("a CHANGED result is refused BEFORE completion and again AFTER it", changed == 2,
		fmt.Sprintf("%d refusal(s) of a changed receipt", changed))
	completions := countEventsIn(lv, requestID, "persisted before outcome ack")
	check("and the identical replay applied NOTHING twice — one durable completion",
		completions == 1, fmt.Sprintf("%d completion line(s)", completions))

	finals, _ := lv.store.ArtifactFinalizationsOf(requestID)
	check("one intent, ADOPTED once, holding the exact receipt",
		len(finals) == 1 && finals[0].ResultOutcome == "ADOPTED" &&
			finals[0].ResultReceiptDigest == finals[0].ReceiptDigest, finalizationBrief(finals))
}

// ------------------------------------------------------------------ the durable laws

// artifactDurableLaws drives the REAL records authority directly — a real SQLite root, the
// real schema, the real transactions — over the four laws no wire arm can reach in one
// attempt: first-wins across attempts, the adopt/abandon conflict, the commit-versus-
// uncommitted-abandon race, and the idempotent completion.
func artifactDurableLaws(root string) {
	head("the first-wins laws, against the real records authority")
	path := filepath.Join(root, "artifact-laws.sqlite3")
	must("clearing the law root", os.RemoveAll(path))
	store, e := records.Open(path)
	must("opening the law root", errOf(e))
	defer store.Close()

	declared, err := json.Marshal([]orchestrator.ArtifactOutput{
		{OutputID: "model", MimeType: orchestrator.ArtifactSnapshotMime, MaxBytes: 1 << 20},
		{OutputID: "tokenizer", MimeType: orchestrator.ArtifactSnapshotMime, MaxBytes: 1 << 20},
	})
	must("rendering the declared artifact outputs", err)
	must("spawning the law worker", errOf(store.SpawnWorker(records.WorkerProcess{
		InstanceID: "ins-art", Endpoint: "fake/artifact", ReleaseID: "rel-art", WorkerID: "w-art",
		Devices: []string{""}, PID: os.Getpid(), Birth: "art", State: "spawned",
	})))
	req, _, e := store.Submit(records.Request{
		ID: records.NewID("job"), IdemKey: "art-idem", BodyDigest: "sha256:art",
		Endpoint: "fake/artifact", Entrypoint: "fake", PlanID: "plan-art",
		Payload: []byte("{}"), Kind: "job", Org: "local",
		Outputs: "model,tokenizer", ArtifactOutputs: string(declared),
	})
	must("submitting the law request", errOf(e))

	// ONE InvocationSpec across both attempts: the artifact transaction is keyed by
	// (request, spec, slot), so a re-dispatch of the same work meets the same intent.
	const spec = "sha256:" + "11111111111111111111111111111111" + "1111111111111111111111111111111a"
	dispatch := func() int64 {
		n, e := store.Dispatch(records.Attempt{
			RequestID: req.ID, InstanceID: "ins-art", SessionID: "boot-art",
			InvocationDigest: spec, InvocationCanonical: []byte("{}"),
			ArtifactOutputs: string(declared),
		})
		must("dispatching a law attempt", errOf(e))
		// The offer boundary is durable and separate: a terminal is only accepted against an
		// assignment a worker was actually offered.
		must("offering a law attempt", errOf(store.OfferDispatch(req.ID, n, "boot-art")))
		return n
	}
	intent := func(slot, disposition, receipt, rootID, decision string) records.ArtifactFinalization {
		return records.ArtifactFinalization{
			RequestID: req.ID, InstanceID: "ins-art", OwnerScope: artifactOwnerScope,
			InvocationDigest: spec, OutputSlot: slot, Disposition: disposition,
			ReceiptDigest: receipt, ScratchRootID: rootID,
			DecisionDigest: decision, DecisionBytes: []byte(decision),
		}
	}
	settle := func(attempt int64, digest string, finals ...records.ArtifactFinalization) (bool, *exit.Error) {
		for i := range finals {
			finals[i].Attempt = attempt // the intent is FK'd to the attempt that minted it
		}
		return store.AcceptTerminal(records.Terminal{
			RequestID: req.ID, Attempt: attempt, SessionID: "boot-art", InvocationDigest: spec,
			TerminalID: records.NewID("trm"), TerminalDigest: digest, Status: "SUCCEEDED",
			RequestState: "queued", ArtifactFinalizations: finals,
		})
	}

	adopt := intent("model", "ADOPT", "sha256:aaa", derivedScratchRoot(req.ID, "model"), "sha256:d1")
	uncommitted := intent("tokenizer", "ABANDON_UNCOMMITTED", "", "", "sha256:d2")

	a1 := dispatch()
	applied, e := settle(a1, "sha256:t1", adopt, uncommitted)
	check("attempt 1 journals ADOPT(model) and ABANDON_UNCOMMITTED(tokenizer) in ONE transaction",
		applied && e == nil, briefly(e))
	must("closing the settled law attempt", errOf(store.Closed(req.ID, a1)))

	a2 := dispatch()
	_, e = settle(a2, "sha256:t2", intent("model", "ABANDON", "sha256:aaa", "", "sha256:d3"))
	check("a later ABANDON for the same slot CONFLICTS — the first decision wins forever",
		e != nil && e.Code == exit.Conflict &&
			strings.Contains(e.Message, "already has final intent"), briefly(e))
	_, e = settle(a2, "sha256:t2", intent("tokenizer", "ADOPT", "sha256:bbb",
		derivedScratchRoot(req.ID, "tokenizer"), "sha256:d4"))
	check("and an UNCOMMITTED-ABANDON slot can never turn into adoption",
		e != nil && e.Code == exit.Conflict, briefly(e))
	held, _ := store.ArtifactFinalization(req.ID, spec, "model")
	check("the durable intent is untouched by both refusals",
		held != nil && held.Disposition == "ADOPT" && held.DecisionDigest == "sha256:d1",
		fmt.Sprintf("%s %s", held.Disposition, held.DecisionDigest))

	// The IDENTICAL decision is not a conflict: replay is how a lost frame converges.
	applied, e = settle(a2, "sha256:t2", adopt, uncommitted)
	check("the IDENTICAL decisions replay and write nothing a second time",
		applied && e == nil, briefly(e))
	all, _ := store.ArtifactFinalizationsOf(req.ID)
	check("two attempts left exactly TWO intent rows", len(all) == 2, finalizationBrief(all))

	head("the completion is idempotent, and a changed one conflicts forever")
	result := records.ArtifactFinalization{
		RequestID: req.ID, InstanceID: "ins-art", InvocationDigest: spec, OutputSlot: "model",
		ResultOutcome: "ADOPTED", ResultDigest: "sha256:r1", ResultBytes: []byte("r1"),
		ResultReceiptDigest: "sha256:aaa", ResultReceiptBytes: []byte("receipt"),
	}
	applied, e = store.RecordArtifactFinalizeResult(result)
	check("the exact result completes the transaction", applied && e == nil, briefly(e))
	applied, e = store.RecordArtifactFinalizeResult(result)
	check("the IDENTICAL result replays and applies nothing twice", !applied && e == nil, briefly(e))
	changed := result
	changed.ResultDigest, changed.ResultBytes = "sha256:r2", []byte("r2")
	_, e = store.RecordArtifactFinalizeResult(changed)
	check("a CHANGED result for the same transaction conflicts",
		e != nil && e.Code == exit.Conflict &&
			strings.Contains(e.Message, "already completed with"), briefly(e))
	foreign := result
	foreign.InstanceID = "ins-somebody-else"
	_, e = store.RecordArtifactFinalizeResult(foreign)
	check("a result from a worker that does not own the transaction refuses",
		e != nil && e.Code == exit.Conflict, briefly(e))
	orphan := result
	orphan.OutputSlot = "never-declared"
	_, e = store.RecordArtifactFinalizeResult(orphan)
	check("and a result with NO persisted intent refuses NOT_FOUND — receipts follow intents",
		e != nil && e.Code == exit.NotFound, briefly(e))

	pending, _ := store.PendingArtifactFinalizations(req.ID, a1)
	check("only the still-open slot is pending a replay", len(pending) == 1 &&
		pending[0].OutputSlot == "tokenizer", finalizationBrief(pending))
}

// ------------------------------------------------------------------------- renderings

func receiptBrief(rows []records.ArtifactReceipt) string {
	out := make([]string, 0, len(rows))
	for _, r := range rows {
		out = append(out, fmt.Sprintf("%s=%s(%dB)", r.OutputSlot, shortSHA(r.ReceiptDigest),
			len(r.ReceiptBytes)))
	}
	return strings.Join(out, " ")
}

func finalizationBrief(rows []records.ArtifactFinalization) string {
	out := make([]string, 0, len(rows))
	for _, f := range rows {
		state := f.Disposition
		if f.ResultOutcome != "" {
			state += "->" + f.ResultOutcome
		}
		out = append(out, fmt.Sprintf("%s %s", f.OutputSlot, state))
	}
	if len(out) == 0 {
		return "no finalization rows"
	}
	return strings.Join(out, " · ")
}

func rootBrief(rows []records.ArtifactFinalization) string {
	out := make([]string, 0, len(rows))
	for _, f := range rows {
		out = append(out, f.OutputSlot+"="+f.ScratchRootID)
	}
	return strings.Join(out, " ")
}

// countEventsIn counts orchestrator log lines naming this request AND the given substring.
func countEventsIn(lv *live, requestID, substr string) int {
	n := 0
	for _, line := range lv.c.Events() {
		if strings.Contains(line, requestID) && strings.Contains(line, substr) {
			n++
		}
	}
	return n
}

// awaitCount waits until at least `want` matching lines exist, and answers how many there are.
func awaitCount(lv *live, requestID, substr string, want int, timeout time.Duration) int {
	deadline := time.Now().Add(timeout)
	for {
		if n := countEventsIn(lv, requestID, substr); n >= want || time.Now().After(deadline) {
			return n
		}
		time.Sleep(50 * time.Millisecond)
	}
}
