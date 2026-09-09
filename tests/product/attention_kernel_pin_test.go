package producttest

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/launch"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// THE SERVED HALF of the execution-path override (cr-125). `cozy-runtime run
// kernel.attention=<name>` already pins a kernel on a dev pod's ephemeral worker; these arms
// are the path `cozy run` takes instead — the reserved namespace, the request row, the
// InvocationSpec, and the placement records the pin must NOT reach.

func pinnedInvocation(t *testing.T, o *owner, request string, attempt uint64) canonical.Doc {
	t.Helper()
	row, problem := o.store.AttemptRow(request, int64(attempt))
	fatal(t, problem)
	spec, err := canonical.Read(row.InvocationCanonical, &pb.InvocationSpec{})
	must(t, err)
	return spec
}

// TestAServedRunPinsOneKernelPerRequestOnOnePreparedWorker is the arm the cr-125 amendment
// owes: TWO attempts on ONE prepared worker, two kernels, two records, one placement. The
// prepare-scoped design needed a re-prepare (or two rentals) to A/B two kernels; the
// per-request pin needs neither, which is the whole reason the wire change is worth its cost.
//
// The peer is `tests/support/fakeworker` — a second, independent implementation of the worker
// protocol — so the InvocationSpec asserted here is the document a real worker received over
// real bytes, not a projection the orchestrator kept in memory.
func TestAServedRunPinsOneKernelPerRequestOnOnePreparedWorker(t *testing.T) {
	o := hostOwner(t, "kernel-pin")
	spec := fakeSpec("pinned", "0", "--arm", "output", "--cozy-home", o.root)
	instance, _, e := o.c.EnsureWorker(spec)
	fatal(t, e)
	plan := planIDOf(t, spec)
	fatal(t, o.c.EnsurePlacementReady(instance, plan, ""))

	arms := []struct{ idem, kernel string }{
		{"pin-fa3", "flash-attn3-fp8"},
		{"pin-sdpa", "sdpa"},
	}
	digests := map[string]string{}
	for _, arm := range arms {
		sub := submission(plan, "fake/pinned", arm.idem, map[string]any{"n": arm.idem})
		sub.AttentionKernel = arm.kernel
		request, attempt, problem := o.c.Submit(sub)
		fatal(t, problem)
		if _, problem := o.c.Await(request, attempt, 60*time.Second); problem != nil {
			t.Fatalf("%s did not settle: %s", arm.idem, briefly(problem))
		}

		// THE REQUEST ROW carries it, because a REQUEUE must derive the same invocation: an
		// attempt that re-ran on the auto kernel would silently answer a different question.
		row, problem := o.store.RequestRow(request)
		fatal(t, problem)
		if row.AttentionKernel != arm.kernel {
			t.Fatalf("%s: request row kernel %q, want %q", arm.idem, row.AttentionKernel, arm.kernel)
		}
		// THE WIRE DOCUMENT the worker actually received.
		invocation := pinnedInvocation(t, o, request, attempt)
		if got := invocation.Str("attention_kernel"); got != arm.kernel {
			t.Fatalf("%s: InvocationSpec kernel %q, want %q", arm.idem, got, arm.kernel)
		}
		attemptRow, problem := o.store.AttemptRow(request, int64(attempt))
		fatal(t, problem)
		digests[arm.idem] = attemptRow.InvocationDigest

		// THE PIN IS NOT A PLACEMENT FACT. It decides HOW, and nothing about WHAT capacity
		// this request needs — so it may not appear in the rental, the ladder, or the
		// placement evidence the plan was chosen against.
		if row.Rental || row.RentalRequired || row.Worker != "" {
			t.Fatalf("%s: a pin moved the request onto rented capacity", arm.idem)
		}
		if len(row.Models) != 0 {
			t.Fatalf("%s: a pin reached the model ladder: %+v", arm.idem, row.Models)
		}
		if strings.Contains(string(attemptRow.ServingPlacementSet), arm.kernel) {
			t.Fatalf("%s: the kernel reached the placement set", arm.idem)
		}
	}

	// TWO DIFFERENT DOCUMENTS off ONE prepared worker. The pin is inside the digest, so the
	// two arms of an A/B are distinguishable attempts rather than one attempt told twice.
	if digests["pin-fa3"] == digests["pin-sdpa"] {
		t.Fatalf("two kernels produced one invocation identity: %+v", digests)
	}
	// ONE PLACEMENT DECISION: the same prepared worker served both arms.
	if worker := o.c.Worker(instance); worker == nil {
		t.Fatal("the prepared worker did not survive both pinned attempts")
	}
	if starts := countEvents(o, "worker.started"); starts > 1 {
		t.Fatalf("the pin moved the placement decision: %d worker starts", starts)
	}
}

// TestAnUnpinnedServedRunWritesNoKernelFieldAtAll. ABSENT BY DEFAULT, and absent means the
// key is not in the document: an ordinary run's InvocationSpec is byte-identical to what
// wire46 produced, so nothing about this change moved an existing request's identity.
func TestAnUnpinnedServedRunWritesNoKernelFieldAtAll(t *testing.T) {
	o := hostOwner(t, "kernel-absent")
	spec := fakeSpec("absent", "0", "--arm", "output", "--cozy-home", o.root)
	instance, _, e := o.c.EnsureWorker(spec)
	fatal(t, e)
	plan := planIDOf(t, spec)
	fatal(t, o.c.EnsurePlacementReady(instance, plan, ""))

	request, attempt, problem := o.c.Submit(
		submission(plan, "fake/absent", "no-pin", map[string]any{"prompt": "fox"}))
	fatal(t, problem)
	if _, problem := o.c.Await(request, attempt, 60*time.Second); problem != nil {
		t.Fatalf("an unpinned run did not settle: %s", briefly(problem))
	}
	row, problem := o.store.RequestRow(request)
	fatal(t, problem)
	if row.AttentionKernel != "" {
		t.Fatalf("an unpinned run recorded a kernel: %q", row.AttentionKernel)
	}
	if _, present := pinnedInvocation(t, o, request, attempt)["attention_kernel"]; present {
		t.Fatal("an unpinned run wrote the field into the InvocationSpec")
	}
}

// TestOneIdempotencyKeyCannotMeanTwoKernels — the pin is part of the submission's identity.
// A key re-sent with a different kernel is a different question and must CONFLICT rather than
// replay the first answer under the second name.
func TestOneIdempotencyKeyCannotMeanTwoKernels(t *testing.T) {
	o := hostOwner(t, "kernel-idem")
	spec := fakeSpec("idem", "0", "--arm", "idle")
	instance, _, e := o.c.EnsureWorker(spec)
	fatal(t, e)
	plan := planIDOf(t, spec)
	fatal(t, o.c.EnsurePlacementReady(instance, plan, ""))

	first := submission(plan, "fake/idem", "one-key", map[string]any{"n": 1})
	first.AttentionKernel = "flash-attn3"
	request, _, problem := o.c.Submit(first)
	fatal(t, problem)

	second := submission(plan, "fake/idem", "one-key", map[string]any{"n": 1})
	second.AttentionKernel = "sdpa"
	if _, _, problem := o.c.Submit(second); problem == nil {
		t.Fatal("one key answered for two different kernels")
	}
	// And the same key with the SAME kernel still replays the recorded request.
	again := submission(plan, "fake/idem", "one-key", map[string]any{"n": 1})
	again.AttentionKernel = "flash-attn3"
	replayed, _, problem := o.c.Submit(again)
	fatal(t, problem)
	if replayed != request {
		t.Fatalf("the same key and kernel started a second request: %s != %s", replayed, request)
	}
}

// TestTheKernelNamespaceIsClaimedBeforeThePayloadGrammar. `kernel.` is reserved WHOLE: a
// mistyped override refuses rather than landing on a package's own field, which is the only
// thing that makes a silent typo impossible. The refusal doubles as discoverability.
func TestTheKernelNamespaceIsClaimedBeforeThePayloadGrammar(t *testing.T) {
	ep := &launch.Entrypoint{Name: "fl2va"}
	ep.Request.Fields = []launch.Field{
		{Name: "prompt", Type: json.RawMessage(`"str"`), Wire: "required"},
		{Name: "seed", Type: json.RawMessage(`"int"`)},
	}
	for _, term := range []string{
		"kernel.gemm=rowwise",       // an axis this product does not have
		"kernel.attention",          // no value
		"kernel.attention=",         // an empty pin is not a pin
		"kernel.attention:=sdpa",    // the typed-JSON form of the payload grammar
		"kernel.Attention=sdpa",     // the namespace is EXACT-CASE, like `model.`
		"kernel.attention.fp8=sdpa", // an axis is one segment
	} {
		_, _, problem := launch.ParsePayload(ep, []string{"a cat", term}, "")
		if problem == nil {
			t.Fatalf("%s was accepted", term)
		}
		if problem.ErrName() != "kernel_override_unknown" {
			t.Fatalf("%s refused as %s, not kernel_override_unknown", term, problem.ErrName())
		}
		if !strings.Contains(problem.Remedy, "kernel.attention=<name>") {
			t.Fatalf("%s: the refusal does not name the reserved namespace: %s", term, problem.Remedy)
		}
	}

	payload, keys, problem := launch.ParsePayload(ep,
		[]string{"a cat", "seed=7000", "kernel.attention=flash-attn3-fp8"}, "")
	fatal(t, problem)
	if keys.AttentionKernel != "flash-attn3-fp8" {
		t.Fatalf("the pin was not claimed: %q", keys.AttentionKernel)
	}
	// AND IT LEFT THE PAYLOAD ALONE. A field named `kernel.attention` is unreachable — wire
	// names carry no dot — so the claim can never shadow a package's own declaration.
	if got := string(payload); got != `{"prompt":"a cat","seed":7000}` {
		t.Fatalf("the claimed pin disturbed the payload: %s", got)
	}
	if _, _, problem := launch.ParsePayload(ep,
		[]string{"a cat", "kernel.attention=sdpa", "kernel.attention=cudnn"}, ""); problem == nil {
		t.Fatal("one request pinned two attention kernels")
	}
}
