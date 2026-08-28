package main

import (
	"bytes"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
	pb "github.com/cozy-creator/cozy-creator/protocol/cozy/worker/v1"
)

// THE ADVERSARY'S ARTIFACT HALF (cl-023/cr-030/th-049).
//
// The fake worker stands in for cozy-runtime's artifact sink: it AUTHORS the exact
// `ArtifactReceipt/1` documents an `AttemptOutcomeBody/3` carries, and it HOSTS the paired
// `ArtifactFinalizeRequest` → `ArtifactFinalizeResult` exchange. Every byte here is minted
// the way worker-protocol's frozen `fixtures/canonical/artifact_*.json` are: the digest is
// over the exact document bytes, the TensorFS receipt rides nested as bytes plus its own
// digest, and the slots are strictly ordered.
//
// It exists so the RecordOwner's artifact fences have a real peer to refuse. Creator never
// mints a receipt, never derives a snapshot id, and never learns a path — so this side owns
// all three, and the red arms are this side lying about them.
const artifactSnapshotMime = "application/vnd.cozy.tensorfs.snapshot"

// artifactSlots reads the ARTIFACT SUBSET out of the offer's own InvocationSpec. Rev5's
// OutputBinding carries no kind, so the snapshot MIME is the discriminator both ends
// agree on: this adversary discovers its artifact contract from the digest-fenced
// document, never from a flag the driver handed it.
func artifactSlots(offer *pb.AttemptOffer) []string {
	doc, err := canonical.Read(offer.InvocationSpecCanonicalBytes, &pb.InvocationSpec{})
	if err != nil {
		return nil
	}
	out := []string{}
	for _, binding := range doc.List("outputs") {
		if binding.Str("mime_type") == artifactSnapshotMime {
			out = append(out, binding.Str("output_id"))
		}
	}
	sort.Strings(out)
	return out
}

// receiptPlant names ONE deviation from an exact receipt. The zero value is the honest
// document; every other field is a red arm the RecordOwner has to refuse by itself.
type receiptPlant struct {
	ownerScope   string // a foreign record-owner authority
	requestID    string // another job's receipt
	specDigest   string // another InvocationSpec's receipt
	slot         string // a slot this job never declared
	transaction  string // a transaction this worker never opened
	breakDigest  bool   // the ref digest does not hash the carried bytes
	breakNested  bool   // the tensorfs digest does not hash the nested bytes
	bareSnapshot bool   // a bare snapshot id where the nested receipt belongs
	plantKey     string // a key the closed ArtifactReceipt document has no slot for
	plantValue   string
}

// tensorfsReceipt is TENSORFS's own document. Creator never parses it: it rides inside the
// ArtifactReceipt as exact bytes plus a digest, which IS the storage-identity boundary.
func tensorfsReceipt(slot, transaction string) ([]byte, string) {
	data, err := canonical.Write(map[string]canonical.Value{
		"format":         "tensorfs.derived_snapshot_receipt/1",
		"output_slot":    slot,
		"snapshot_id":    spellString("snapshot/" + transaction + "/" + slot),
		"transaction_id": transaction,
	})
	if err != nil {
		panic("the fake worker cannot mint a tensorfs receipt: " + err.Error())
	}
	spelled, _ := canonical.Spell(canonical.Digest(data))
	return data, spelled
}

func spellString(s string) string {
	spelled, _ := canonical.Spell(canonical.Digest([]byte(s)))
	return spelled
}

// receiptRef mints one ArtifactReceipt and wraps it in the ref an AttemptOutcomeBody/3 and
// an ArtifactFinalizeResult both carry. An honest ref is remembered under its slot, so the
// finalize exchange returns the SAME bytes rather than re-minting them.
func (f *fakeControl) receiptRef(requestID, specDigest, slot string,
	plant receiptPlant) *pb.ArtifactReceiptRef {
	scope, request, spec, name := f.ownerScope, requestID, specDigest, slot
	if plant.ownerScope != "" {
		scope = plant.ownerScope
	}
	if plant.requestID != "" {
		request = plant.requestID
	}
	if plant.specDigest != "" {
		spec = plant.specDigest
	}
	if plant.slot != "" {
		name = plant.slot
	}
	transaction := plant.transaction
	if transaction == "" {
		transaction = spellString("txn/" + f.bootID + "/" + request + "/" + name)
	}
	nested, nestedDigest := tensorfsReceipt(name, transaction)
	if plant.breakNested {
		nestedDigest = spellString("not the nested tensorfs receipt")
	}
	if plant.bareSnapshot {
		// A BARE SNAPSHOT ID is exactly what Creator must refuse: an identity with no
		// receipt behind it. The nested bytes go away and the id is offered in their place.
		nested, nestedDigest = nil, spellString("snapshot/"+name)
	}
	data, digest, err := canonical.Identity(&pb.ArtifactReceipt{
		OwnerAuthorityScope: scope, RequestId: request, InvocationSpecDigest: spec,
		OutputSlot: name, ArtifactTransactionId: transaction,
		TensorfsReceiptDigest: nestedDigest, TensorfsReceiptCanonicalBytes: nested,
	})
	if err != nil {
		panic("the fake worker cannot mint an artifact receipt: " + err.Error())
	}
	if plant.plantKey != "" {
		// Written BY HAND because the schema cannot express it: a path, a root, or any
		// other local fact has no slot in this document, which is the point of the arm.
		doc, rerr := canonical.Read(data, &pb.ArtifactReceipt{})
		if rerr == nil {
			raw := map[string]canonical.Value(doc)
			raw[plant.plantKey] = plant.plantValue
			if planted, werr := canonical.Write(raw); werr == nil {
				data, digest = planted, canonical.Digest(planted)
			}
		}
	}
	spelled, _ := canonical.Spell(digest)
	if plant.breakDigest {
		digest = canonical.Digest([]byte("not the receipt"))
	}
	if plant == (receiptPlant{}) {
		f.artMu.Lock()
		if f.artMinted == nil {
			f.artMinted = map[string]*pb.ArtifactReceiptRef{}
		}
		f.artMinted[request+"\x00"+name] = &pb.ArtifactReceiptRef{
			ArtifactReceiptDigest: digest, ArtifactReceiptCanonicalBytes: data,
		}
		f.artMu.Unlock()
		f.say("minted ArtifactReceipt %s for slot %s (%d B, txn %s)", shortSHA(spelled), name,
			len(data), shortSHA(transaction))
	}
	return &pb.ArtifactReceiptRef{
		ArtifactReceiptDigest: digest, ArtifactReceiptCanonicalBytes: data,
	}
}

func (f *fakeControl) mintedRef(requestID, slot string) *pb.ArtifactReceiptRef {
	f.artMu.Lock()
	defer f.artMu.Unlock()
	return f.artMinted[requestID+"\x00"+slot]
}

// artifactOutcome builds one AttemptOutcomeBody/3 carrying the given receipt refs. The
// refs are sent in the order given: an arm that wants the strict-ordering fence to fire
// hands them over reversed.
func artifactOutcome(offer *pb.AttemptOffer, status pb.OutcomeStatus, message string,
	refs []*pb.ArtifactReceiptRef) *pb.AttemptOutcome {
	spelled, _ := canonical.Spell(offer.InvocationSpecDigest)
	cause := &pb.OutcomeCause{Code: pb.CauseCode_CAUSE_CODE_UNSPECIFIED,
		Origin: pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME, Detail: "a fake worker's artifact outcome"}
	if status != pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED {
		// AUTHOR_EXCEPTION is deliberate: it is FINAL, so the artifact transaction has to be
		// closed rather than left open behind a requeue.
		cause = &pb.OutcomeCause{Code: pb.CauseCode_CAUSE_CODE_AUTHOR_EXCEPTION,
			Origin: pb.CauseOrigin_CAUSE_ORIGIN_AUTHOR, Detail: "a fake worker's artifact outcome"}
	}
	body := &pb.AttemptOutcomeBody{
		RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
		InvocationSpecDigest: spelled, Status: status, SafeMessage: message,
		ExecutionStarted: true, Cause: cause, ArtifactReceipts: refs,
	}
	data, digest, err := canonical.Identity(body)
	if err != nil {
		panic("the fake worker cannot mint an artifact outcome: " + err.Error())
	}
	return &pb.AttemptOutcome{
		RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
		InvocationSpecDigest: offer.InvocationSpecDigest, OutcomeId: "out-" + randomHex(8),
		OutcomeDigest: digest, OutcomeCanonicalBytes: data,
	}
}

// artifactAttempt is the whole worker side of one artifact job, per arm.
func (f *fakeControl) artifactAttempt(emit func(*pb.AttemptOutcome), offer *pb.AttemptOffer) {
	slots := artifactSlots(offer)
	spelled, _ := canonical.Spell(offer.InvocationSpecDigest)
	f.say("artifact attempt %s#%d declares %d snapshot slot(s): %s", offer.RequestId,
		offer.AttemptOrdinal, len(slots), strings.Join(slots, ","))
	honest := func() []*pb.ArtifactReceiptRef {
		refs := make([]*pb.ArtifactReceiptRef, 0, len(slots))
		for _, slot := range slots {
			refs = append(refs, f.receiptRef(offer.RequestId, spelled, slot, receiptPlant{}))
		}
		return refs
	}
	switch f.arm {
	case "artifactuncommitted":
		// A FINAL failure with NO receipt: the transaction never committed, so there is
		// nothing to abandon but the open slot itself.
		emit(artifactOutcome(offer, pb.OutcomeStatus_OUTCOME_STATUS_FAILED,
			"the artifact transaction never committed", nil))
	case "artifactabandon":
		// COMMITTED, then failed. The receipt is real and the job's answer is not: the
		// owner must abandon the committed root WITH its receipt as evidence.
		emit(artifactOutcome(offer, pb.OutcomeStatus_OUTCOME_STATUS_FAILED,
			"the artifact committed and the job then failed", honest()))
	case "artifactred":
		f.artifactRedOutcomes(emit, offer, slots, spelled)
	default:
		emit(artifactOutcome(offer, pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED,
			"artifact transaction committed", honest()))
	}
}

// artifactRedOutcomes is the receipt refusal matrix, sent in order against ONE open
// attempt. Only the LAST outcome is admissible, and the ack that follows it is the
// RecordOwner saying so.
func (f *fakeControl) artifactRedOutcomes(emit func(*pb.AttemptOutcome), offer *pb.AttemptOffer,
	slots []string, spelled string) {
	slot := "model"
	if len(slots) > 0 {
		slot = slots[0]
	}
	one := func(plant receiptPlant) []*pb.ArtifactReceiptRef {
		return []*pb.ArtifactReceiptRef{f.receiptRef(offer.RequestId, spelled, slot, plant)}
	}
	send := func(why string, status pb.OutcomeStatus, refs []*pb.ArtifactReceiptRef,
		mutate func(map[string]canonical.Value)) {
		t := artifactOutcome(offer, status, why, refs)
		if mutate != nil {
			doc, err := canonical.Read(t.OutcomeCanonicalBytes, &pb.AttemptOutcomeBody{})
			if err == nil {
				raw := map[string]canonical.Value(doc)
				mutate(raw)
				if planted, werr := canonical.Write(raw); werr == nil {
					t.OutcomeCanonicalBytes, t.OutcomeDigest = planted, canonical.Digest(planted)
				}
			}
		}
		f.say("ARM: %s", why)
		emit(t)
		time.Sleep(350 * time.Millisecond)
	}
	succeeded := pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED

	send("the receipt digest does not hash the carried bytes", succeeded,
		one(receiptPlant{breakDigest: true}), nil)
	send("the nested tensorfs digest does not hash the nested bytes", succeeded,
		one(receiptPlant{breakNested: true}), nil)
	send("a BARE SNAPSHOT ID stands where the nested tensorfs receipt belongs", succeeded,
		one(receiptPlant{bareSnapshot: true}), nil)
	send("the receipt carries a local PATH the closed document has no slot for", succeeded,
		one(receiptPlant{plantKey: "path", plantValue: "/tmp/scratch/model"}), nil)
	send("the receipt names a JOB-AUTHORED authority, not this record owner", succeeded,
		one(receiptPlant{ownerScope: "job/self-declared"}), nil)
	send("the receipt names ANOTHER JOB", succeeded,
		one(receiptPlant{requestID: "job-somebody-else"}), nil)
	send("the receipt names another InvocationSpec", succeeded,
		one(receiptPlant{specDigest: spellString("another invocation")}), nil)
	send("the receipt names an UNDECLARED output slot", succeeded,
		one(receiptPlant{slot: "zzz-undeclared"}), nil)
	send("two receipts arrive out of strict slot order", succeeded, []*pb.ArtifactReceiptRef{
		f.receiptRef(offer.RequestId, spelled, "zzz-second", receiptPlant{slot: "zzz-second"}),
		f.receiptRef(offer.RequestId, spelled, slot, receiptPlant{}),
	}, nil)
	send("a SUCCEEDED artifact job returns no receipt for its declared slot", succeeded, nil, nil)
	send("an artifact-only job ALSO returns ordinary output-manifest entries", succeeded,
		one(receiptPlant{}), func(raw map[string]canonical.Value) {
			raw["output_manifest"] = map[string]canonical.Value{
				"publication_receipt_digest": spellString("an asset publication"),
				"outputs": []canonical.Value{map[string]canonical.Value{
					"output_id": slot, "digest": spellString("asset"),
					"length": int64(3), "mime_type": "image/png",
				}},
			}
		})

	// The ONE admissible outcome, after eleven refusals against the same open attempt.
	f.say("ARM: the admissible artifact outcome")
	emit(artifactOutcome(offer, succeeded, "artifact transaction committed",
		one(receiptPlant{})))
}

// ------------------------------------------------------- the finalize exchange (th-049)

// onFinalizeRequest is the worker half of the ONE durable artifact exchange. A real
// runtime validates the decision, drives TensorFS's atomic root disposition, and answers
// with the exact result; this one validates identically and answers from its own journal.
func (f *fakeControl) onFinalizeRequest(send func(*pb.WorkerFrame),
	env func(func(uint64, uint64, string)), r *pb.ArtifactFinalizeRequest) {
	if !bytes.Equal(canonical.Digest(r.DecisionCanonicalBytes), r.DecisionDigest) {
		f.say("ArtifactFinalizeRequest %s/%s REFUSED: the decision digest does not hash its bytes",
			r.RequestId, r.OutputSlot)
		return
	}
	doc, err := canonical.Read(r.DecisionCanonicalBytes, &pb.ArtifactFinalizeDecision{})
	if err != nil {
		f.say("ArtifactFinalizeRequest %s/%s REFUSED: inadmissible decision (%v)",
			r.RequestId, r.OutputSlot, err)
		return
	}
	spelledSpec, _ := canonical.Spell(r.InvocationSpecDigest)
	if doc.Str("request_id") != r.RequestId || doc.Str("output_slot") != r.OutputSlot ||
		doc.Str("invocation_spec_digest") != spelledSpec {
		f.say("ArtifactFinalizeRequest %s/%s REFUSED: envelope/document divergence",
			r.RequestId, r.OutputSlot)
		return
	}
	disposition := pb.ArtifactFinalizeDisposition(doc.Int("disposition"))
	f.say("ArtifactFinalizeRequest %s/%s %s root=%s receipt=%s", r.RequestId, r.OutputSlot,
		pb.ArtifactFinalizeDisposition_name[int32(disposition)], doc.Str("scratch_root_id"),
		shortSHA(doc.Str("artifact_receipt_digest")))

	outcome := pb.ArtifactFinalizeOutcome_ARTIFACT_FINALIZE_OUTCOME_ABANDONED
	if disposition == pb.ArtifactFinalizeDisposition_ARTIFACT_FINALIZE_DISPOSITION_ADOPT {
		outcome = pb.ArtifactFinalizeOutcome_ARTIFACT_FINALIZE_OUTCOME_ADOPTED
	}
	// The receipt is EVIDENCE, returned only where one was committed. An uncommitted
	// abandonment has nothing to return, and saying otherwise would be inventing one.
	ref := f.mintedRef(r.RequestId, r.OutputSlot)
	if disposition == pb.ArtifactFinalizeDisposition_ARTIFACT_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED {
		ref = nil
	}
	exact := f.resultFrame(r, spelledSpec, outcome, ref, resultPlant{})

	if f.arm == "artifacthalf" {
		// The crash arm: exactly ONE slot is answered. The other stays owed, which is how
		// an owner can be killed with one exact result durable and one still in flight.
		f.artMu.Lock()
		f.artAnswered++
		answered := f.artAnswered
		f.artMu.Unlock()
		if answered > 1 {
			f.say("ARM: the finalize request for %s/%s is HELD", r.RequestId, r.OutputSlot)
			return
		}
	}

	if f.arm == "artifactresultred" {
		f.artifactRedResults(send, env, r, spelledSpec, outcome, ref, exact)
		return
	}
	f.sendResult(send, env, exact)
	if f.arm == "artifactlostresult" {
		// A LOST RESULT looks like exactly this from the owner's side: the identical
		// frame again, byte for byte, until it is answered.
		time.Sleep(300 * time.Millisecond)
		f.say("ARM: the finalize result is REPLAYED byte for byte")
		f.sendResult(send, env, exact)
	}
}

// resultPlant names one deviation from the exact ArtifactFinalizeResult.
type resultPlant struct {
	slot           string // a slot with no persisted intent
	envelopeSlot   string // the envelope names a slot the document does not
	breakDigest    bool   // the result digest does not hash the carried bytes
	foreignReceipt bool   // a DIFFERENT receipt for the same owner/request/spec/slot
	omitReceipt    bool   // ADOPTED with no receipt at all
	flipOutcome    bool   // the opposite of the persisted intent
}

func (f *fakeControl) resultFrame(r *pb.ArtifactFinalizeRequest, spelledSpec string,
	outcome pb.ArtifactFinalizeOutcome, ref *pb.ArtifactReceiptRef,
	plant resultPlant) *pb.ArtifactFinalizeResultFrame {
	slot := r.OutputSlot
	if plant.slot != "" {
		slot = plant.slot
	}
	if plant.flipOutcome {
		if outcome == pb.ArtifactFinalizeOutcome_ARTIFACT_FINALIZE_OUTCOME_ADOPTED {
			outcome = pb.ArtifactFinalizeOutcome_ARTIFACT_FINALIZE_OUTCOME_ABANDONED
		} else {
			outcome = pb.ArtifactFinalizeOutcome_ARTIFACT_FINALIZE_OUTCOME_ADOPTED
		}
	}
	if plant.foreignReceipt {
		ref = f.receiptRef(r.RequestId, spelledSpec, slot,
			receiptPlant{transaction: spellString("a second transaction for the same slot")})
	}
	if plant.omitReceipt {
		ref = nil
	}
	data, digest, err := canonical.Identity(&pb.ArtifactFinalizeResult{
		OwnerAuthorityScope: f.ownerScope, RequestId: r.RequestId,
		InvocationSpecDigest: spelledSpec, OutputSlot: slot,
		Outcome: outcome, ArtifactReceipt: ref,
	})
	if err != nil {
		// An inadmissible document is a legitimate arm: send the bytes the writer refused
		// to fence and let the owner's reader be the one that says no.
		f.say("the finalize result document could not be minted: %v", err)
		return nil
	}
	if plant.breakDigest {
		digest = canonical.Digest([]byte("not the result"))
	}
	envelopeSlot := slot
	if plant.envelopeSlot != "" {
		envelopeSlot = plant.envelopeSlot
	}
	return &pb.ArtifactFinalizeResultFrame{
		RequestId: r.RequestId, InvocationSpecDigest: r.InvocationSpecDigest,
		OutputSlot: envelopeSlot, ResultDigest: digest, ResultCanonicalBytes: data,
	}
}

func (f *fakeControl) sendResult(send func(*pb.WorkerFrame),
	env func(func(uint64, uint64, string)), frame *pb.ArtifactFinalizeResultFrame) {
	if frame == nil {
		return
	}
	env(func(e, g uint64, b string) {
		frame.RecordOwnerEpoch, frame.ControlStreamGeneration, frame.WorkerBootId = e, g, b
	})
	send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ArtifactFinalizeResult{ArtifactFinalizeResult: frame}})
}

// artifactRedResults is the finalize-result refusal matrix over ONE persisted intent:
// eight inadmissible answers, then the exact one, then a byte-identical replay of it, then
// a CHANGED answer for a transaction that is already complete.
func (f *fakeControl) artifactRedResults(send func(*pb.WorkerFrame),
	env func(func(uint64, uint64, string)), r *pb.ArtifactFinalizeRequest, spelledSpec string,
	outcome pb.ArtifactFinalizeOutcome, ref *pb.ArtifactReceiptRef,
	exact *pb.ArtifactFinalizeResultFrame) {
	step := func(why string, plant resultPlant) {
		f.say("ARM: %s", why)
		f.sendResult(send, env, f.resultFrame(r, spelledSpec, outcome, ref, plant))
		time.Sleep(300 * time.Millisecond)
	}
	step("the result digest does not hash the carried bytes", resultPlant{breakDigest: true})
	step("the result names a slot with NO persisted intent", resultPlant{slot: "zzz-no-intent"})
	step("the envelope names a slot the document does not", resultPlant{envelopeSlot: "zzz-other"})
	step("the result CONTRADICTS the persisted intent", resultPlant{flipOutcome: true})
	step("ADOPTED returns a DIFFERENT receipt than the persisted one",
		resultPlant{foreignReceipt: true})
	step("ADOPTED omits the exact adopted receipt", resultPlant{omitReceipt: true})

	f.say("ARM: the exact result")
	f.sendResult(send, env, exact)
	time.Sleep(400 * time.Millisecond)
	// A LOST RESULT: the identical frame again. Nothing may be applied twice.
	f.say("ARM: the exact result REPLAYED byte for byte")
	f.sendResult(send, env, exact)
	time.Sleep(400 * time.Millisecond)
	// A CHANGED result for a completed transaction conflicts forever.
	step("a CHANGED result arrives after the transaction completed",
		resultPlant{foreignReceipt: true})
}
