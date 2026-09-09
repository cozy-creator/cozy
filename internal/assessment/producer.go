// Package assessment verifies report association against Creator's retained work.
// cozy-eval owns report semantics; this package compares only its validated facts.
package assessment

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

const MaxBytes = 8 << 20

type Inspection struct {
	Schema string `json:"schema"`
	Report struct {
		Digest string `json:"digest"`
		Length int64  `json:"length"`
		Schema string `json:"schema"`
	} `json:"report"`
	Subject struct {
		Type       string `json:"type"`
		Candidate  string `json:"candidate_checkpoint"`
		Reference  string `json:"reference_checkpoint"`
		Invocation string `json:"producer_invocation"`
		Build      string `json:"producer_build"`
		Receipt    string `json:"weights_receipt"`
	} `json:"subject"`
	OutputSlot string          `json:"output_slot"`
	Verdict    string          `json:"publisher_reported_verdict"`
	Weights    json.RawMessage `json:"weights"`
}

func VerifyProducer(st *records.Store, requestID string, report []byte, inspected []byte) (hub.Ref, Inspection, *exit.Error) {
	var info Inspection
	refuse := func() (hub.Ref, Inspection, *exit.Error) {
		return hub.Ref{}, info, exit.Named(exit.Conflict, "assessment.producer_binding_mismatch", "the report does not match this Creator's retained producer evidence")
	}
	if len(report) == 0 || len(report) > MaxBytes || len(inspected) > MaxBytes || json.Unmarshal(inspected, &info) != nil {
		return refuse()
	}
	digest, err := canonical.Spell(canonical.Digest(report))
	if err != nil || info.Schema != "cozy-eval/report-inspection@2" || info.Report.Schema != "cozy-eval/checkpoint-validation@2" || info.Report.Digest != digest || info.Report.Length != int64(len(report)) || info.Subject.Type != "producer" || info.Verdict != "indeterminate" {
		return refuse()
	}
	request, problem := st.RequestRow(requestID)
	if problem != nil {
		return hub.Ref{}, info, problem
	}
	if request == nil || request.Kind != "job" || request.ModelTransfer == nil || request.ModelTransfer.Kind != "model-upload" {
		return refuse()
	}
	ref, problem := hub.ParseRef(request.ModelTransfer.Destination)
	if problem != nil {
		return hub.Ref{}, info, problem
	}
	attempt, problem := st.AttemptRow(requestID, request.Ordinal)
	if problem != nil {
		return hub.Ref{}, info, problem
	}
	if attempt == nil || attempt.InvocationDigest != info.Subject.Invocation || len(attempt.TerminalBody) == 0 {
		return refuse()
	}
	invocation, err := canonical.Read(attempt.InvocationCanonical, &pb.InvocationSpec{})
	if err != nil || invocation.Sub("job").Str("build_id") != info.Subject.Build {
		return refuse()
	}
	actual, err := canonical.Spell(canonical.Digest(attempt.InvocationCanonical))
	if err != nil || actual != info.Subject.Invocation {
		return refuse()
	}
	source := false
	for _, input := range invocation.List("inputs") {
		source = source || (strings.HasPrefix(input.Str("input_id"), "model:") &&
			input.Str("kind_mime") == "application/vnd.cozy.model-manifest" && input.Str("digest") == info.Subject.Reference)
	}
	if !source {
		return refuse()
	}
	outputs, problem := st.AllModelTransferWeights(requestID, request.Ordinal)
	if problem != nil {
		return hub.Ref{}, info, problem
	}
	matched := false
	for _, output := range outputs {
		if output.OutputSlot != info.OutputSlot {
			continue
		}
		receipt, err := canonical.Read(output.Receipt, &pb.WeightsReceipt{})
		receiptDigest, spellErr := canonical.Spell(canonical.Digest(output.Receipt))
		if err != nil || spellErr != nil || output.FinalID == "" || output.ManifestID != info.Subject.Candidate || output.ReceiptDigest != info.Subject.Receipt || receiptDigest != info.Subject.Receipt || receipt.Str("invocation_spec_digest") != info.Subject.Invocation || receipt.Str("request_id") != requestID || receipt.Str("output_slot") != info.OutputSlot {
			return refuse()
		}
		matched = true
	}
	if !matched {
		return refuse()
	}
	outcome, err := canonical.Read(attempt.TerminalBody, &pb.AttemptOutcomeBody{})
	if err != nil {
		return refuse()
	}
	outcomeDigest, err := canonical.Spell(canonical.Digest(attempt.TerminalBody))
	if err != nil || outcomeDigest != attempt.TerminalDigest {
		return refuse()
	}
	var result struct {
		Weights []json.RawMessage `json:"weight_fidelity_this_run"`
	}
	resultBytes, err := base64.StdEncoding.Strict().DecodeString(outcome.Sub("result").Str("inline_result"))
	if err != nil || json.Unmarshal(resultBytes, &result) != nil {
		return refuse()
	}
	expected := []json.RawMessage{}
	for _, row := range result.Weights {
		var label struct {
			Output string `json:"output_slot"`
		}
		if json.Unmarshal(row, &label) != nil {
			return refuse()
		}
		if label.Output == info.OutputSlot {
			expected = append(expected, row)
		}
	}
	raw, _ := json.Marshal(expected)
	left, err := canonical.NormalizeJCS(raw)
	if err != nil {
		return refuse()
	}
	right, err := canonical.NormalizeJCS(info.Weights)
	if err != nil || !bytes.Equal(left, right) {
		return refuse()
	}
	return ref, info, nil
}
