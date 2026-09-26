package producttest

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/assessment"
	"github.com/cozy-creator/cozy/internal/records"
)

var assessmentProofHome = flag.String("assessment-proof-home", "", "private retained completed ordinary-job proof home")
var assessmentProofReport = flag.String("assessment-proof-report", "", "actual canonical producer report for that job")
var assessmentProofInspection = flag.String("assessment-proof-inspection", "", "actual public evaluator inspection of the same report")
var assessmentProofRequest = flag.String("assessment-proof-request", "", "exact retained producer request id")

func TestProducerAssessmentBindsRetainedInvocationResultAndReceipt(t *testing.T) {
	if *assessmentProofHome == "" || *assessmentProofReport == "" || *assessmentProofInspection == "" || *assessmentProofRequest == "" {
		t.Skip("requires actual completed ordinary job evidence and public canonical inspection")
	}
	st, problem := records.Open(filepath.Join(*assessmentProofHome, "creator.sqlite"))
	fatal(t, problem)
	defer st.Close()
	raw, err := os.ReadFile(*assessmentProofReport)
	must(t, err)
	inspected, err := os.ReadFile(*assessmentProofInspection)
	must(t, err)
	_, valid, problem := assessment.VerifyProducer(st, *assessmentProofRequest, raw, inspected)
	fatal(t, problem)
	for _, arm := range []struct {
		name   string
		change func(*assessment.Inspection)
	}{
		{"candidate", func(i *assessment.Inspection) { i.Subject.Candidate = "sha256:" + strings.Repeat("9", 64) }},
		{"reference", func(i *assessment.Inspection) { i.Subject.Reference = "sha256:" + strings.Repeat("9", 64) }},
		{"invocation", func(i *assessment.Inspection) { i.Subject.Invocation = "sha256:" + strings.Repeat("9", 64) }},
		{"installation", func(i *assessment.Inspection) { i.Subject.Installation = "other-installation" }},
		{"receipt", func(i *assessment.Inspection) { i.Subject.Receipt = "sha256:" + strings.Repeat("9", 64) }},
		{"wrong output", func(i *assessment.Inspection) { i.OutputSlot = "other" }},
		{"false weights", func(i *assessment.Inspection) {
			i.Weights = json.RawMessage(`[{"output_slot":"model","component":"model","stats":{"encoded_keys":1}}]`)
		}},
	} {
		t.Run(arm.name, func(t *testing.T) {
			changed := valid
			arm.change(&changed)
			body, err := json.Marshal(changed)
			must(t, err)
			if _, _, problem := assessment.VerifyProducer(st, *assessmentProofRequest, raw, body); problem == nil {
				t.Fatal("unrelated evidence was admitted")
			}
		})
	}
	t.Log("actual retained producer invocation/result/receipt matches; all identity and measurement substitutions refuse")
}
