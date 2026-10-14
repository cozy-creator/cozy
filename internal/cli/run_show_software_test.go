package cli

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/output"
)

func TestRunSoftwareUsesAttemptReceiptAndKeepsChildIdentity(t *testing.T) {
	life := api.Lifecycle{RequestID: "root", Status: "completed", Number: 1, CreatedAt: "2026-10-13T00:00:00Z"}
	evidence := api.Evidence{Triage: json.RawMessage(`{"measurements":{"software":{"attempt_ordinal":2,"tensord":"0.5.26","tensord_commit":"123456789abcdef","runtime":"0.23.30","tensorfs":"0.6.1"}}}`),
		Events: []api.EvidenceEvent{
			{Type: "machine.executor", Payload: json.RawMessage(`{"runtime_version":"unrelated-latest-sdk"}`)},
			{Type: "machine.call", At: "2026-10-13T00:00:01Z", Payload: json.RawMessage(`{"request":"root/0","export":"compute","status":"succeeded","attempt":1,"software":{"attempt_ordinal":3,"tensord":"0.5.25","runtime":"package-fallback-sdk"}}`)},
			{Type: "machine.executor", Payload: json.RawMessage(`{"request":"root/0","runtime_version":"unrelated-latest-sdk"}`)},
		}}
	report := buildRunReport(life, evidence)
	if report.Software == nil || report.Software.AttemptOrdinal != 2 || report.Software.TensorDCommit != "123456789abcdef" || report.Runtime != "cozy-runtime 0.23.30 · tensorfs 0.6.1" {
		t.Fatalf("wrong root software: %+v", report)
	}
	if len(report.Calls) != 1 || report.Calls[0].Software == nil || report.Calls[0].Software.AttemptOrdinal != 3 || report.Calls[0].Attempt != 3 || report.Calls[0].Runtime != "cozy-runtime package-fallback-sdk" {
		t.Fatalf("child software was replaced by its parent's: %+v", report.Calls)
	}
	var human bytes.Buffer
	if err := report.Emit(&human, output.Mode{Human: true}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(human.String(), "TensorD 0.5.26 (123456789ab") || !strings.Contains(human.String(), "cozy-runtime 0.23.30") || strings.Contains(human.String(), "unrelated-latest-sdk") {
		t.Fatalf("human report lost captured software: %s", human.String())
	}
	var encoded bytes.Buffer
	if err := report.Emit(&encoded, output.Mode{JSON: true}); err != nil {
		t.Fatal(err)
	}
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(encoded.Bytes(), &doc); err != nil || len(doc["software"]) == 0 {
		t.Fatalf("JSON software absent: %s", encoded.String())
	}
}

func TestLegacyOrUnfamiliarSoftwareNeverInventsVersionsOrHidesStages(t *testing.T) {
	for _, software := range []string{`null`, `{"runtime":{"new_shape":true}}`, `{}`} {
		report := buildRunReport(api.Lifecycle{RequestID: "root", Status: "completed"}, api.Evidence{
			Triage: json.RawMessage(`{"measurements":{"software":` + software + `,"attribution":{"stages":{"encode":{"total_ms":12,"count":1}}}}}`),
		})
		if report.Software != nil || report.Runtime != "" || len(report.Stages) != 1 {
			t.Fatalf("legacy/unfamiliar evidence was changed: %+v", report)
		}
	}
}
