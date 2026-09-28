package producttest

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestNativeAndMixedOutputMediaReservation(t *testing.T) {
	native, _ := json.Marshal([]orchestrator.WeightsOutput{{OutputID: "model", MimeType: orchestrator.WeightsManifestMime, MaxBytes: 200 << 30}})
	for _, test := range []struct {
		outputs string
		weights string
		want    int64
	}{
		{"model", string(native), 0},
		{"model", `[{"output_id":"model","mime_type":"application/vnd.cozy.model-manifest","max_bytes":9007199254740991}]`, 0},
		{"image,model", string(native), 256},
		{"image,video", `[]`, 512},
	} {
		got, problem := orchestrator.MediaReservationBytes(records.Request{Outputs: test.outputs, WeightsOutputs: test.weights}, 256)
		fatal(t, problem)
		if got != test.want {
			t.Fatalf("%s reserved%d want%d", test.outputs, got, test.want)
		}
	}
	if _, problem := orchestrator.MediaReservationBytes(records.Request{Outputs: "a,b", WeightsOutputs: `[]`}, math.MaxInt64); problem == nil {
		t.Fatal("media bound overflow was accepted")
	}
}
