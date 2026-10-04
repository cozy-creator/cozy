package cli

import (
	"encoding/json"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/output"
)

func TestHumanSavedResultRetainsExactApplicationScalarFields(t *testing.T) {
	var life api.Lifecycle
	if err := json.Unmarshal([]byte(`{"result":{"seed":18446744073709551615,"number":1.0,"asset":{"digest":"hidden"}}}`), &life); err != nil {
		t.Fatal(err)
	}
	fields := expandSavedResult([]output.Field{{K: "saved", V: "file.mp4"}, {K: "result", V: life.Result}})
	if len(fields) != 3 || fields[1].K != "number" || fields[1].V != json.Number("1.0") || fields[2].K != "seed" || fields[2].V != json.Number("18446744073709551615") {
		t.Fatalf("human scalar fields rounded or duplicated the saved handle: %+v", fields)
	}
}
