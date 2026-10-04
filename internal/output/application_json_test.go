package output

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestJSONRenderingPreservesApplicationNumbersWithoutTOONRoundTrip(t *testing.T) {
	var buffer bytes.Buffer
	value := map[string]any{"seed": uint64(18446744073709551615), "number": json.Number("1.0"), "ordered": []uint64{9007199254740993, 9007199254740992}}
	if err := Write(&buffer, value, Mode{JSON: true}); err != nil {
		t.Fatal(err)
	}
	for _, exact := range []string{`"seed":18446744073709551615`, `"number":1.0`, `"ordered":[9007199254740993,9007199254740992]`} {
		if !strings.Contains(buffer.String(), exact) {
			t.Fatalf("authored application value changed: %s", buffer.String())
		}
	}
}

func TestDefaultRenderingFallsBackWhenTOONWouldRoundAuthoredValues(t *testing.T) {
	var buffer bytes.Buffer
	value := map[string]any{"seed": uint64(18446744073709551615), "number": json.Number("1.0")}
	if err := Write(&buffer, value, Mode{}); err != nil {
		t.Fatal(err)
	}
	var got struct {
		Seed   uint64
		Number json.Number
	}
	if err := json.Unmarshal(buffer.Bytes(), &got); err != nil || got.Seed != ^uint64(0) || got.Number.String() != "1.0" {
		t.Fatalf("default output rounded authored values: %s (%v)", buffer.String(), err)
	}
}
