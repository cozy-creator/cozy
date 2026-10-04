package api

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestApplicationResultStateReadersRetainAuthoredNumbers(t *testing.T) {
	for _, state := range []any{&JobState{}, &Lifecycle{}} {
		first := `{"status":"succeeded","attempt":2,"future_advisory":true,"result":{"seed":18446744073709551615,"number":1.0}}`
		if err := json.Unmarshal([]byte(first), state); err != nil {
			t.Fatal(err)
		}
		out, err := json.Marshal(state)
		if err != nil || !strings.Contains(string(out), `"seed":18446744073709551615`) || !strings.Contains(string(out), `"number":1.0`) || !strings.Contains(string(out), `"attempt":2`) {
			t.Fatalf("state changed authored numbers or typed fields: %s %v", out, err)
		}
		if err := json.Unmarshal([]byte(`{"status":"running"}`), state); err != nil {
			t.Fatal(err)
		}
		out, _ = json.Marshal(state)
		if strings.Contains(string(out), `"result"`) {
			t.Fatalf("reused reader retained a previous result: %s", out)
		}
	}
}
