package canonical

import (
	"bytes"
	"strings"
	"testing"
)

func TestApplicationJSONRetainsIntegerPrecisionAndNumberKind(t *testing.T) {
	first := []byte(`{ "seed": 18446744073709551615, "number": 1e0, "ordered": [9007199254740993, 9007199254740992] }`)
	got, err := NormalizeApplication(first)
	want := `{"number":1.0,"ordered":[9007199254740993,9007199254740992],"seed":18446744073709551615}`
	if err != nil || string(got) != want {
		t.Fatalf("%s: %v", got, err)
	}
	for _, changed := range []string{
		strings.Replace(want, "18446744073709551615", "18446744073709551614", 1),
		strings.Replace(want, `"number":1.0`, `"number":1`, 1),
		strings.Replace(want, "[9007199254740993,9007199254740992]", "[9007199254740992,9007199254740993]", 1),
	} {
		other, err := NormalizeApplication([]byte(changed))
		if err != nil || bytes.Equal(got, other) {
			t.Fatalf("different authored semantics merged: %s (%v)", changed, err)
		}
	}
	if _, err := NormalizeJCS(first); err == nil {
		t.Fatal("application entry point weakened the bounded JCS protocol profile")
	}
	if protocol, err := NormalizeJCS([]byte(`{"number":1.0}`)); err != nil || string(protocol) != `{"number":1}` {
		t.Fatalf("JCS float spelling changed: %s %v", protocol, err)
	}
}

func TestApplicationJSONKeepsBoundedStrictParsing(t *testing.T) {
	for _, invalid := range []string{`{"a":1,"a":2}`, `{"a":"\ud800"}`, `{"a":1} {}`, `{"a":1e999}`, strings.Repeat("[", 34) + "0" + strings.Repeat("]", 34)} {
		if _, err := NormalizeApplication([]byte(invalid)); err == nil {
			t.Fatalf("invalid application JSON admitted: %s", invalid)
		}
	}
}
