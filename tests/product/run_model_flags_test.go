package producttest

import (
	"bytes"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestDashedModelFlagsReuseExistingResolver(t *testing.T) {
	root, _, _, _, _ := runModelCatalog(t)
	base := []string{"--json", "run", "proof/quantize/quantize", "steps=7", "--rental-only", "--publish-to", "proof/output"}
	canonical := append(append([]string{}, base[:4]...), append([]string{"model.dits=proof/source@1.0.0/bf16", "model.shared=proof/source@1.0.0/bf16"}, base[4:]...)...)
	want, _, out := submitRun(t, root, "model-flags-canonical", canonical...)
	if want == nil {
		t.Fatalf("canonical run was not admitted: %s", out)
	}
	for i, args := range [][]string{
		append(append([]string{}, base...), "--model.dits=proof/source@1.0.0/bf16", "--model.shared=proof/source@1.0.0/bf16"),
		append([]string{"--json", "run", "--model.dits=proof/source@1.0.0/bf16", "--model.shared=proof/source@1.0.0/bf16"}, base[2:]...),
	} {
		got, _, out := submitRun(t, root, fmt.Sprintf("model-flags-dashed-%d", i), args...)
		if got == nil || !reflect.DeepEqual(got.Models, want.Models) || !bytes.Equal(got.Payload, want.Payload) {
			t.Fatalf("dashed override changed resolver meaning: %+v %s", got, out)
		}
	}
	for _, extra := range []string{"--model.other=proof/source@1.0.0/bf16", "--model.dits=proof/source@1.0.0/bf16"} {
		code, out := runCozy(t, root, append(append([]string{}, canonical...), extra)...)
		if code == 0 || strings.Contains(out, "unknown flag") {
			t.Fatalf("unknown/duplicate model slot bypassed the existing resolver: %d %s", code, out)
		}
	}
	for _, args := range [][]string{
		{"--json", "model", "info", "proof/source", "--model.other=bad"},
		{"--json", "--fields", "run", "model", "info", "proof/source", "--model.other=bad"},
		{"--json", "run", "watch", "1", "--model.other=bad"},
		{"--json", "run", "proof/quantize/quantize", "--describe", "--stream"},
	} {
		code, out := runCozy(t, root, args...)
		if code == 0 || !strings.Contains(out, "unknown flag") {
			t.Fatalf("unsupported flag was reinterpreted: %d %s", code, out)
		}
	}
}
