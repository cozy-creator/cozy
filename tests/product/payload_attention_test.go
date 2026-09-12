package producttest

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
)

func TestParsePayloadAttentionKernelIsReserved(t *testing.T) {
	ep := &launch.Entrypoint{Name: "generate", Request: launch.Struct{Fields: []launch.Field{{Name: "prompt", Type: json.RawMessage(`"str"`), Wire: "required"}}}}
	payload, keys, problem := launch.ParsePayload(ep, []string{"prompt=hello", "kernel.attention=flash-attn3-fp8"}, "")
	if problem != nil {
		t.Fatalf("parse failed: %v", problem)
	}
	if keys.AttentionKernel != "flash-attn3-fp8" {
		t.Fatalf("attention pin = %q", keys.AttentionKernel)
	}
	if strings.Contains(string(payload), "attention") {
		t.Fatalf("execution pin leaked into package payload: %s", payload)
	}
}

func TestParsePayloadRejectsUnknownKernelAxis(t *testing.T) {
	ep := &launch.Entrypoint{Name: "generate", Request: launch.Struct{Fields: []launch.Field{{Name: "prompt", Type: json.RawMessage(`"str"`), Wire: "required"}}}}
	_, _, problem := launch.ParsePayload(ep, []string{"kernel.gemm=foo"}, "")
	if problem == nil || !strings.Contains(problem.Message, "execution-path override") {
		t.Fatalf("unknown axis problem = %v", problem)
	}
}
