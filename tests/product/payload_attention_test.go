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

func TestAttentionOverrideSyntaxPreservesScopedBackends(t *testing.T) {
	ep := &launch.Entrypoint{Name: "generate", Request: launch.Struct{Fields: []launch.Field{{Name: "prompt", Type: json.RawMessage(`"str"`), Wire: "required"}}}}
	for _, pin := range []string{"kitchen-int8", "fl2va_dit=kitchen-int8", "model/fl2va_dit=flashinfer-bf16-fp8", "base_model/ref2va_dit=sol-attn"} {
		payload, keys, problem := launch.ParsePayload(ep, []string{"prompt=hello", "kernel.attention=" + pin}, "")
		if problem != nil || keys.AttentionKernel != pin || strings.Contains(string(payload), "attention") {
			t.Fatalf("pin %q: keys=%+v payload=%s problem=%v", pin, keys, payload, problem)
		}
	}
	for _, pin := range []string{"=kitchen-int8", "fl2va_dit=", "fl2va_dit=kitchen-int8=extra", "fl2va_dit =kitchen-int8", " kitchen-int8", "kitchen-int8\u2003"} {
		_, _, problem := launch.ParsePayload(ep, []string{"prompt=hello", "kernel.attention=" + pin}, "")
		if problem == nil || problem.ErrName() != "attention_override_invalid" {
			t.Fatalf("invalid pin %q accepted: %v", pin, problem)
		}
	}
}
