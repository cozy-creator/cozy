package cli

import (
	"bytes"
	"reflect"
	"strings"
	"testing"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/output"
)

func TestProgressLineKeepsOnlyUsefulHumanStatus(t *testing.T) {
	tests := []struct {
		name  string
		event localapi.Event
		want  string
	}{
		{
			name:  "runtime fraction",
			event: localapi.Event{Type: "request.progress", Payload: map[string]any{"value": 0.35}},
			want:  "  running — 35%",
		},
		{
			name: "named fraction",
			event: localapi.Event{Type: "request.progress", Payload: map[string]any{
				"value": map[string]any{"name": "denoising", "fraction": 0.6},
			}},
			want: "  denoising — 60%",
		},
		{
			name: "named stage",
			event: localapi.Event{Type: "request.stage", Payload: map[string]any{
				"value": map[string]any{"name": "decoding", "value": 82.4},
			}},
			want: "  decoding",
		},
		{
			name:  "unlabeled stage timing",
			event: localapi.Event{Type: "request.stage", Payload: map[string]any{"value": 80.941}},
		},
		{
			name:  "metric",
			event: localapi.Event{Type: "request.metric", Payload: map[string]any{"value": 126.25}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := progressLine(test.event, false); got != test.want {
				t.Fatalf("progressLine() = %q, want %q", got, test.want)
			}
		})
	}
}

func TestProgressLineFullPreservesDiagnosticPayload(t *testing.T) {
	event := localapi.Event{Type: "request.stage", Payload: map[string]any{"value": 80.941}}
	if got, want := progressLine(event, true), "  stage value=80.941"; got != want {
		t.Fatalf("progressLine(full) = %q, want %q", got, want)
	}
}

func TestRunProgressSuppressesLossyFramesWhenRedirected(t *testing.T) {
	var stderr bytes.Buffer
	progress := newProgress(&Context{
		Inv: &Invocation{Mode: output.Mode{Human: true}}, Err: &stderr,
	}, false)
	progress.on(localapi.Event{
		Type: "request.progress", Payload: map[string]any{"value": 0.5},
	})
	progress.done()
	if stderr.Len() != 0 {
		t.Fatalf("redirected progress = %q, want no lossy output", stderr.String())
	}
}

func TestRunProgressFullRedirectHasNoTerminalControlCodes(t *testing.T) {
	var stderr bytes.Buffer
	progress := newProgress(&Context{
		Inv: &Invocation{Mode: output.Mode{Human: true, Full: true}}, Err: &stderr,
	}, false)
	progress.on(localapi.Event{
		Type: "request.metric", Payload: map[string]any{"value": 126.25},
	})
	progress.done()
	if got, want := stderr.String(), "  metric value=126.25\n"; got != want {
		t.Fatalf("redirected full progress = %q, want %q", got, want)
	}
	if strings.Contains(stderr.String(), "\x1b") || strings.Contains(stderr.String(), "\r") {
		t.Fatalf("redirected full progress contains terminal controls: %q", stderr.String())
	}
}

func TestExpandSavedResultKeepsUsefulScalars(t *testing.T) {
	fields := expandSavedResult([]output.Field{
		{K: "target", V: "paul/sdxl/generate"},
		{K: "result", V: map[string]any{
			"image":               map[string]any{"asset_ref": "attempt:req/image/1"},
			"digest":              "0123456789abcdef",
			"width":               float64(1024),
			"height":              float64(1024),
			"steps":               float64(20),
			"guidance":            7.0,
			"hidiffusion_applied": true,
			"target":              "result-owned-target",
		}},
		{K: "saved", V: []string{"image.png (2.4 MiB)"}},
	})
	want := []output.Field{
		{K: "target", V: "paul/sdxl/generate"},
		{K: "guidance", V: 7.0},
		{K: "height", V: float64(1024)},
		{K: "hidiffusion_applied", V: true},
		{K: "steps", V: float64(20)},
		{K: "width", V: float64(1024)},
		{K: "saved", V: []string{"image.png (2.4 MiB)"}},
	}
	if got := len(fields); got != len(want) {
		t.Fatalf("field count = %d, want %d: %#v", got, len(want), fields)
	}
	for i := range want {
		if fields[i].K != want[i].K || !reflect.DeepEqual(fields[i].V, want[i].V) {
			t.Fatalf("field[%d] = %#v, want %#v", i, fields[i], want[i])
		}
	}
}
