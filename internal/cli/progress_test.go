package cli

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
	"time"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/output"
)

// fixedClock advances a deterministic 200ms per reading, so two identical drives of the
// renderer produce identical bytes.
func fixedClock(start time.Time) func() time.Time {
	tick := 0
	return func() time.Time {
		tick++
		return start.Add(time.Duration(tick) * 200 * time.Millisecond)
	}
}

func progressContext(mode output.Mode) (*Context, *bytes.Buffer) {
	buf := &bytes.Buffer{}
	return &Context{Inv: &Invocation{Mode: mode}, Err: buf, Out: &bytes.Buffer{}}, buf
}

func stepEvent(stage string, step, total int) localapi.Event {
	return localapi.Event{Type: "request.progress", Payload: map[string]any{
		"value": map[string]any{
			"name": stage, "fraction": float64(step) / float64(total),
			"position": step, "step_ms": 500.0,
		},
	}}
}

func denoiseRun(total int) []localapi.Event {
	events := []localapi.Event{
		{Type: "request.dispatched", Payload: map[string]any{}},
		{Type: "request.accepted", Payload: map[string]any{}},
	}
	for step := 1; step <= total; step++ {
		events = append(events, stepEvent("denoise", step, total))
	}
	return events
}

func drive(p *runProgress, events []localapi.Event) {
	for _, e := range events {
		p.on(e)
	}
	p.done()
}

// TestProgressTTYRewritesOneLine: the terminal arm is ONE in-place line — every write is
// a \r + erase rewrite carrying steps, percentage, and elapsed; the only newline is the
// final cursor release.
func TestProgressTTYRewritesOneLine(t *testing.T) {
	began := time.Unix(1700000000, 0)
	ctx, buf := progressContext(output.Mode{Human: true, Color: true})
	p := newProgress(ctx, false, began)
	p.now = fixedClock(began)
	p.width = func() int { return 120 }
	drive(p, denoiseRun(30))
	got := buf.String()
	if !strings.Contains(got, "\r\033[K") {
		t.Fatalf("TTY arm lost its in-place rewrite\n%q", got)
	}
	if n := strings.Count(got, "\n"); n != 1 || !strings.HasSuffix(got, "\n") {
		t.Fatalf("TTY arm must hold one line and release it once at the end, got %d newlines\n%q", n, got)
	}
	last := got[strings.LastIndex(got, "\r"):]
	for _, want := range []string{"denoising", "30/30", "100%", "s/step", "elapsed"} {
		if !strings.Contains(last, want) {
			t.Errorf("final rewrite omitted %q\n%q", want, last)
		}
	}
	if !strings.Contains(got, "15/30") || !strings.Contains(got, "50%") {
		t.Errorf("intermediate steps were not rewritten in place\n%q", got)
	}
}

// TestProgressTTYClampsToTerminalWidth: a rewritten line never reaches the terminal
// edge, so it cannot wrap and leave rows \r can no longer reach after a resize.
func TestProgressTTYClampsToTerminalWidth(t *testing.T) {
	began := time.Unix(1700000000, 0)
	ctx, buf := progressContext(output.Mode{Human: true, Color: true})
	p := newProgress(ctx, false, began)
	p.now = fixedClock(began)
	width := 24
	p.width = func() int { return width }
	drive(p, denoiseRun(30))
	for _, chunk := range strings.Split(buf.String(), "\r") {
		line := strings.TrimSuffix(strings.TrimPrefix(chunk, "\033[K"), "\n")
		if len([]rune(line)) >= width {
			t.Fatalf("rewrite %q is %d runes wide on a %d-column terminal", line, len([]rune(line)), width)
		}
	}
}

// TestProgressPipedIsSparseAndByteStable: the redirected human lane is append-only —
// plain lines, no control sequences, one line per tenth — and byte-identical across two
// identical drives.
func TestProgressPipedIsSparseAndByteStable(t *testing.T) {
	began := time.Unix(1700000000, 0)
	render := func() string {
		ctx, buf := progressContext(output.Mode{Human: true})
		p := newProgress(ctx, false, began)
		p.now = fixedClock(began)
		drive(p, denoiseRun(30))
		return buf.String()
	}
	first, second := render(), render()
	if first != second {
		t.Fatalf("piped output is not byte-stable\nfirst:\n%q\nsecond:\n%q", first, second)
	}
	if strings.ContainsAny(first, "\r\033") {
		t.Fatalf("piped output carries terminal control bytes\n%q", first)
	}
	lines := strings.Split(strings.TrimSuffix(first, "\n"), "\n")
	// 30 steps in tenths: 10% .. 100%, plus the two status lines.
	if len(lines) < 5 || len(lines) > 13 {
		t.Fatalf("piped lane is not sparse: %d lines for 30 steps\n%q", len(lines), first)
	}
	for _, line := range lines {
		if strings.Contains(line, "value=") {
			t.Fatalf("piped lane leaked a diagnostic spelling\n%q", line)
		}
	}
	joined := first
	for _, want := range []string{"worker selected", "running", "denoising 3/30 · 10%", "denoising 30/30 · 100%", "elapsed"} {
		if !strings.Contains(joined, want) {
			t.Errorf("piped lane omitted %q\n%q", want, joined)
		}
	}
}

// TestProgressPipedHeartbeat: a quiet stretch inside one tenth still appends one line
// per five seconds, so a slow step count never looks hung.
func TestProgressPipedHeartbeat(t *testing.T) {
	began := time.Unix(1700000000, 0)
	ctx, buf := progressContext(output.Mode{Human: true})
	p := newProgress(ctx, false, began)
	current := began
	p.now = func() time.Time { return current }
	p.on(stepEvent("denoise", 100, 1000)) // 10% — appended
	before := buf.Len()
	current = current.Add(2 * time.Second)
	p.on(stepEvent("denoise", 101, 1000)) // same tenth, quiet for 2s — suppressed
	if buf.Len() != before {
		t.Fatalf("a quiet same-tenth step should not append\n%q", buf.String())
	}
	current = current.Add(4 * time.Second)
	p.on(stepEvent("denoise", 102, 1000)) // same tenth, 6s quiet — heartbeat
	if buf.Len() == before {
		t.Fatalf("five quiet seconds must append a heartbeat line\n%q", buf.String())
	}
	p.done()
}

// TestProgressJSONStaysSilent: --json holds the machine contract — the run's one JSON
// document on stdout and NOTHING from the progress renderer.
func TestProgressJSONStaysSilent(t *testing.T) {
	ctx, buf := progressContext(output.Mode{JSON: true})
	p := newProgress(ctx, false, time.Unix(1700000000, 0))
	drive(p, denoiseRun(30))
	if buf.Len() != 0 {
		t.Fatalf("--json progress must stay silent\n%q", buf.String())
	}
}

// TestProgressStreamIsNDJSON: --stream remains the typed envelope, one JSON line per
// event, untouched by the human lanes.
func TestProgressStreamIsNDJSON(t *testing.T) {
	ctx, buf := progressContext(output.Mode{Human: true})
	p := newProgress(ctx, true, time.Unix(1700000000, 0))
	events := denoiseRun(3)
	drive(p, events)
	lines := strings.Split(strings.TrimSuffix(buf.String(), "\n"), "\n")
	if len(lines) != len(events) {
		t.Fatalf("stream lane emitted %d lines for %d events\n%q", len(lines), len(events), buf.String())
	}
	for i, line := range lines {
		if !strings.HasPrefix(line, "{") || !strings.Contains(line, `"type"`) {
			t.Fatalf("stream line %d is not a JSON envelope: %q", i, line)
		}
	}
	if !strings.Contains(buf.String(), fmt.Sprintf(`"fraction":%g`, 1.0/3.0)) {
		t.Fatalf("stream lane altered the envelope payload\n%q", buf.String())
	}
}
