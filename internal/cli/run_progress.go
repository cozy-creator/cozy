package cli

import (
	"fmt"
	"maps"
	"strings"
	"time"
	"unicode"

	"golang.org/x/text/width"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
)

// Spend and placement arrive interleaved. Each has its own last observation;
// neither an unchanged heartbeat nor an unrelated stage makes it new again.
func (p *RunProgress) rentalNotice(e localapi.Event) {
	previous := &p.rentalLine
	if strings.TrimPrefix(e.Type, "request.") == "placement" {
		previous = &p.placementLine
	}
	line, _ := e.Payload["line"].(string)
	if line == "" || line == *previous {
		return
	}
	*previous = line
	p.notice(e, line)
}

// reattachNotice says, once per daemon restart, that a follower re-attached: through the
// live progress block while one is drawing, plainly otherwise.
type reattachNotice struct {
	ctx   *Context
	lines *RunProgress
}

func (n *reattachNotice) say() {
	const line = "daemon restarted; reattached"
	// Later dials in this command address the restarted daemon, not the one that stopped.
	n.ctx.Daemon = daemon.Probe(n.ctx.Cfg)
	if n.lines != nil {
		n.lines.mu.Lock()
		defer n.lines.mu.Unlock()
		n.lines.notice(localapi.Event{}, line)
		return
	}
	fmt.Fprintln(n.ctx.Err, line)
}

func waitingPlacement(payload map[string]any) bool {
	waiting := false
	candidates, _ := payload["candidates"].([]any)
	for _, value := range candidates {
		candidate, _ := value.(map[string]any)
		switch candidate["verdict"] {
		case orchestrator.VerdictChosen:
			return false
		case orchestrator.VerdictAttaching:
			waiting = true
		}
	}
	return waiting
}

// Manual rentals observe the same Hub lifecycle facts as a run's acquisition.
// The shared renderer owns elapsed clocks and terminal history for both commands.
func (p *RunProgress) rentalAcquisition(r hub.Rental) {
	name := orchestrator.PhaseOfHubRental(r.State, r.ProviderState, r.ContainerState, r.Failure != nil)
	if name == "" {
		if !r.Attachable() {
			return
		}
		name = "connecting"
	}
	p.On(localapi.Event{Type: "request.phase", Payload: map[string]any{
		"value": map[string]any{"phase": name, "machine": r.Name,
			"rental": map[string]any{"accelerator_model": r.AcceleratorModel,
				"accelerator_count": r.AcceleratorCount, "hourly_rate_usd_micros": r.HourlyRateUSDMicros,
				"base_worker_image_digest": r.BaseWorkerImageDigest,
				"base_worker_image_tag":    r.BaseWorkerImageTag, "base_worker_profile": r.BaseWorkerProfile},
		},
	}})
}

func (p *RunProgress) sparsePhase(e localapi.Event) {
	fields, _ := e.Payload["value"].(map[string]any)
	name, _ := fields["phase"].(string)
	if name == "" {
		return
	}
	key := "phase:" + name
	if key != p.sparseStage {
		p.sparseStarted = eventTime(e)
	} else if p.now().Sub(p.sparseAt) < 5*time.Second {
		return
	}
	p.sparseStage, p.sparseAt = key, p.now()
	for _, row := range phaseRows(phaseFieldsAt(e, p.sparseStarted, p.now())) {
		fmt.Fprintln(p.ctx.Err, row)
	}
}

func phaseFieldsAt(e localapi.Event, started, at time.Time) map[string]any {
	original, _ := e.Payload["value"].(map[string]any)
	fields := make(map[string]any, len(original)+1)
	for k, v := range original {
		fields[k] = v
	}
	if prior, measured := number(fields["elapsed_ms"]); measured {
		fields["elapsed_ms"] = prior + float64(max(at.Sub(eventTime(e)), 0).Milliseconds())
	} else {
		fields["elapsed_ms"] = float64(max(at.Sub(started), 0).Milliseconds())
	}
	if age, measured := number(fields["sample_age_ms"]); measured {
		fields["sample_age_ms"] = age + float64(max(at.Sub(eventTime(e)), 0).Milliseconds())
	}
	if models, ok := fields["models"].([]any); ok {
		aged := make([]any, len(models))
		for i, raw := range models {
			aged[i] = raw
			if model, ok := raw.(map[string]any); ok {
				copy := maps.Clone(model)
				if age, measured := number(copy["sample_age_ms"]); measured {
					copy["sample_age_ms"] = age + float64(max(at.Sub(eventTime(e)), 0).Milliseconds())
				}
				aged[i] = copy
			}
		}
		fields["models"] = aged
	}
	return fields
}

func (f stepFacts) countLabel() string {
	if !f.counted {
		return ""
	}
	if f.bytes {
		return " · " + output.Bytes(f.current) + " / " + output.Bytes(f.total)
	}
	if f.scoped {
		return fmt.Sprintf(" · step %d/%d", f.current, f.total)
	}
	return fmt.Sprintf(" %d/%d", f.current, f.total)
}

func (f stepFacts) timing() string {
	if f.bytes {
		// Runtime's measured rate since its previous sample; a zero rate is a stall.
		if !f.hasRate {
			return ""
		}
		line := " · " + output.Bytes(int64(f.rate)) + "/s"
		if f.rate > 0 && f.current < f.total {
			line += " · ETA ~" + shortDuration(time.Duration(float64(f.total-f.current)/f.rate*float64(time.Second)))
		}
		return line
	}
	if !f.counted || f.perStep <= 0 {
		return ""
	}
	remaining := time.Duration(float64(f.total-f.current) * f.perStep * float64(time.Second))
	return fmt.Sprintf(" · %.2fs/step avg · ETA ~%s", f.perStep, shortDuration(remaining))
}

func phaseRows(fields map[string]any) []string {
	line := HumanPhaseLine(fields)
	if line == "" {
		return nil
	}
	rows := append([]string{line}, rentalRows(fields)...)
	// Older workers report one aggregate transfer. Preserve that honest fallback;
	// per-model rows are shown only when their producer provides separate counters.
	models, _ := fields["models"].([]any)
	for _, value := range models {
		model, ok := value.(map[string]any)
		if !ok {
			continue
		}
		name, _ := model["model"].(string)
		if name == "" {
			continue
		}
		if lane, _ := model["lane"].(string); lane != "" {
			name += "/" + lane
		}
		rows = append(rows, "    "+name)
		row := "      "
		moved, hasMoved := number(model["moved_bytes"])
		total, hasTotal := number(model["total_bytes"])
		if hasMoved {
			row += output.Bytes(int64(moved))
			if hasTotal && total > 0 {
				row += " / " + output.Bytes(int64(total)) + " " + progressBar(moved/total, 10)
			}
		}
		age, measured := number(model["sample_age_ms"])
		if !measured {
			age, _ = number(fields["sample_age_ms"])
		}
		if time.Duration(age)*time.Millisecond > orchestrator.PreparationRateMaxAge {
			rows = append(rows, row+" · last update "+shortDuration(time.Duration(age)*time.Millisecond)+" ago")
			continue
		}
		if rate, ok := number(model["rate_bytes_per_second"]); ok && rate > 0 {
			row += " · " + output.Bytes(int64(rate)) + "/s"
		}
		if remaining, ok := number(model["remaining_ms"]); ok && remaining > 0 {
			row += " · ETA ~" + shortDuration(time.Duration(remaining)*time.Millisecond)
		}
		if strings.TrimSpace(row) != "" {
			rows = append(rows, row)
		}
	}
	return rows
}

// rentalRows are a rental's accelerators, price and worker image.
func rentalRows(fields map[string]any) []string {
	rental, _ := fields["rental"].(map[string]any)
	var rows []string
	if model, _ := rental["accelerator_model"].(string); model != "" {
		row := "    " + model
		if count, ok := number(rental["accelerator_count"]); ok && count > 1 {
			row = fmt.Sprintf("    %.0f × %s", count, model)
		}
		if price, ok := number(rental["hourly_rate_usd_micros"]); ok && price >= 0 {
			row += fmt.Sprintf(" · $%.2f/hour", price/1_000_000)
		}
		rows = append(rows, row)
	}
	for _, field := range []string{"base_worker_image_tag", "base_worker_profile"} {
		if label, _ := rental[field].(string); label != "" {
			return append(rows, "    image: "+label)
		}
	}
	return rows
}

// Count terminal cells, not UTF-8 bytes or code points. Counting a joined emoji
// conservatively can leave spare room; undercounting would wrap into scrollback.
func runeCells(r rune) int {
	if unicode.IsControl(r) || unicode.Is(unicode.Mn, r) || unicode.Is(unicode.Me, r) || unicode.Is(unicode.Cf, r) {
		return 0
	}
	switch width.LookupRune(r).Kind() {
	case width.EastAsianWide, width.EastAsianFullwidth:
		return 2
	default:
		return 1
	}
}

func textCells(s string) int {
	n := 0
	for _, r := range s {
		n += runeCells(r)
	}
	return n
}
