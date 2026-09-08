package cli

import (
	"fmt"
	"strings"
	"time"
	"unicode"

	"golang.org/x/text/width"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/output"
)

// liveProgress owns only the current terminal block. Finishing a stage commits it
// to scrollback; later events can never erase its line. Diagnostic and JSON streams
// bypass this state entirely.
type liveProgress struct {
	key     string
	rows    []string
	drawn   int
	cells   []int
	started time.Time
	event   localapi.Event
	timer   *time.Timer
	counted bool
}

func (p *RunProgress) interactive(e localapi.Event) {
	kind := strings.TrimPrefix(e.Type, "request.")
	fields, _ := e.Payload["value"].(map[string]any)
	key := kind
	var rows []string
	counted := false
	switch kind {
	case "phase":
		if strings.HasPrefix(p.terminal.key, "stage:") {
			return // replayed preparation cannot replace current execution
		}
		name, _ := fields["phase"].(string)
		key = "phase:" + name
		rows = phaseRows(fields)
	case "progress":
		name, _ := fields["stage"].(string)
		key = "stage:" + name
		facts, usable := p.observe(fields)
		if usable {
			rows = p.stepRows(facts)
			counted = facts.counted
		} else if name != "" {
			rows = []string{"  " + stageLabel(name)}
		}
	case "stage":
		// Timing brackets emit on exit, including exceptional exit. They do not
		// prove successful completion and must not move an active nested stage back.
		return
	case "queued", "parked":
		// A queue heartbeat has less detail than the live preparation phase.
		if strings.HasPrefix(p.terminal.key, "phase:") {
			return
		}
		key = "wait"
		rows = []string{p.waitLine(e)}
	case "accepted":
		// A delayed accepted event must not replace a stage already observed.
		if strings.HasPrefix(p.terminal.key, "stage:") {
			return
		}
		rows = []string{"  starting request"}
	case "dispatched", "submitted", "metric", "log":
		return
	case "rentals", "placement":
		line, _ := e.Payload["line"].(string)
		if line != "" && line != p.last {
			p.eraseLive()
			fmt.Fprintln(p.ctx.Err, line)
			p.last = line
			p.drawLive()
		}
		return
	case "completed", "succeeded":
		p.finishLive("done")
		return
	case "failed", "canceled":
		p.finishLive(kind)
		return
	case "attempt_failed", "requeued":
		p.finishLive("retrying")
		rows = []string{progressLine(e, false)}
	default:
		return
	}
	if len(rows) == 0 || rows[0] == "" {
		return
	}
	if p.terminal.key != key {
		p.finishLive("done")
		p.terminal.started = eventTime(e)
	}
	p.terminal.key, p.terminal.rows = key, rows
	p.terminal.event, p.terminal.counted = e, counted
	p.drawLive()
	p.armLiveClock()
}

func (p *RunProgress) stepRows(f stepFacts) []string {
	line := "  " + f.label
	if f.counted {
		line += fmt.Sprintf(" %d/%d", f.current, f.total)
	}
	if f.hasStageFraction {
		line += fmt.Sprintf(" %s %.0f%% stage", progressBar(f.stageFraction, 10), f.stageFraction*100)
	}
	if f.counted && f.perStep > 0 {
		line += fmt.Sprintf(" · %.2fs/step avg", f.perStep)
		remaining := time.Duration(float64(f.total-f.current) * f.perStep * float64(time.Second))
		line += " · ETA ~" + shortDuration(remaining)
	}
	rows := []string{line}
	if f.hasOverall {
		overall := fmt.Sprintf("    overall %.0f%%", f.overallFraction*100)
		if f.hasOverallETA {
			overall += " · ETA ~" + shortDuration(f.overallRemaining)
		}
		rows = append(rows, overall)
	}
	return rows
}

func phaseRows(fields map[string]any) []string {
	line := HumanPhaseLine(fields)
	if line == "" {
		return nil
	}
	rows := []string{line}
	if rental, ok := fields["rental"].(map[string]any); ok {
		model, _ := rental["accelerator_model"].(string)
		if model != "" {
			row := "    " + model
			if count, ok := number(rental["accelerator_count"]); ok && count > 1 {
				row = fmt.Sprintf("    %.0f × %s", count, model)
			}
			if price, ok := number(rental["hourly_rate_usd_micros"]); ok && price >= 0 {
				row += fmt.Sprintf(" · $%.2f/hour", price/1_000_000)
			}
			rows = append(rows, row)
		}
	}
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

// A boot has no defensible percentage until a producer supplies an estimate. Keep
// its elapsed clock moving during quiet periods instead of inventing a 360s total.
func (p *RunProgress) armLiveClock() {
	if p.terminal.timer != nil {
		p.terminal.timer.Stop()
	}
	p.terminal.timer = time.AfterFunc(time.Second, func() {
		p.mu.Lock()
		defer p.mu.Unlock()
		if p.closed || p.terminal.key == "" {
			return
		}
		p.drawLive()
		p.armLiveClock()
	})
}

func (p *RunProgress) visibleRows() []string {
	rows := append([]string(nil), p.terminal.rows...)
	if len(rows) == 0 {
		return rows
	}
	elapsed := max(p.now().Sub(p.terminal.started), 0)
	if strings.HasPrefix(p.terminal.key, "phase:") {
		original, _ := p.terminal.event.Payload["value"].(map[string]any)
		fields := make(map[string]any, len(original)+1)
		for k, v := range original {
			fields[k] = v
		}
		prior, _ := number(fields["elapsed_ms"])
		fields["elapsed_ms"] = prior + float64(max(p.now().Sub(eventTime(p.terminal.event)), 0).Milliseconds())
		return phaseRows(fields)
	}
	// Counted rows prioritize useful speed and ETA; their clock gets its own short
	// line so those facts survive on an ordinary 80-column terminal.
	if p.terminal.counted {
		rows = append(rows, "    elapsed "+shortDuration(elapsed))
	} else {
		rows[0] += " · elapsed " + shortDuration(elapsed)
	}
	return rows
}

// eraseLive returns to the first active row. Completed rows are outside this block.
func (p *RunProgress) eraseLive() {
	if p.terminal.drawn == 0 {
		return
	}
	physical := p.terminal.drawn
	if columns := p.width(); columns > 0 {
		physical = 0
		for _, cells := range p.terminal.cells {
			physical += max(1, (cells+columns-1)/columns)
		}
	}
	if physical > 1 {
		fmt.Fprintf(p.ctx.Err, "\033[%dA", physical-1)
	}
	fmt.Fprint(p.ctx.Err, "\r\033[J")
	p.terminal.drawn = 0
}

func (p *RunProgress) drawLive() {
	rows := p.visibleRows()
	if len(rows) == 0 {
		return
	}
	p.eraseLive()
	width := p.width()
	p.terminal.cells = p.terminal.cells[:0]
	for i, row := range rows {
		if i > 0 {
			fmt.Fprint(p.ctx.Err, "\n")
		}
		shown := clampLine(row, width)
		p.terminal.cells = append(p.terminal.cells, textCells(shown))
		fmt.Fprintf(p.ctx.Err, "\r\033[K%s", shown)
	}
	p.terminal.drawn = len(rows)
}

func (p *RunProgress) finishLive(status string) {
	if p.terminal.timer != nil {
		p.terminal.timer.Stop()
	}
	if p.terminal.key == "" {
		return
	}
	rows := p.visibleRows()
	p.eraseLive()
	for i, row := range rows {
		columns := p.width()
		if i == 0 && status != "" {
			suffix := " · " + status
			if columns > len(suffix)+1 {
				row = clampLine(row, columns-len(suffix))
			}
			row += suffix
		}
		fmt.Fprintf(p.ctx.Err, "\r\033[K%s\n", clampLine(row, columns))
	}
	p.terminal = liveProgress{}
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
