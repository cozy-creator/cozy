package cli

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/hub"
)

// A rental's boot in the words every view uses: `cozy run list`, `run show`, the live
// `run watch` and `cozy rental list` all say the same thing about the same boot.

// bootWords is one boot described at a moment.
type bootWords struct {
	// head names the attempt when it is not the rental's first.
	head string
	// replan is the attempt this one replaced: where, for how long, and why it ended.
	replan string
	// where is this attempt's datacenter.
	where string
	// elapsed is how long this attempt has run; empty before it has a start.
	elapsed string
	// stage is how far the provider has brought the container.
	stage string
	// activity is when the provider's boot log last moved, before a container runs.
	activity string
}

func describeBoot(machine string, b *hub.RentalBoot, now time.Time) bootWords {
	var w bootWords
	if b.Attempt > 1 {
		w.head = fmt.Sprintf("boot attempt %d of rental %s", b.Attempt, machine)
	}
	if from := b.ReplannedFrom; from != nil {
		why := strings.ReplaceAll(from.FailureCode, "_", " ")
		if from.ObservedMaxMS > 0 {
			why = fmt.Sprintf("no progress past the observed %.1fm max", float64(from.ObservedMaxMS)/float64(time.Minute/time.Millisecond))
		}
		w.replan = "replanned from " + from.Datacenter + " after " + brief(from.EndedAt.Sub(from.StartedAt))
		if why != "" {
			w.replan += " (" + why + ")"
		}
	}
	w.where = b.Datacenter
	if w.where != "" && b.ReplannedFrom != nil {
		w.where = "now " + w.where
	}
	if !b.StartedAt.IsZero() {
		w.elapsed = brief(now.Sub(b.StartedAt))
	}
	switch {
	case b.State == "replanning":
		w.stage = "finding another host"
	case b.State == "obligated" || b.State == "ambiguous":
		w.stage = "creating pod"
	case !b.HostAnsweredAt.IsZero():
		w.stage = "starting Runtime (" + brief(now.Sub(b.HostAnsweredAt)) + ")"
	case !b.ContainerStartedAt.IsZero():
		w.stage = "starting supervisor (" + brief(now.Sub(b.ContainerStartedAt)) + ")"
	case b.Started():
		w.stage = "starting Runtime"
	case b.Activity != "":
		w.stage = strings.ReplaceAll(b.Activity, "_", " ")
	case b.Container == "created":
		w.stage = "container created"
	default:
		w.stage = "container not started"
	}
	if !b.Started() && !b.BootLogAt.IsZero() {
		w.activity = "boot log active " + ago(now.Sub(b.BootLogAt))
	}
	return w
}

// line is the whole sentence: "waiting for nonomiya to boot · AP-IN-1 · 6m · container not
// started · boot log active 17s ago", or for a replacement attempt "boot attempt 2 of rental
// nonomiya · replanned from AP-IN-1 after 5m (…) · now EU-RO-1 · 40s · pulling image".
func (w bootWords) line(machine string) string {
	return joinParts(w.subject(machine), w.replan, w.where, w.elapsed, w.stage, w.activity)
}

func (w bootWords) subject(machine string) string {
	if w.head != "" {
		return w.head
	}
	return "waiting for " + machine + " to boot"
}

// facts are the details after the subject and elapsed time, for a view that shows those
// apart in a narrow cell: the current attempt first, the one it replaced last.
func (w bootWords) facts() string {
	return joinParts(w.where, w.stage, w.activity, w.replan)
}

func joinParts(parts ...string) string {
	kept := parts[:0:0]
	for _, part := range parts {
		if part != "" {
			kept = append(kept, part)
		}
	}
	return strings.Join(kept, " · ")
}

// brief is a coarse duration: seconds under a minute, then minutes, then hours and minutes.
func brief(d time.Duration) string {
	d = max(d, 0)
	switch {
	case d < time.Minute:
		return fmt.Sprintf("%ds", int(d/time.Second))
	case d < time.Hour:
		return fmt.Sprintf("%dm", int(d/time.Minute))
	}
	return fmt.Sprintf("%dh%dm", int(d/time.Hour), int(d%time.Hour/time.Minute))
}

func ago(d time.Duration) string {
	if d < time.Second {
		return "just now"
	}
	return brief(d) + " ago"
}

// bootOf reads the boot a phase frame carries.
func bootOf(fields map[string]any) *hub.RentalBoot {
	raw, ok := fields["boot"]
	if !ok || raw == nil {
		return nil
	}
	if boot, ok := raw.(*hub.RentalBoot); ok {
		return boot
	}
	data, err := json.Marshal(raw)
	if err != nil {
		return nil
	}
	var boot hub.RentalBoot
	if json.Unmarshal(data, &boot) != nil {
		return nil
	}
	return &boot
}

// jsonValue is a frame value as the event stream carries it: JSON objects, not Go types.
func jsonValue(value map[string]any) map[string]any {
	data, err := json.Marshal(value)
	if err != nil {
		return value
	}
	var out map[string]any
	if json.Unmarshal(data, &out) != nil {
		return value
	}
	return out
}
