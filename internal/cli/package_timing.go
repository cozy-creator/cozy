package cli

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/output"
)

// packagePublishTimings is retained in the successful publication record so a
// caller can distinguish local preparation, object transfer, and Hub
// finalization. Durations are seconds (rather than formatted strings) in the
// machine document; the Human implementation keeps the compact terminal
// result readable.
type packagePublishTimings struct {
	Total  float64            `json:"total_seconds"`
	Phases map[string]float64 `json:"phases_seconds"`
}

func newPackagePublishTimings() *packagePublishTimings {
	return &packagePublishTimings{Phases: map[string]float64{}}
}

func (t *packagePublishTimings) measure(label string, started time.Time) {
	if t == nil {
		return
	}
	t.Phases[label] = time.Since(started).Seconds()
}

func (t *packagePublishTimings) stage(ctx *Context, label string, run func() *exit.Error) *exit.Error {
	started := time.Now()
	problem := packagePublishStage(ctx, label, run)
	t.measure(label, started)
	return problem
}

func (t *packagePublishTimings) finish(started time.Time) {
	if t != nil {
		t.Total = time.Since(started).Seconds()
	}
}

// Human keeps the default publication result useful without making callers
// pass --full. The exact phase values remain available to JSON consumers.
func (t packagePublishTimings) Human() string {
	if len(t.Phases) == 0 {
		return fmt.Sprintf("%.2fs", t.Total)
	}
	keys := make([]string, 0, len(t.Phases))
	for key := range t.Phases {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys)+1)
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s %.2fs", strings.ReplaceAll(key, "_", " "), t.Phases[key]))
	}
	parts = append(parts, fmt.Sprintf("total %.2fs", t.Total))
	return strings.Join(parts, ", ")
}

var _ output.Human = packagePublishTimings{}
