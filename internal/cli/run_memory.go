package cli

import (
	"cmp"
	"fmt"
	"io"
	"maps"
	"slices"
	"strings"

	"github.com/cozy-creator/cozy/internal/units"
)

// runMemory is TensorD's record of one call's device memory (`measurements.memory`): the
// call's own peaks, its executor process at the call's end, and what its weight plane and
// torch did since that executor's previous call. A nil count is unmeasured, never zero.
type runMemory struct {
	PeakAllocatedBytes  *int64            `json:"peak_allocated_bytes,omitempty"`
	PeakReservedBytes   *int64            `json:"peak_reserved_bytes,omitempty"`
	ActivationPeakBytes *int64            `json:"activation_peak_bytes,omitempty"`
	StagePeaks          map[string]int64  `json:"stage_peaks,omitempty"`
	ProcessBytes        *int64            `json:"process_bytes,omitempty"`
	ContextBytes        *int64            `json:"context_bytes,omitempty"`
	ReservedBytes       *int64            `json:"reserved_bytes,omitempty"`
	WeightsBytes        *int64            `json:"weights_bytes,omitempty"`
	BudgetBytes         *int64            `json:"budget_bytes,omitempty"`
	CapBytes            *int64            `json:"cap_bytes,omitempty"`
	PagedInBytes        *int64            `json:"paged_in_bytes,omitempty"`
	DiskBytes           *int64            `json:"disk_bytes,omitempty"`
	BounceBytes         *int64            `json:"bounce_bytes,omitempty"`
	RegisterFailed      *int64            `json:"register_failed,omitempty"`
	Evictions           *int64            `json:"evictions,omitempty"`
	PagedOut            *int64            `json:"paged_out,omitempty"`
	PagedOutBytes       *int64            `json:"paged_out_bytes,omitempty"`
	AllocRetries        *int64            `json:"alloc_retries,omitempty"`
	AllocOOMs           *int64            `json:"alloc_ooms,omitempty"`
	OOMRetries          *int64            `json:"oom_retries,omitempty"`
	CacheReleases       *int64            `json:"cache_releases,omitempty"`
	CacheReleasedBytes  *int64            `json:"cache_released_bytes,omitempty"`
	Modes               map[string]string `json:"modes,omitempty"`
	Shortfall           *memoryShortfall  `json:"shortfall,omitempty"`
}

// memoryShortfall is what a refused stage needed at once and what the GPU could give.
type memoryShortfall struct {
	Resource       string `json:"resource"`
	NeededBytes    int64  `json:"needed_bytes"`
	AvailableBytes int64  `json:"available_bytes"`
	RequestShape   string `json:"request_shape"`
}

// emitMemory prints one line per kind of fact the record has.
func emitMemory(w io.Writer, m *runMemory) {
	if m == nil {
		return
	}
	var lines [][2]string
	add := func(name string, parts ...string) {
		if parts = slices.DeleteFunc(parts, func(p string) bool { return p == "" }); len(parts) > 0 {
			lines = append(lines, [2]string{name, strings.Join(parts, " · ")})
		}
	}
	activations := sized(m.ActivationPeakBytes, "activations ", "")
	if len(m.StagePeaks) > 0 {
		activations = "stage activations " + stagePeaks(m.StagePeaks)
	}
	add("peak", sized(m.PeakAllocatedBytes, "", " allocated"), sized(m.PeakReservedBytes, "", " reserved"), activations)
	add("at end", sized(m.ProcessBytes, "process ", ""), sized(m.ContextBytes, "context ", ""),
		sized(m.ReservedBytes, "torch ", ""), sized(m.WeightsBytes, "weights ", ""),
		sized(m.BudgetBytes, "budget ", ""), sized(m.CapBytes, "cap ", ""))
	paged := sized(m.PagedInBytes, "paged in ", "")
	add("weights", paged, nonzeroBytes("from disk ", m.DiskBytes), nonzeroBytes("bounced ", m.BounceBytes),
		nonzero("evicted ", m.Evictions, nil), nonzero("paged out ", m.PagedOut, m.PagedOutBytes),
		nonzero("page locks refused ", m.RegisterFailed, nil))
	add("allocator", nonzero("cache drops ", m.AllocRetries, nil), nonzero("OOMs ", m.AllocOOMs, nil),
		nonzero("OOM retries ", m.OOMRetries, nil), nonzero("cache releases ", m.CacheReleases, m.CacheReleasedBytes))
	if s := m.Shortfall; s != nil {
		add("refused", fmt.Sprintf("%s for %s: needed %s, %s available", cmp.Or(s.Resource, "memory"),
			dash(s.RequestShape), units.Bytes(s.NeededBytes), units.Bytes(s.AvailableBytes)))
	}
	for _, stage := range slices.Sorted(maps.Keys(m.Modes)) {
		add("mode", stage+" "+m.Modes[stage])
	}
	if len(lines) == 0 {
		return
	}
	fmt.Fprintln(w, "\nmemory")
	for _, line := range lines {
		fmt.Fprintf(w, "  %-10s %s\n", line[0], line[1])
	}
}

// sized spells a measured byte count between its label and suffix; "" when unmeasured.
func sized(n *int64, label, suffix string) string {
	if n == nil {
		return ""
	}
	return label + units.Bytes(*n) + suffix
}

// nonzero spells a count after its label, with its bytes when given. Zero is the usual case
// and says nothing.
func nonzero(label string, n, bytes *int64) string {
	switch {
	case n == nil || *n == 0:
		return ""
	case bytes != nil:
		return fmt.Sprintf("%s%d (%s)", label, *n, units.Bytes(*bytes))
	}
	return fmt.Sprintf("%s%d", label, *n)
}

// nonzeroBytes spells a nonzero byte count after its label.
func nonzeroBytes(label string, n *int64) string {
	if n == nil || *n == 0 {
		return ""
	}
	return label + units.Bytes(*n)
}

// stagePeaks names the largest stage peaks, largest first.
func stagePeaks(peaks map[string]int64) string {
	stages := slices.SortedFunc(maps.Keys(peaks), func(a, b string) int { return cmp.Compare(peaks[b], peaks[a]) })
	var parts []string
	for _, stage := range stages[:min(len(stages), 4)] {
		parts = append(parts, stage+" "+units.Bytes(peaks[stage]))
	}
	return strings.Join(parts, ", ")
}
