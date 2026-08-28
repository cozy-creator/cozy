// Package retention owns cozy-creator's LOCAL OUTPUT RETENTION POLICY (cl-033).
//
// The mirror is not a cache. An attempt's outputs are fetched onto this host BEFORE its
// terminal is acknowledged, and are deliberately kept when the pod that produced them is
// destroyed — a rented GPU costs dollars an hour and its results have to outlive it. So
// the local output namespace is the one plane here that grows forever, and something has
// to bound it without ever being the reason a user lost work.
//
// THE POLICY IS ONE RULE: an output becomes reclaimable when it is older than the
// horizon. Nothing else. In particular it is NOT an unreferenced sweep — every retained
// output belongs to a settled request by construction, so "nothing references it" is true
// of all of them and would delete exactly what this plane exists to preserve.
//
// Age is the only dimension that can never surprise: whatever a burst of generation does
// to the disk, everything produced inside the horizon is untouchable, which is precisely
// the "shut the pod down, then look at my files" guarantee. A total-bytes cap bounds the
// disk more strictly but evicts oldest-first, so a large burst can reclaim media minutes
// old — the exact failure this issue exists to prevent. The cap belongs behind a
// free-space signal, and is recorded as a follow-up rather than half-built here.
package retention

import (
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/records"
)

// DefaultHorizon is what a bare `cozy gc` plans against. Thirty days is chosen to be
// longer than any plausible "I rented a pod last week and want to look at the results"
// and short enough that a forgotten local root stops being an unbounded one.
const DefaultHorizon = 720 * time.Hour

// ParseHorizon reads `--keep-media`. A duration, and `0` means keep nothing — every
// settled output becomes a candidate. There is no magic word: a caller who wants media
// left alone passes a horizon longer than the media is old.
func ParseHorizon(v string) (time.Duration, *exit.Error) {
	if strings.TrimSpace(v) == "" {
		return DefaultHorizon, nil
	}
	d, err := time.ParseDuration(strings.TrimSpace(v))
	if err != nil || d < 0 {
		return 0, exit.New(exit.Usage, "--keep-media %q is not a non-negative duration", v).
			WithRemedy("a Go duration: 720h keeps a month, 0 keeps nothing").
			WithNext("cozy gc")
	}
	return d, nil
}

// Item is one reclaimable output: the row's identity, the bytes it will actually free,
// and why it qualified.
type Item struct {
	MediaID   string
	RequestID string
	Attempt   int64
	OutputID  string
	Endpoint  string
	Path      string
	Bytes     int64
	Age       time.Duration
	Reason    string
}

// Plan is the whole inspectable answer: what would go, what would stay, and under which
// horizon. `cozy gc` without `--yes` prints it and removes nothing.
type Plan struct {
	Horizon       time.Duration
	Items         []Item
	Bytes         int64 // what executing this plan frees
	RetainedItems int   // outputs whose bytes stay on this disk
	RetainedBytes int64
	// Foreign counts rows whose recorded path is NOT under the local output namespace.
	// They are never planned and never removed. The count is surfaced rather than
	// swallowed: a row pointing elsewhere is corruption, not a case to handle.
	Foreign int
}

// under is the deleter's fence. cozy-creator composes no store path and this function
// composes none either — it asks whether a path the RECORD carries resolves inside the
// one directory the orchestrator grants into, and refuses everything else.
func under(root, path string) bool {
	rel, err := filepath.Rel(root, path)
	if err != nil {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func onDisk(path string) int64 {
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return 0
	}
	return info.Size()
}

// Build reads the two retention questions out of the ONE local database and answers
// both. It mutates nothing.
func Build(l home.Layout, st *records.Store, horizon time.Duration, now time.Time) (Plan, *exit.Error) {
	p := Plan{Horizon: horizon}
	cutoff := now.UTC().Add(-horizon).Format(time.RFC3339Nano)
	rows, e := st.ReclaimableOutputs(cutoff)
	if e != nil {
		return p, e
	}
	planned := map[string]bool{}
	for _, r := range rows {
		if !under(l.Outputs, r.Path) {
			p.Foreign++
			continue
		}
		age := time.Duration(0)
		if t, err := time.Parse(time.RFC3339Nano, r.VisibleAt); err == nil {
			age = now.UTC().Sub(t)
		}
		bytes := onDisk(r.Path)
		reason := "older than the " + Short(horizon) + " retention horizon"
		if bytes == 0 {
			reason = "its bytes are already absent; the row is stamped so it stops being asked"
		}
		p.Items = append(p.Items, Item{
			MediaID: r.MediaID, RequestID: r.RequestID, Attempt: r.Attempt,
			OutputID: r.OutputID, Endpoint: r.Endpoint, Path: r.Path,
			Bytes: bytes, Age: age, Reason: reason,
		})
		p.Bytes += bytes
		planned[r.MediaID] = true
	}
	all, e := st.RetainedOutputs()
	if e != nil {
		return p, e
	}
	for _, r := range all {
		// Only the plane gc governs is counted as kept HERE. A job's bytes are its
		// publication and a foreign path is corruption; neither is a mirror this
		// horizon is deciding about.
		if r.ReclaimedAt != "" || planned[r.MediaID] || !under(l.Outputs, r.Path) {
			continue
		}
		p.RetainedItems++
		p.RetainedBytes += onDisk(r.Path)
	}
	return p, nil
}

// Collect executes the plan. The order is BYTES FIRST, STAMP SECOND on purpose: a crash
// between the two leaves a row whose file is gone, which the next plan re-selects, frees
// nothing, and stamps. The converse order would leave bytes nothing ever looks at again.
func Collect(l home.Layout, st *records.Store, p Plan) (int64, *exit.Error) {
	var freed int64
	dirs := map[string]bool{}
	for _, it := range p.Items {
		// Re-checked at the moment of removal, not only when the plan was built.
		if !under(l.Outputs, it.Path) {
			return freed, exit.Internalf(
				"refusing to reclaim %s: %s is outside the local output namespace",
				it.MediaID, it.Path)
		}
		if err := os.Remove(it.Path); err != nil && !os.IsNotExist(err) {
			return freed, exit.Internalf("cannot reclaim %s: %s", it.MediaID, err)
		}
		if e := st.MarkOutputReclaimed(it.MediaID); e != nil {
			return freed, e
		}
		freed += it.Bytes
		dirs[filepath.Dir(it.Path)] = true
	}
	prune(l, dirs)
	return freed, nil
}

// prune removes attempt directories this collection emptied, and the request directory
// above one when it too is empty. os.Remove on a non-empty directory fails, which is the
// check — nothing walks and nothing decides.
func prune(l home.Layout, dirs map[string]bool) {
	for dir := range dirs {
		if !under(l.Outputs, dir) || dir == l.Outputs {
			continue
		}
		if os.Remove(dir) != nil {
			continue
		}
		parent := filepath.Dir(dir)
		if under(l.Outputs, parent) && parent != l.Outputs {
			_ = os.Remove(parent)
		}
	}
}

// Short renders a horizon the way a user typed it: 720h reads as 30d.
func Short(d time.Duration) string {
	if d == 0 {
		return "0"
	}
	if d%(24*time.Hour) == 0 {
		return itoa(int64(d/(24*time.Hour))) + "d"
	}
	return d.String()
}

// Age renders how old one retained output is, at day resolution once it has one.
func Age(d time.Duration) string {
	if d >= 24*time.Hour {
		return itoa(int64(d/(24*time.Hour))) + "d"
	}
	if d >= time.Hour {
		return itoa(int64(d/time.Hour)) + "h"
	}
	if d >= time.Minute {
		return itoa(int64(d/time.Minute)) + "m"
	}
	if d < 0 {
		return "0s"
	}
	return itoa(int64(d/time.Second)) + "s"
}

func itoa(n int64) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
