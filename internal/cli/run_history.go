package cli

import (
	"context"
	"sort"

	"github.com/cozy-creator/cozy/internal/api"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
)

const runHistoryPage = 50

// Only the board's single in-flight fetch touches this cache. Navigation reads
// already rendered rows and never calls the daemon per key or per row.
type runHistory struct {
	client             *localapi.Client
	state, packageName string
	limit              int
	rows               []api.Lifecycle
	more               bool
	// hubs names each run's hub, for --full.
	hubs *config.Config
}

func (h *runHistory) snapshot() listSnapshot {
	list := runListRows(h.rows)
	list.Aggregates = nil // Loaded pages are not a census of all retained history.
	if h.hubs != nil {
		nameHubs(*h.hubs, &list)
	}
	return listSnapshot{list: list, more: h.more}
}

func (h *runHistory) refresh(ctx context.Context, top, visible int) (listSnapshot, *exit.Error) {
	start := min(top/runHistoryPage*runHistoryPage, len(h.rows))
	before := int64(0)
	if start > 0 {
		before = h.rows[start-1].Number
	}
	count := runHistoryPage
	if top > 0 && len(h.rows) > 0 {
		if start == 0 {
			before = h.rows[0].Number + 1
		}
		count = min(500, max(runHistoryPage, top-start+visible))
		// New head rows and the visible historical page are refreshed in two
		// bounded reads. Offscreen history remains cached.
		if _, problem := h.load(ctx, 0, min(runHistoryPage, positiveLimit(h.limit)), false); problem != nil {
			return listSnapshot{}, problem
		}
	}
	if h.limit > 0 {
		count = min(count, max(0, h.limit-start))
	}
	return h.load(ctx, before, count, false)
}

func (h *runHistory) next(ctx context.Context) (listSnapshot, *exit.Error) {
	if !h.more || len(h.rows) == 0 {
		return h.snapshot(), nil
	}
	count := runHistoryPage
	if h.limit > 0 {
		count = min(count, h.limit-len(h.rows))
	}
	return h.load(ctx, h.rows[len(h.rows)-1].Number, count, true)
}

func (h *runHistory) load(ctx context.Context, before int64, count int, appendPage bool) (listSnapshot, *exit.Error) {
	if count <= 0 {
		h.more = false
		return h.snapshot(), nil
	}
	history, problem := h.client.RequestPage(ctx, h.state, h.packageName, count, before)
	if problem != nil {
		return listSnapshot{}, problem
	}
	page := history.Requests
	if len(page) > 0 {
		next := page[len(page)-1].Number
		if next < 1 || (before > 0 && next >= before) {
			return listSnapshot{}, exit.New(exit.Conflict, "run history page did not advance")
		}
	}
	if len(h.rows) == 0 || appendPage {
		h.more = len(page) == count
	}
	byID := make(map[string]api.Lifecycle, len(h.rows)+len(page))
	lower := int64(0)
	if len(page) == count {
		lower = page[len(page)-1].Number
	}
	for _, row := range h.rows {
		if !appendPage && (before == 0 || row.Number < before) && row.Number >= lower {
			continue
		}
		byID[row.RequestID] = row
	}
	for _, row := range page {
		byID[row.RequestID] = row
	}
	h.rows = h.rows[:0]
	for _, row := range byID {
		h.rows = append(h.rows, row)
	}
	sort.Slice(h.rows, func(i, j int) bool { return h.rows[i].Number > h.rows[j].Number })
	if h.limit > 0 && len(h.rows) >= h.limit {
		h.rows = h.rows[:h.limit]
		h.more = false
	}
	return h.snapshot(), nil
}

func positiveLimit(limit int) int {
	if limit > 0 {
		return limit
	}
	return runHistoryPage
}
