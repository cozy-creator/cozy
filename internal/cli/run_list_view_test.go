package cli

import (
	"bufio"
	"context"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/output"
)

func TestRunListViewportScrollAndRefreshAnchor(t *testing.T) {
	rows := make([]map[string]string, 30)
	for index := range rows {
		rows[index] = map[string]string{"id": "request-" + string(rune('a'+index))}
	}
	view := runListViewport{list: output.List{
		Name: "invocations", Rows: rows,
		Aggregates: []output.Field{{K: "queued", V: 1}, {K: "completed", V: 29}},
	}}
	if page := view.page(10, false); len(page.Rows) != 4 || page.Rows[0]["id"] != "request-a" {
		t.Fatalf("initial viewport is not height-bounded at the newest row: %+v", page.Rows)
	}
	for range 3 {
		view.move(runListDown, 10, false)
	}
	if page := view.page(10, false); page.Rows[0]["id"] != "request-d" {
		t.Fatalf("downward scroll did not move three rows: %+v", page.Rows)
	}
	newest := map[string]string{"id": "request-new"}
	view.update(output.List{Name: "invocations", Rows: append([]map[string]string{newest}, rows...),
		Aggregates: view.list.Aggregates})
	if page := view.page(10, false); page.Rows[0]["id"] != "request-d" {
		t.Fatalf("refresh moved a scrolled reader when a new row arrived: %+v", page.Rows)
	}
	view.move(runListHome, 10, false)
	if page := view.page(10, false); page.Rows[0]["id"] != "request-new" {
		t.Fatalf("Home did not resume the live top: %+v", page.Rows)
	}
}

func TestRunListInputMouseKeysAndQuit(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	navigation := make(chan runListNavigation, 16)
	readRunListInput(bufio.NewReader(strings.NewReader(
		"\x1b[<65;2;3M\x1b[A\x1b[6~\x1b[Hq")), cancel, navigation)
	if ctx.Err() == nil {
		t.Fatal("q did not exit the live view")
	}
	want := []runListNavigation{
		runListDown, runListDown, runListDown, runListUp, runListPageDown, runListHome,
	}
	for index, expected := range want {
		select {
		case got := <-navigation:
			if got != expected {
				t.Fatalf("navigation %d = %d, want %d", index, got, expected)
			}
		default:
			t.Fatalf("missing navigation %d (%d)", index, expected)
		}
	}
}
