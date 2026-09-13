package cli

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"golang.org/x/term"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/output"
)

// The live list board (cl-107, cl-114): an alternate-screen table refreshed in place at a
// one-second cadence, scrolled by keys and wheel, restored exactly on exit. `cozy run
// list` and `cozy rental list` are both this board with their own snapshot function; the
// board itself never mutates anything — it only re-reads and redraws.

type listNavigation uint8

const (
	listUp listNavigation = iota + 1
	listDown
	listPageUp
	listPageDown
	listHome
	listEnd
)

// watchList runs the board until q/Esc/Ctrl-C. anchor names the column whose value keeps
// the scrolled-to row stable across refreshes; fetch produces each snapshot.
func watchList(ctx *Context, anchor string,
	fetch func(context.Context) (output.List, *exit.Error),
) *exit.Error {
	navigation := make(chan listNavigation, 32)
	watchCtx, restoreInput, mouse, problem := liveWatchContext(ctx, context.Background(), navigation)
	if problem != nil {
		return problem
	}
	controls := "\x1b[?1049h\x1b[?25l"
	if mouse {
		controls += "\x1b[?1000h\x1b[?1006h"
	}
	if _, err := io.WriteString(ctx.Out, controls); err != nil {
		restoreInput()
		return exit.As(err)
	}
	defer func() {
		if mouse {
			_, _ = io.WriteString(ctx.Out, "\x1b[?1006l\x1b[?1000l")
		}
		_, _ = io.WriteString(ctx.Out, "\x1b[?25h\x1b[?1049l")
		restoreInput()
	}()
	viewport := listViewport{anchor: anchor}
	var drawnWidth, drawnHeight int
	draw := func() *exit.Error {
		drawnWidth, drawnHeight = terminalSize(ctx.Out)
		body := viewport.frame(drawnWidth, drawnHeight, ctx.Mode().Full)
		if _, err := fmt.Fprintf(ctx.Out, "\x1b[H\x1b[J%s", body); err != nil {
			return exit.As(err)
		}
		return nil
	}
	refresh := func() *exit.Error {
		list, problem := fetch(watchCtx)
		if problem != nil {
			return problem
		}
		if err := viewport.update(list, ctx.Mode()); err != nil {
			return exit.As(err)
		}
		return draw()
	}
	if problem := refresh(); problem != nil {
		if watchCtx.Err() != nil {
			return nil
		}
		return problem
	}
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	resize := time.NewTicker(100 * time.Millisecond)
	defer resize.Stop()
	for {
		select {
		case <-watchCtx.Done():
			return nil
		case move := <-navigation:
			viewport.move(move, terminalHeight(ctx.Out), ctx.Mode().Full)
			if problem := draw(); problem != nil {
				return problem
			}
		case <-resize.C:
			if width, height := terminalSize(ctx.Out); width != drawnWidth || height != drawnHeight {
				if problem := draw(); problem != nil {
					return problem
				}
			}
		case <-ticker.C:
			if problem := refresh(); problem != nil {
				if watchCtx.Err() != nil {
					return nil
				}
				return problem
			}
		}
	}
}

type listViewport struct {
	anchor      string
	list        output.List
	table       output.Table
	top         int
	anchorValue string
}

func (v *listViewport) update(list output.List, mode output.Mode) error {
	table, err := list.Table(mode)
	if err != nil {
		return err
	}
	if v.top > 0 && v.anchorValue != "" {
		for index, row := range list.Rows {
			if row[v.anchor] == v.anchorValue {
				v.top = index
				break
			}
		}
	}
	v.list = list
	v.table = table
	return nil
}

func (v *listViewport) layout(height int, full bool) (lead, footer []string, rows int) {
	if height <= 0 {
		height = 24
	}
	// Reserve the heading and navigation before data. On a one-line terminal the
	// heading alone wins; on two lines it shares the screen with navigation.
	available := max(0, height-2)
	contextRows := max(0, available-1)
	lead = v.table.Lead[:min(len(v.table.Lead), contextRows)]
	for len(lead) > 0 && lead[len(lead)-1] == "" {
		lead = lead[:len(lead)-1]
	}
	footer = v.table.Footer[:min(len(v.table.Footer), contextRows-len(lead))]
	for len(footer) > 0 && footer[len(footer)-1] == "" {
		footer = footer[:len(footer)-1]
	}
	rows = available - len(lead) - len(footer)
	if !full {
		rows = min(rows, 20)
	}
	return lead, footer, rows
}

func (v *listViewport) clamp(pageRows int) {
	v.top = min(max(v.top, 0), max(0, len(v.list.Rows)-max(1, pageRows)))
	if v.top == 0 || len(v.list.Rows) == 0 {
		v.anchorValue = ""
		return
	}
	v.anchorValue = v.list.Rows[v.top][v.anchor]
}

func (v *listViewport) move(move listNavigation, height int, full bool) {
	_, _, pageRows := v.layout(height, full)
	switch move {
	case listUp:
		v.top--
	case listDown:
		v.top++
	case listPageUp:
		v.top -= max(1, pageRows)
	case listPageDown:
		v.top += max(1, pageRows)
	case listHome:
		v.top = 0
	case listEnd:
		v.top = len(v.list.Rows)
	}
	v.clamp(pageRows)
}

func (v *listViewport) frame(width, height int, full bool) string {
	lead, footer, pageRows := v.layout(height, full)
	v.clamp(pageRows)
	end := min(v.top+pageRows, len(v.list.Rows))
	location := "no rows"
	if end > v.top {
		location = fmt.Sprintf("rows %d-%d/%d", v.top+1, end, len(v.list.Rows))
	} else if len(v.list.Rows) > 0 {
		location = fmt.Sprintf("%d rows · enlarge terminal", len(v.list.Rows))
	}
	lines := append(append([]string(nil), lead...), v.table.Header)
	lines = append(lines, v.table.Rows[v.top:end]...)
	lines = append(lines, footer...)
	if height != 1 {
		lines = append(lines, fmt.Sprintf(
			"%s · q/Esc/Ctrl-C exits · wheel/↑↓/PgUp/PgDn/Home/End · 1s refresh", location))
	}
	for i, line := range lines {
		lines[i] = clampLine(line, width)
	}
	// The shared terminal input mode keeps normal newline processing. No final
	// newline: at the bottom it would scroll the fixed heading out of the frame.
	return strings.Join(lines, "\n")
}

func terminalHeight(w io.Writer) int {
	_, height := terminalSize(w)
	return height
}

func terminalSize(w io.Writer) (int, int) {
	file, ok := w.(*os.File)
	if !ok {
		return 0, 0
	}
	width, height, err := term.GetSize(int(file.Fd()))
	if err != nil {
		return 0, 0
	}
	return width, height
}
