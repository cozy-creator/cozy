package cli

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"
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
	signalCtx, stopSignals := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	watchCtx, cancel := context.WithCancel(signalCtx)
	defer stopSignals()
	defer cancel()
	navigation := make(chan listNavigation, 32)
	restoreInput, mouse := startListInput(cancel, navigation)
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
	// Explicit CRLF also works with raw input. No final newline: at the bottom
	// of the screen it would scroll the fixed heading out of the visible frame.
	return strings.Join(lines, "\r\n")
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

func startListInput(cancel context.CancelFunc, navigation chan<- listNavigation) (func(), bool) {
	fd := int(os.Stdin.Fd()) //cozy:stdin-value live-list navigation, never a prompt
	if !term.IsTerminal(fd) {
		return func() {}, false
	}
	state, err := term.MakeRaw(fd)
	if err != nil {
		return func() {}, false
	}
	go readListInput(bufio.NewReader(os.Stdin), cancel, navigation) //cozy:stdin-value navigation only
	return func() { _ = term.Restore(fd, state) }, true
}

func readListInput(in *bufio.Reader, cancel context.CancelFunc, navigation chan<- listNavigation) {
	send := func(move listNavigation) {
		select {
		case navigation <- move:
		default:
		}
	}
	for {
		key, err := in.ReadByte()
		if err != nil {
			return
		}
		switch key {
		case 3, 'q', 'Q':
			cancel()
			return
		case 'k':
			send(listUp)
		case 'j':
			send(listDown)
		case 'g':
			send(listHome)
		case 'G':
			send(listEnd)
		case 0x1b:
			if !readListEscape(in, send) {
				cancel()
				return
			}
		}
	}
}

// readListEscape decides what the ESC byte just read meant, and reports whether the
// board keeps running. Esc is both a key and the first byte of every arrow, page, and
// wheel report, so the two have to be told apart before Esc can exit.
//
// The tell is structural, never a deadline waited out: a terminal writes a report's
// bytes in one burst, so a report's introducer — CSI '[' or SS3 'O' — is already sitting
// in the reader behind the ESC that introduced it. An ESC with nothing behind it, or
// with a byte that cannot introduce a sequence behind it (a second ESC, an Alt chord),
// is the Esc KEY, and the Esc key exits exactly as q does.
func readListEscape(in *bufio.Reader, send func(listNavigation)) bool {
	if in.Buffered() == 0 {
		return false
	}
	introducer, err := in.Peek(1)
	if err != nil || len(introducer) == 0 {
		return false
	}
	switch introducer[0] {
	case '[':
		_, _ = in.ReadByte()
		readListCSISequence(in, send)
		return true
	case 'O':
		// SS3, the same cursor keys from a terminal in application cursor mode.
		_, _ = in.ReadByte()
		final, err := in.ReadByte()
		if err != nil {
			return true
		}
		switch final {
		case 'A':
			send(listUp)
		case 'B':
			send(listDown)
		case 'H':
			send(listHome)
		case 'F':
			send(listEnd)
		}
		return true
	}
	return false
}

func readListCSISequence(in *bufio.Reader, send func(listNavigation)) {
	sequence := readListCSI(in)
	if sequence == "" {
		return
	}
	switch {
	case sequence == "A":
		send(listUp)
	case sequence == "B":
		send(listDown)
	case sequence == "5~":
		send(listPageUp)
	case sequence == "6~":
		send(listPageDown)
	case sequence == "H", sequence == "1~", sequence == "7~":
		send(listHome)
	case sequence == "F", sequence == "4~", sequence == "8~":
		send(listEnd)
	case strings.HasPrefix(sequence, "<64;") && strings.HasSuffix(sequence, "M"):
		for range 3 {
			send(listUp)
		}
	case strings.HasPrefix(sequence, "<65;") && strings.HasSuffix(sequence, "M"):
		for range 3 {
			send(listDown)
		}
	}
}

func readListCSI(in *bufio.Reader) string {
	var sequence strings.Builder
	for sequence.Len() < 64 {
		value, err := in.ReadByte()
		if err != nil {
			return ""
		}
		sequence.WriteByte(value)
		if value >= 0x40 && value <= 0x7e {
			return sequence.String()
		}
	}
	return ""
}
