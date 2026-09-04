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
	draw := func() *exit.Error {
		list := viewport.page(terminalHeight(ctx.Out), ctx.Mode().Full)
		var frame strings.Builder
		if err := list.Emit(&frame, ctx.Mode()); err != nil {
			return exit.As(err)
		}
		body := frame.String()
		if mouse {
			// MakeRaw disables terminal newline translation. An explicit carriage return keeps
			// every table row in column one on every refresh.
			body = strings.ReplaceAll(body, "\n", "\r\n")
		}
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
		viewport.update(list)
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
	for {
		select {
		case <-watchCtx.Done():
			return nil
		case move := <-navigation:
			viewport.move(move, terminalHeight(ctx.Out), ctx.Mode().Full)
			if problem := draw(); problem != nil {
				return problem
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
	top         int
	anchorValue string
}

func (v *listViewport) update(list output.List) {
	if v.top > 0 && v.anchorValue != "" {
		for index, row := range list.Rows {
			if row[v.anchor] == v.anchorValue {
				v.top = index
				break
			}
		}
	}
	v.list = list
}

func (v *listViewport) capacity(height int, full bool) int {
	if height <= 0 {
		height = 24
	}
	fixed := len(v.list.Aggregates) + len(v.list.Trail) + 4
	if lead := len(v.list.Lead); lead > 0 {
		fixed += lead + 1
	}
	rows := max(1, height-fixed)
	if !full {
		rows = min(rows, 20)
	}
	return rows
}

func (v *listViewport) clamp(pageRows int) {
	v.top = min(max(v.top, 0), max(0, len(v.list.Rows)-pageRows))
	if v.top == 0 || len(v.list.Rows) == 0 {
		v.anchorValue = ""
		return
	}
	v.anchorValue = v.list.Rows[v.top][v.anchor]
}

func (v *listViewport) move(move listNavigation, height int, full bool) {
	pageRows := v.capacity(height, full)
	switch move {
	case listUp:
		v.top--
	case listDown:
		v.top++
	case listPageUp:
		v.top -= pageRows
	case listPageDown:
		v.top += pageRows
	case listHome:
		v.top = 0
	case listEnd:
		v.top = len(v.list.Rows)
	}
	v.clamp(pageRows)
}

func (v *listViewport) page(height int, full bool) output.List {
	pageRows := v.capacity(height, full)
	v.clamp(pageRows)
	end := min(v.top+pageRows, len(v.list.Rows))
	page := v.list
	page.Rows = v.list.Rows[v.top:end]
	page.Total = len(page.Rows)
	location := "no rows"
	if end > v.top {
		location = fmt.Sprintf("rows %d-%d/%d", v.top+1, end, len(v.list.Rows))
	}
	page.Trail = append(append([]string(nil), v.list.Trail...), fmt.Sprintf(
		"1s refresh · %s · wheel/↑↓/PgUp/PgDn/Home/End · q/Esc/Ctrl-C exits", location))
	return page
}

func terminalHeight(w io.Writer) int {
	file, ok := w.(*os.File)
	if !ok {
		return 0
	}
	_, height, err := term.GetSize(int(file.Fd()))
	if err != nil {
		return 0
	}
	return height
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
