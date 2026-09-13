package cli

import (
	"bufio"
	"context"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"

	"github.com/muesli/cancelreader"
	"golang.org/x/term"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Every live screen borrows one terminal input session. Keys become watch
// interrupts or optional table navigation; they are never echoed or submitted.
// The cancellable reader is joined before restoring the caller's terminal state.
func liveSignals(ctx *Context, navigation chan<- listNavigation) (chan os.Signal, func(), bool, *exit.Error) {
	interrupts := make(chan os.Signal, 2)
	signal.Notify(interrupts, syscall.SIGINT, syscall.SIGTERM)
	stop := func() { signal.Stop(interrupts) }
	if ctx.Mode().JSON || !ctx.Mode().Color {
		return interrupts, stop, false, nil
	}
	fd := int(os.Stdin.Fd()) //cozy:stdin-value live terminal controls, never a prompt
	if !term.IsTerminal(fd) {
		return interrupts, stop, false, nil
	}
	state, err := term.GetState(fd)
	if err != nil {
		stop()
		return nil, func() {}, false, exit.As(err)
	}
	input, err := cancelreader.NewReader(os.Stdin) //cozy:stdin-value live terminal controls
	if err == nil {
		err = setLiveInputMode(fd)
	}
	if err != nil {
		if input != nil {
			_ = input.Close()
		}
		_ = term.Restore(fd, state)
		stop()
		return nil, func() {}, false, exit.As(err)
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		readListInput(bufio.NewReader(input), func() {
			select {
			case interrupts <- os.Interrupt:
			default:
			}
		}, navigation)
	}()
	var once sync.Once
	return interrupts, func() {
		once.Do(func() {
			// CancelReader owns its wakeup handles, never stdin. A platform
			// fallback may not join a blocked read; Close still releases its handles.
			if input.Cancel() {
				<-done
			}
			_ = input.Close()
			_ = term.Restore(fd, state)
			stop()
		})
	}, true, nil
}

func liveWatchContext(ctx *Context, parent context.Context, navigation chan<- listNavigation) (context.Context, func(), bool, *exit.Error) {
	interrupts, restore, interactive, problem := liveSignals(ctx, navigation)
	if problem != nil {
		return nil, restore, false, problem
	}
	watch, cancel := context.WithCancel(parent)
	go func() {
		select {
		case <-interrupts:
			cancel()
		case <-watch.Done():
		}
	}()
	return watch, func() { cancel(); restore() }, interactive, nil
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
