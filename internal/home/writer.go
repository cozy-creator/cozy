package home

import (
	"fmt"
	"io"
	"os"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/flock"
)

// Writer is a claim on the install root: exclusive for install mutations and sweeps,
// shared for captures, which touch only their own new installs.
type Writer struct{ file *os.File }

// LockWriter claims the root exclusively without waiting; daemon housekeeping defers instead.
func LockWriter(layout Layout) (*Writer, *exit.Error) {
	file, problem := openWriter(layout)
	if problem != nil {
		return nil, problem
	}
	if err := flock.Exclusive(file); err != nil {
		file.Close()
		return nil, exit.New(exit.Conflict, "another Cozy writer holds %s", layout.Lock).
			WithRemedy("wait for the current local mutation to finish")
	}
	return &Writer{file: file}, nil
}

// WaitWriter claims the root for a command, waiting while a live process holds a conflicting
// claim. The kernel drops a dead holder's claim, so a crashed command never strands it.
func WaitWriter(layout Layout, shared bool, notice io.Writer) (*Writer, *exit.Error) {
	file, problem := openWriter(layout)
	if problem != nil {
		return nil, problem
	}
	try, wait := flock.Exclusive, flock.Block
	if shared {
		try, wait = flock.Shared, flock.BlockShared
	}
	if try(file) != nil {
		fmt.Fprintf(notice, "waiting for another Cozy command to release %s\n", layout.Lock)
		if err := wait(file); err != nil {
			file.Close()
			return nil, exit.Internalf("cannot claim the writer lock %s: %s", layout.Lock, err)
		}
	}
	return &Writer{file: file}, nil
}

func openWriter(layout Layout) (*os.File, *exit.Error) {
	file, err := os.OpenFile(layout.Lock, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, exit.Internalf("cannot open the writer lock %s: %s", layout.Lock, err)
	}
	return file, nil
}

func (w *Writer) Unlock() {
	if w == nil || w.file == nil {
		return
	}
	_ = flock.Release(w.file)
	_ = w.file.Close()
	w.file = nil
}
