package install

import (
	"os"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/flock"
	"github.com/cozy-creator/cozy-creator/internal/home"
)

// Writer is the single-writer claim held by local install mutations.
type Writer struct{ file *os.File }

func Lock(layout home.Layout) (*Writer, *exit.Error) {
	file, err := os.OpenFile(layout.Lock, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, exit.Internalf("cannot open the writer lock %s: %s", layout.Lock, err)
	}
	if err := flock.Exclusive(file); err != nil {
		file.Close()
		return nil, exit.New(exit.Conflict, "another Cozy writer holds %s", layout.Lock).
			WithRemedy("wait for the current local mutation to finish")
	}
	return &Writer{file: file}, nil
}

func (w *Writer) Unlock() {
	if w == nil || w.file == nil {
		return
	}
	_ = flock.Release(w.file)
	_ = w.file.Close()
	w.file = nil
}
