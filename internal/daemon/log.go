package daemon

import (
	"fmt"
	"os"
	"sync"
)

// LogBytes bounds the daemon log on disk. A write that would carry the file past it
// rotates first: the file becomes its own ".1" predecessor (replacing the previous one)
// and a fresh log starts, so at most two files of this size ever exist. The bound is on
// observed bytes, never on elapsed time.
const LogBytes = 32 << 20

// Log is the daemon's append-only log writer, opened once per daemon life.
type Log struct {
	mu   sync.Mutex
	path string
	file *os.File
	size int64
}

// OpenLog appends to the log at path, creating it. It carries on from the size it finds
// so a daemon restarted onto an existing log rotates at the same bound.
func OpenLog(path string) (*Log, error) {
	l := &Log{path: path}
	if err := l.open(); err != nil {
		return nil, err
	}
	return l, nil
}

func (l *Log) open() error {
	file, err := os.OpenFile(l.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return fmt.Errorf("cannot open the daemon log %s: %w", l.path, err)
	}
	info, err := file.Stat()
	if err != nil {
		file.Close()
		return fmt.Errorf("cannot read the daemon log %s: %w", l.path, err)
	}
	l.file, l.size = file, info.Size()
	return nil
}

func (l *Log) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.size > 0 && l.size+int64(len(p)) > LogBytes {
		l.file.Close()
		if err := os.Rename(l.path, l.path+".1"); err != nil && !os.IsNotExist(err) {
			return 0, fmt.Errorf("cannot rotate the daemon log %s: %w", l.path, err)
		}
		if err := l.open(); err != nil {
			return 0, err
		}
	}
	n, err := l.file.Write(p)
	l.size += int64(n)
	return n, err
}

func (l *Log) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.file.Close()
}
