package flock

import (
	"context"
	"os"
	"time"
)

// Wait takes the same live-process lock as Exclusive, but waits for the current
// holder. It is used for content-addressed transfers: one process moves bytes while
// concurrent callers wait, then re-read the durable TensorFS release row it committed.
func Wait(ctx context.Context, file *os.File) error {
	for {
		if err := Exclusive(file); err == nil {
			return nil
		}
		timer := time.NewTimer(100 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
}
