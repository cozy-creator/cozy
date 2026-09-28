package transfer

import (
	"context"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

// retryStorage is the ONE storage-edge retry shape (cl-028): `attempts` tries with
// linear backoff. Both directions of the transfer plane — publish's presigned uploads
// and fetch's ranged downloads — carried their own copies of this loop, and two copies
// of one bound drift. op answers (retryable, err): a nil err ends the loop, a
// non-retryable err returns at once, and an exhausted budget returns the LAST err with
// exhausted=true so the caller wraps it under its own remedy. A canceled ctx is a stop,
// not a storage failure: it ends the loop at once and is never retried.
func retryStorage(ctx context.Context, op func(try int) (retryable bool, err error)) (exhausted bool, err error) {
	var last error
	for try := 0; try < attempts; try++ {
		retryable, err := op(try)
		if err == nil {
			return false, nil
		}
		if ctx.Err() != nil {
			return false, stopped(ctx)
		}
		if !retryable {
			return false, err
		}
		last = err
		select {
		case <-ctx.Done():
			return false, stopped(ctx)
		case <-time.After(time.Duration(try+1) * time.Second):
		}
	}
	return true, last
}

func stopped(ctx context.Context) *exit.Error {
	return exit.Named(exit.Canceled, "transfer.stopped", "the transfer was stopped: %s", context.Cause(ctx))
}
