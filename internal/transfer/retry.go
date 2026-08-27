package transfer

import "time"

// retryStorage is the ONE storage-edge retry shape (cl-028): `attempts` tries with
// linear backoff. Both directions of the transfer plane — publish's presigned uploads
// and fetch's ranged downloads — carried their own copies of this loop, and two copies
// of one bound drift. op answers (retryable, err): a nil err ends the loop, a
// non-retryable err returns at once, and an exhausted budget returns the LAST err with
// exhausted=true so the caller wraps it under its own remedy.
func retryStorage(op func(try int) (retryable bool, err error)) (exhausted bool, err error) {
	var last error
	for try := 0; try < attempts; try++ {
		retryable, err := op(try)
		if err == nil {
			return false, nil
		}
		if !retryable {
			return false, err
		}
		last = err
		time.Sleep(time.Duration(try+1) * time.Second)
	}
	return true, last
}
