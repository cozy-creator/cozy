package records

import (
	"context"
	"database/sql"
	"errors"
	"strconv"
	"time"

	"modernc.org/sqlite"
	sqlite3 "modernc.org/sqlite/lib"
)

// beginRentalObservation waits for the local writer before recording a Hub fact.
// Nothing in the transaction, and no paid request, is retried. SQLite's busy
// handler can hold a thread past context cancellation, so this exclusively held
// connection uses cancellable waits between BEGIN attempts instead. Its original
// busy policy is restored before it returns to the Store's one-connection pool.
func (s *Store) beginRentalObservation(ctx context.Context) (*sql.Tx, func(), error) {
	conn, err := s.db.Conn(ctx)
	if err != nil {
		return nil, nil, err
	}
	var timeout int
	if err := conn.QueryRowContext(ctx, "PRAGMA busy_timeout").Scan(&timeout); err != nil {
		conn.Close()
		return nil, nil, err
	}
	release := func() {
		_, _ = conn.ExecContext(context.Background(), "PRAGMA busy_timeout="+strconv.Itoa(timeout))
		conn.Close()
	}
	if _, err := conn.ExecContext(ctx, "PRAGMA busy_timeout=0"); err != nil {
		release()
		return nil, nil, err
	}
	for {
		tx, err := conn.BeginTx(ctx, nil)
		if err == nil {
			return tx, release, nil
		}
		var busy *sqlite.Error
		if !errors.As(err, &busy) || busy.Code() != sqlite3.SQLITE_BUSY {
			release()
			return nil, nil, err
		}
		timer := time.NewTimer(100 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			release()
			return nil, nil, ctx.Err()
		case <-timer.C:
		}
	}
}
