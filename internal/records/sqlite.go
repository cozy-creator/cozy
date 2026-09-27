package records

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"

	"modernc.org/sqlite"
	sqlite3 "modernc.org/sqlite/lib"
)

// driverName is SQLite with one change: a statement that meets another writer's lock waits
// for it instead of failing. SQLite's locks die with the process that holds them, so the
// wait lasts exactly as long as a live writer holds the database; a caller's context
// still ends it. busy_timeout (pragmas) is only the length of one wait round, the
// interval at which cancellation is observed.
const driverName = "cozy-sqlite"

func init() { sql.Register(driverName, waitingDriver{&sqlite.Driver{}}) }

type waitingDriver struct{ *sqlite.Driver }

func (d waitingDriver) Open(name string) (driver.Conn, error) {
	c, err := untilUnlocked(context.Background(), func() (driver.Conn, error) { return d.Driver.Open(name) })
	if err != nil {
		return nil, err
	}
	return waitingConn{c.(sqliteConn)}, nil
}

type sqliteConn interface {
	driver.Conn
	driver.ConnBeginTx
	driver.ConnPrepareContext
	driver.ExecerContext
	driver.QueryerContext
	driver.Pinger
	driver.SessionResetter
	driver.Validator
}

type waitingConn struct{ sqliteConn }

func (c waitingConn) Ping(ctx context.Context) error {
	_, err := untilUnlocked(ctx, func() (struct{}, error) { return struct{}{}, c.sqliteConn.Ping(ctx) })
	return err
}

func (c waitingConn) BeginTx(ctx context.Context, opts driver.TxOptions) (driver.Tx, error) {
	return untilUnlocked(ctx, func() (driver.Tx, error) { return c.sqliteConn.BeginTx(ctx, opts) })
}

func (c waitingConn) ExecContext(ctx context.Context, query string, args []driver.NamedValue) (driver.Result, error) {
	return untilUnlocked(ctx, func() (driver.Result, error) { return c.sqliteConn.ExecContext(ctx, query, args) })
}

func (c waitingConn) QueryContext(ctx context.Context, query string, args []driver.NamedValue) (driver.Rows, error) {
	return untilUnlocked(ctx, func() (driver.Rows, error) { return c.sqliteConn.QueryContext(ctx, query, args) })
}

// untilUnlocked repeats a statement SQLite refused as busy; a busy statement did nothing.
// A stale read snapshot (BUSY_SNAPSHOT) cannot clear by waiting and is returned.
func untilUnlocked[T any](ctx context.Context, do func() (T, error)) (T, error) {
	for {
		value, err := do()
		var busy *sqlite.Error
		if !errors.As(err, &busy) || busy.Code()&0xff != sqlite3.SQLITE_BUSY || busy.Code() == sqlite3.SQLITE_BUSY_SNAPSHOT {
			return value, err
		}
		if ctx.Err() != nil {
			return value, ctx.Err()
		}
	}
}
