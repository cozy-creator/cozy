package records

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"
	"sync"

	"modernc.org/sqlite"
	sqlite3 "modernc.org/sqlite/lib"
)

// driverName is SQLite with two changes. A statement that meets another process's write lock
// waits for it instead of failing: SQLite's locks die with the process that holds them, so
// the wait lasts exactly as long as a live writer holds the database; a caller's context
// still ends it, and busy_timeout (pragmas) is only the length of one wait round. And this
// process's own writers take turns at one gate per database, while readers run on
// connections of their own: in WAL a reader never waits on a writer, so neither a slow write
// nor one waiting out another process's lock holds back a read.
const driverName = "cozy-sqlite"

func init() { sql.Register(driverName, waitingDriver{&sqlite.Driver{}}) }

type waitingDriver struct{ *sqlite.Driver }

// writers is each database's write gate, by the name it was opened under.
var writers sync.Map

func (d waitingDriver) Open(name string) (driver.Conn, error) {
	c, err := untilUnlocked(context.Background(), func() (driver.Conn, error) { return d.Driver.Open(name) })
	if err != nil {
		return nil, err
	}
	gate, _ := writers.LoadOrStore(name, &sync.Mutex{})
	return &waitingConn{sqliteConn: c.(sqliteConn), gate: gate.(*sync.Mutex)}, nil
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

// waitingConn is one connection. A transaction holds the database's write gate from begin to
// its end (every begin is IMMEDIATE, a writer's), and so does a statement executed outside one.
type waitingConn struct {
	sqliteConn
	gate *sync.Mutex
	inTx bool
}

func (c *waitingConn) Ping(ctx context.Context) error {
	_, err := untilUnlocked(ctx, func() (struct{}, error) { return struct{}{}, c.sqliteConn.Ping(ctx) })
	return err
}

func (c *waitingConn) BeginTx(ctx context.Context, opts driver.TxOptions) (driver.Tx, error) {
	c.gate.Lock()
	tx, err := untilUnlocked(ctx, func() (driver.Tx, error) { return c.sqliteConn.BeginTx(ctx, opts) })
	if err != nil {
		c.gate.Unlock()
		return nil, err
	}
	c.inTx = true
	return &gatedTx{Tx: tx, conn: c}, nil
}

func (c *waitingConn) ExecContext(ctx context.Context, query string, args []driver.NamedValue) (driver.Result, error) {
	if !c.inTx {
		c.gate.Lock()
		defer c.gate.Unlock()
	}
	return untilUnlocked(ctx, func() (driver.Result, error) { return c.sqliteConn.ExecContext(ctx, query, args) })
}

func (c *waitingConn) QueryContext(ctx context.Context, query string, args []driver.NamedValue) (driver.Rows, error) {
	return untilUnlocked(ctx, func() (driver.Rows, error) { return c.sqliteConn.QueryContext(ctx, query, args) })
}

type gatedTx struct {
	driver.Tx
	conn *waitingConn
}

func (t *gatedTx) Commit() error   { defer t.end(); return t.Tx.Commit() }
func (t *gatedTx) Rollback() error { defer t.end(); return t.Tx.Rollback() }

func (t *gatedTx) end() {
	if t.conn != nil {
		t.conn.inTx = false
		t.conn.gate.Unlock()
		t.conn = nil
	}
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
