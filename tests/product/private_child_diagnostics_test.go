package producttest

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"
	"time"
)

// Keep the real fixture observable on a cold CI worker. This only reads the
// fixture's records and bounded logs; it never cancels work or changes a deadline.
func tracePrivateChildWait(t *testing.T, root string) func() {
	t.Helper()
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan struct{})
	go func() {
		defer close(done)
		tick := time.NewTicker(30 * time.Second)
		defer tick.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-tick.C:
				tracePrivateChildState(t, ctx, root)
			}
		}
	}()
	return func() { cancel(); <-done }
}

func tracePrivateChildState(t *testing.T, parent context.Context, root string) {
	ctx, cancel := context.WithTimeout(parent, 5*time.Second)
	defer cancel()
	db, err := sql.Open("sqlite", "file:"+filepath.Join(root, "creator.sqlite")+"?mode=ro")
	if err != nil {
		t.Logf("private composition diagnostic: %v", err)
		return
	}
	defer db.Close()
	for _, query := range []struct{ name, sql string }{
		{"requests", `SELECT json_group_array(json_object('id',id,'state',state,'ordinal',ordinal,'worker',worker,'parent',parent_request_id)) FROM (SELECT * FROM requests LIMIT 20)`},
		{"attempts", `SELECT json_group_array(json_object('request',request_id,'attempt',attempt,'state',state,'instance',instance_id,'terminal',terminal_status,'cause',terminal_cause,'message',safe_message)) FROM (SELECT * FROM attempts LIMIT 20)`},
		{"workers", `SELECT json_group_array(json_object('instance',instance_id,'pid',pid,'state',state,'devices',devices)) FROM (SELECT * FROM worker_processes LIMIT 20)`},
	} {
		var value string
		if err := db.QueryRowContext(ctx, query.sql).Scan(&value); err != nil {
			t.Logf("private composition %s: %v", query.name, err)
		} else {
			t.Logf("private composition %s: %s", query.name, value)
		}
	}
	t.Logf("private composition daemon: %s", privateChildLogTail(filepath.Join(root, "daemon.log")))
	logs, _ := filepath.Glob(filepath.Join(root, "workers", "*", "*.log"))
	for _, path := range logs {
		t.Logf("private composition worker %s: %s", filepath.Base(filepath.Dir(path)), privateChildLogTail(path))
	}
}

func privateChildLogTail(path string) string {
	value := tail(path)
	if len(value) > 8192 {
		value = value[len(value)-8192:]
	}
	return value
}
