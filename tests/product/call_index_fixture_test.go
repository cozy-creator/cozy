package producttest

import (
	"database/sql"
	"strings"
	"testing"
)

// Old-schema fixtures start with the released uint32-lifetime predecessor bound.
// Preserve every row and historical index while removing schema35 and later additions.
func restorePriorCallIndexBounds(t *testing.T, db *sql.DB) {
	t.Helper()
	_, err := db.Exec(`DROP TABLE rental_runtime_updates; DROP TABLE capture_pins; DROP TRIGGER IF EXISTS machine_execution_no_local_attempt; DROP TABLE IF EXISTS machine_executions`)
	must(t, err)
	_, err = db.Exec(`DROP INDEX IF EXISTS byte_outputs_native_service`)
	must(t, err)
	var byteDDL string
	if db.QueryRow(`SELECT sql FROM sqlite_master WHERE type='table' AND name='byte_outputs'`).Scan(&byteDDL) == nil && strings.Contains(byteDDL, "native_service_id") {
		_, err = db.Exec(`ALTER TABLE byte_outputs DROP COLUMN native_service_id`)
		must(t, err)
	}
	_, err = db.Exec(`PRAGMA legacy_alter_table=ON`)
	must(t, err)
	_, err = db.Exec(`DROP TABLE IF EXISTS request_operation_contexts`)
	must(t, err)
	for _, name := range []string{"requests_active_children", "native_active_calls"} {
		_, err := db.Exec(`DROP INDEX IF EXISTS ` + name)
		must(t, err)
	}
	for _, table := range []string{"requests", "native_calls"} {
		var ddl string
		must(t, db.QueryRow(`SELECT sql FROM sqlite_master WHERE type='table' AND name=?`, table).Scan(&ddl))
		var indexes []string
		rows, err := db.Query(`SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL`, table)
		must(t, err)
		for rows.Next() {
			var text string
			must(t, rows.Scan(&text))
			indexes = append(indexes, text)
		}
		must(t, rows.Close())
		_, err = db.Exec(`ALTER TABLE ` + table + ` RENAME TO prior_call_bound`)
		must(t, err)
		ddl = strings.Replace(ddl, "  attention_kernel TEXT NOT NULL DEFAULT '',\n", "", 1)
		ddl = strings.Replace(ddl, ",\n  requested_rental TEXT NOT NULL DEFAULT ''", "", 1)
		_, err = db.Exec(strings.ReplaceAll(ddl, "call_index<4294967296", "call_index<32"))
		must(t, err)
		current, err := db.Query(`SELECT * FROM ` + table + ` LIMIT 0`)
		must(t, err)
		columns, err := current.Columns()
		must(t, err)
		must(t, current.Close())
		joined := strings.Join(columns, ",")
		_, err = db.Exec(`INSERT INTO ` + table + `(` + joined + `) SELECT ` + joined + ` FROM prior_call_bound`)
		must(t, err)
		_, err = db.Exec(`DROP TABLE prior_call_bound`)
		must(t, err)
		for _, statement := range indexes {
			_, err = db.Exec(statement)
			must(t, err)
		}
	}
}
