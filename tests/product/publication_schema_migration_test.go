package producttest

import (
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

var publicationMigrationCopy = flag.String("publication-migration-copy", "", "private stopped-owner schema23 snapshot; only a fresh test copy is migrated")

func databaseRows(t *testing.T, path string) map[string]string {
	t.Helper()
	db, err := sql.Open("sqlite", "file:"+path+"?mode=ro")
	must(t, err)
	defer db.Close()
	names, err := db.Query(`SELECT name FROM sqlite_master WHERE type='table' ORDER BY name`)
	must(t, err)
	var tables []string
	for names.Next() {
		var name string
		must(t, names.Scan(&name))
		tables = append(tables, name)
	}
	must(t, names.Close())
	out := map[string]string{}
	for _, name := range tables {
		query := `SELECT * FROM "` + name + `"`
		identity := name
		if name == "request_model_checkpoints" {
			var changed int
			must(t, db.QueryRow(`SELECT count(*) FROM request_model_checkpoints WHERE kind<>'source' OR length(subject)<>0 OR attempt<>0`).Scan(&changed))
			if changed != 0 {
				t.Fatal("migration invented weights authority")
			}
			query = `SELECT request_id,slot,worker_boot_id,observed,acknowledged,grant_revision FROM request_model_checkpoints`
			identity = "request_model_source_checkpoints"
		}
		if name == "request_model_checkpoint_publications" {
			identity = "request_model_source_publications"
		}
		rows, err := db.Query(query)
		must(t, err)
		columns, err := rows.Columns()
		must(t, err)
		var encoded []string
		for rows.Next() {
			values := make([]any, len(columns))
			ptrs := make([]any, len(columns))
			for i := range values {
				ptrs[i] = &values[i]
			}
			must(t, rows.Scan(ptrs...))
			// Schema 25 appends opt-in retention and retry lineage. Compare every
			// pre-existing cell and require safe defaults for ordinary requests.
			preserved := make([]any, 0, len(values))
			for i, column := range columns {
				if name == "requests" && column == "retain_work" {
					if values[i] != int64(0) {
						t.Fatal("migration changed a legacy request's retention policy")
					}
					continue
				}
				if name == "requests" && (column == "retry_of" || column == "reuse_scope") {
					if values[i] != "" {
						t.Fatal("migration invented retry lineage for a legacy request")
					}
					continue
				}
				if name == "requests" && column == "control_revision" {
					if values[i] != int64(0) {
						t.Fatal("migration invented lifecycle changes for a legacy request")
					}
					continue
				}
				preserved = append(preserved, values[i])
			}
			raw, err := json.Marshal(preserved)
			must(t, err)
			encoded = append(encoded, string(raw))
		}
		must(t, rows.Close())
		sort.Strings(encoded)
		raw, err := json.Marshal(encoded)
		must(t, err)
		out[identity] = fmt.Sprintf("%d:%x", len(encoded), sha256.Sum256(raw))
	}
	return out
}

func TestPublicationCancellationMigratesPrivateCopyWithoutChangingRows(t *testing.T) {
	if *publicationMigrationCopy == "" {
		t.Skip("requires the private captured schema23 owner database")
	}
	original, err := os.ReadFile(*publicationMigrationCopy)
	must(t, err)
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	must(t, os.WriteFile(path, original, 0600))
	before := databaseRows(t, path)
	if store, problem := records.Open(path); problem == nil {
		store.Close()
		t.Fatal("ordinary reader unexpectedly migrated schema23")
	}
	store, problem := records.OpenForDaemon(path, filepath.Join(t.TempDir(), "triage"))
	fatal(t, problem)
	store.Close()
	after := databaseRows(t, path)
	if len(before) != len(after) {
		t.Fatal("migration changed table roster")
	}
	for name, value := range before {
		if after[name] != value {
			t.Fatalf("migration changed rows in %s", name)
		}
	}
	db, err := sql.Open("sqlite", path)
	must(t, err)
	defer db.Close()
	var version int
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	if version != 26 {
		t.Fatal("migration did not stamp schema26")
	}
	fk, err := db.Query(`PRAGMA foreign_key_check`)
	must(t, err)
	if fk.Next() {
		t.Fatal("migration broke a foreign key")
	}
	must(t, fk.Close())
	still, err := os.ReadFile(*publicationMigrationCopy)
	must(t, err)
	if sha256.Sum256(still) != sha256.Sum256(original) {
		t.Fatal("source snapshot changed")
	}
	t.Logf("schema23→26 private copy preserves all rows in %d tables, including source heads/ACKs/revisions; foreign keys valid; original snapshot unchanged", len(before))
}
