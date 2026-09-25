package records

import (
	"database/sql"
	"net/url"

	"github.com/cozy-creator/cozy/internal/exit"
)

// OpenReadOnly opens an existing database of exactly the current schema for reading.
// It never creates, initializes, migrates or waits on a lock: a missing file, another
// schema generation, or a busy database is refused at once.
func OpenReadOnly(path string) (*Store, *exit.Error) {
	name := (&url.URL{Scheme: "file", Path: path, RawQuery: "mode=ro&_pragma=query_only(1)"}).String()
	db, err := sql.Open("sqlite", name)
	if err != nil {
		return nil, exit.Internalf("cannot open the local records database %s: %s", path, err)
	}
	db.SetMaxOpenConns(1)
	version, err := databaseVersion(db)
	if err != nil {
		db.Close()
		return nil, exit.Internalf("cannot read the records schema version in %s: %s", path, err)
	}
	if version != schemaVersion {
		db.Close()
		return nil, exit.Named(exit.Conflict, "records_schema_mismatch",
			"records database has schema %d; this Creator reads schema %d", version, schemaVersion)
	}
	return &Store{db: db}, nil
}
