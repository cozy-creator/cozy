package records

import (
	"database/sql"
	"net/url"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// OpenReadOnly opens an existing database for reading. It never creates, initializes,
// changes or waits on a lock: a missing file, a busy database, or one lacking a table or
// column this Creator reads is refused at once. Extra tables, columns and indexes and
// another schema number are fine.
func OpenReadOnly(path string) (*Store, *exit.Error) {
	name := (&url.URL{Scheme: "file", Path: path, RawQuery: "mode=ro&_pragma=query_only(1)"}).String()
	db, err := sql.Open("sqlite", name)
	if err != nil {
		return nil, exit.Internalf("cannot open the local records database %s: %s", path, err)
	}
	db.SetMaxOpenConns(1)
	required, err := requiredCurrentShape()
	if err != nil {
		db.Close()
		return nil, exit.Internalf("cannot derive current records schema: %s", err)
	}
	lacking, err := missing(db, required)
	if err != nil {
		db.Close()
		return nil, exit.Internalf("cannot inspect the records schema in %s: %s", path, err)
	}
	if len(lacking) > 0 {
		db.Close()
		return nil, exit.Named(exit.Conflict, "records_schema_mismatch",
			"records database lacks %s", strings.Join(lacking, ", "))
	}
	return &Store{db: db}, nil
}
