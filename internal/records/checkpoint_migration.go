package records

import (
	"database/sql"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Prior DDL derives from the current shape; it is not another authority.
func priorCheckpointSchema() string {
	prior := strings.Replace(modelCheckpointSchema, "request_model_checkpoints", "request_model_source_checkpoints", 1)
	prior = strings.Replace(prior, "  kind TEXT NOT NULL DEFAULT 'source' CHECK(kind IN ('source','weights')),\n", "", 1)
	prior = strings.Replace(prior, "  subject BLOB NOT NULL DEFAULT x'',\n  attempt INTEGER NOT NULL DEFAULT 0 CHECK(attempt>=0),\n", "", 1)
	prior = strings.Replace(prior, "  observed TEXT NOT NULL DEFAULT '',", "  observed TEXT NOT NULL,", 1)
	return strings.Replace(prior, "PRIMARY KEY(request_id,kind,slot)", "PRIMARY KEY(request_id,slot)", 1)
}

func migrateCheckpoints(tx *sql.Tx, path string) *exit.Error {
	for _, statement := range []string{
		modelCheckpointSchema,
		`INSERT INTO request_model_checkpoints(request_id,slot,worker_boot_id,observed,acknowledged,grant_revision)
   SELECT request_id,slot,worker_boot_id,observed,acknowledged,grant_revision FROM request_model_source_checkpoints`,
		modelCheckpointPublicationSchema,
		`INSERT INTO request_model_checkpoint_publications(request_id,operation,objects,opened,released)
   SELECT request_id,operation,objects,opened,released FROM request_model_source_publications`,
		`DROP TABLE request_model_source_checkpoints`,
		`DROP TABLE request_model_source_publications`,
	} {
		if _, err := tx.Exec(statement); err != nil {
			return exit.Internalf("cannot migrate checkpoint custody in %s: %s", path, err)
		}
	}
	return nil
}
