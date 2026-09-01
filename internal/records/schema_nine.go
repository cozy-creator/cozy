package records

// schemaNineModelProduction is immutable migration evidence for the last released v9
// physical shape. It is never created by current code; v10 verifies these exact tables,
// drops them, and preserves every ordinary package/request/event/rental row.
var schemaNineModelProduction = []string{`
CREATE TABLE IF NOT EXISTS model_productions (
  id          TEXT PRIMARY KEY,
  plan_digest TEXT NOT NULL,
  plan        BLOB NOT NULL,
  state       TEXT NOT NULL,
  step_index  INTEGER NOT NULL DEFAULT 0,
  rental_id   TEXT NOT NULL DEFAULT '',
  selected_sku TEXT NOT NULL DEFAULT '',
  cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)),
  safe_code   TEXT NOT NULL DEFAULT '',
  safe_detail TEXT NOT NULL DEFAULT '',
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
)`, `
CREATE TABLE IF NOT EXISTS model_production_source_files (
  operation_id TEXT NOT NULL REFERENCES model_productions(id),
  selection_digest TEXT NOT NULL,
  member TEXT NOT NULL,
  object_id TEXT NOT NULL,
  length INTEGER NOT NULL CHECK(length>0),
  capability_revision INTEGER NOT NULL DEFAULT 0,
  state TEXT NOT NULL DEFAULT 'pending',
  transferred_bytes INTEGER NOT NULL DEFAULT 0,
  safe_code TEXT NOT NULL DEFAULT '',
  safe_detail TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(operation_id,member)
)`, `
CREATE TABLE IF NOT EXISTS model_production_sources (
  operation_id TEXT NOT NULL REFERENCES model_productions(id),
  slot TEXT NOT NULL,
  profile TEXT NOT NULL,
  manifest_id TEXT NOT NULL DEFAULT '',
  manifest_length INTEGER NOT NULL DEFAULT 0 CHECK(manifest_length>=0),
  checkpoint_evidence BLOB NOT NULL DEFAULT x'',
  PRIMARY KEY(operation_id,slot)
)`, `
CREATE TABLE IF NOT EXISTS model_production_steps (
  operation_id TEXT NOT NULL REFERENCES model_productions(id),
  step_index INTEGER NOT NULL,
  step_name TEXT NOT NULL,
  request_id TEXT NOT NULL DEFAULT '',
  state TEXT NOT NULL DEFAULT 'pending',
  PRIMARY KEY(operation_id,step_index),
  UNIQUE(operation_id,step_name)
)`, `
CREATE TABLE IF NOT EXISTS model_production_artifacts (
  operation_id TEXT NOT NULL REFERENCES model_productions(id),
  step_name TEXT NOT NULL,
  output_slot TEXT NOT NULL,
  request_id TEXT NOT NULL,
  attempt INTEGER NOT NULL,
  invocation_digest TEXT NOT NULL,
  transaction_id TEXT NOT NULL,
  writer_generation INTEGER NOT NULL CHECK(writer_generation>0),
  receipt_digest TEXT NOT NULL,
  receipt BLOB NOT NULL,
  manifest_id TEXT NOT NULL,
  manifest_length INTEGER NOT NULL CHECK(manifest_length>0),
  checkpoint_evidence BLOB NOT NULL,
  publication_id TEXT NOT NULL DEFAULT '',
  state TEXT NOT NULL DEFAULT 'received',
  PRIMARY KEY(operation_id,step_name,output_slot),
  UNIQUE(request_id,attempt,output_slot)
)`, `
CREATE TABLE IF NOT EXISTS model_production_objects (
  operation_id TEXT NOT NULL,
  step_name TEXT NOT NULL,
  output_slot TEXT NOT NULL,
  object_id TEXT NOT NULL,
  length INTEGER NOT NULL CHECK(length>0),
  source_ref TEXT NOT NULL,
  transfer_operation_id TEXT NOT NULL DEFAULT '',
  grant_revision INTEGER NOT NULL DEFAULT 0,
  update_sequence INTEGER NOT NULL DEFAULT 0,
  state TEXT NOT NULL DEFAULT 'pending',
  transferred_bytes INTEGER NOT NULL DEFAULT 0,
  safe_code TEXT NOT NULL DEFAULT '',
  safe_detail TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(operation_id,step_name,output_slot,object_id),
  FOREIGN KEY(operation_id,step_name,output_slot)
    REFERENCES model_production_artifacts(operation_id,step_name,output_slot)
)`}

// SchemaNineMigrationDDL returns a copy of the immutable retired table shape so
// migration/recovery product tests can build a real exact v9 database.
func SchemaNineMigrationDDL() []string {
	return append([]string(nil), schemaNineModelProduction...)
}
