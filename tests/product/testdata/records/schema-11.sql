-- The released schema-11 records shape, dumped from that schema's own sqlite_master.
-- Tables first so the fixture replays in one script; records.Open refuses anything that
-- is not this exact shape, so a drifted copy fails the migration test rather than passing it.
CREATE TABLE attempts (
  request_id       TEXT    NOT NULL REFERENCES requests(id),
  attempt          INTEGER NOT NULL,
  attempt_key      TEXT    NOT NULL UNIQUE,
  instance_id      TEXT    NOT NULL REFERENCES worker_processes(instance_id),
  session_id       TEXT    NOT NULL,
  invocation_digest TEXT    NOT NULL,
  invocation        BLOB    NOT NULL,
  weights_outputs  TEXT    NOT NULL DEFAULT '[]',
  state            TEXT    NOT NULL,
  plan_digest      TEXT    NOT NULL DEFAULT '',
  construction     TEXT    NOT NULL DEFAULT '',
  plan_summary     TEXT    NOT NULL DEFAULT '',
  terminal_id      TEXT    NOT NULL DEFAULT '',
  terminal_digest  TEXT    NOT NULL DEFAULT '',
  terminal_status  TEXT    NOT NULL DEFAULT '',
  terminal_cause   TEXT    NOT NULL DEFAULT '',
  safe_message     TEXT    NOT NULL DEFAULT '',
  triage_subject   TEXT    NOT NULL DEFAULT '',
  triage_digest    TEXT    NOT NULL DEFAULT '',
  triage_length    INTEGER NOT NULL DEFAULT 0,
  triage_path      TEXT    NOT NULL DEFAULT '',
  terminal_body    BLOB,
  dispatched_at    TEXT    NOT NULL,
  accepted_at      TEXT    NOT NULL DEFAULT '',
  closed_at        TEXT    NOT NULL DEFAULT '',
  media_cleaned    INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (request_id, attempt)
);

CREATE TABLE install_generations (
  id            TEXT PRIMARY KEY,
  package      TEXT    NOT NULL,
  major         INTEGER NOT NULL,
  version       TEXT    NOT NULL,
  source_kind   TEXT    NOT NULL,
  source_ref    TEXT    NOT NULL,
  source_digest TEXT    NOT NULL,
  verified      INTEGER NOT NULL,
  dir           TEXT    NOT NULL,
  python        TEXT    NOT NULL,
  runtime       TEXT    NOT NULL DEFAULT '',
  project_dir   TEXT    NOT NULL DEFAULT '',
  uv            TEXT    NOT NULL,
  lock_digest   TEXT    NOT NULL,
  platform      TEXT    NOT NULL,
  extra         TEXT    NOT NULL,
  packages      INTEGER NOT NULL,
  closure       TEXT    NOT NULL,
  package_descriptor TEXT    NOT NULL,
  placement_set_digest TEXT  NOT NULL DEFAULT '',
  bytes_excl    INTEGER NOT NULL,
  bytes_shared  INTEGER NOT NULL,
  created_at    TEXT    NOT NULL
);

CREATE TABLE job_checkpoints (
  request_id    TEXT    NOT NULL,
  attempt       INTEGER NOT NULL,
  operation_key TEXT    NOT NULL,
  logical_key   TEXT    NOT NULL,
  content_digest TEXT   NOT NULL,
  receipt_id    TEXT    NOT NULL,
  outcome       TEXT    NOT NULL,
  recorded_at   TEXT    NOT NULL,
  PRIMARY KEY (request_id, attempt, operation_key, logical_key),
  FOREIGN KEY (request_id, attempt) REFERENCES attempts(request_id, attempt)
);

CREATE TABLE outputs (
  request_id TEXT    NOT NULL,
  attempt    INTEGER NOT NULL,
  output_id  TEXT    NOT NULL,
  media_id   TEXT    NOT NULL UNIQUE,
  path       TEXT    NOT NULL,
  digest     TEXT    NOT NULL,
  length     INTEGER NOT NULL,
  mime_type  TEXT    NOT NULL,
  PRIMARY KEY (request_id, attempt, output_id),
  FOREIGN KEY (request_id, attempt) REFERENCES attempts(request_id, attempt)
);

CREATE TABLE pins (
  package     TEXT    NOT NULL,
  major        INTEGER NOT NULL,
  generation   TEXT    NOT NULL REFERENCES install_generations(id),
  activated_at TEXT    NOT NULL,
  PRIMARY KEY (package)
);

CREATE TABLE publications (
  request_id   TEXT PRIMARY KEY REFERENCES requests(id),
  attempt      INTEGER NOT NULL,
  repo         TEXT    NOT NULL,
  root         TEXT    NOT NULL,
  status       TEXT    NOT NULL,
  cause        TEXT    NOT NULL DEFAULT '',
  entries      INTEGER NOT NULL DEFAULT 0,
  bytes        INTEGER NOT NULL DEFAULT 0,
  committed_at TEXT    NOT NULL
);

CREATE TABLE rental_operations (
  operation_key    TEXT PRIMARY KEY,
  request_digest   TEXT NOT NULL,
  request_body     BLOB NOT NULL,
  hub              TEXT NOT NULL,
  reason           TEXT NOT NULL,
  hourly_rate_usd_micros INTEGER NOT NULL,
  managed_request_id TEXT NOT NULL DEFAULT '',
  rental_id        TEXT NOT NULL DEFAULT '',
  state            TEXT NOT NULL,
  created_at       TEXT NOT NULL,
  updated_at       TEXT NOT NULL
);

CREATE TABLE rentals (
  id                TEXT PRIMARY KEY,
  machine_name      TEXT NOT NULL DEFAULT '',
  sku               TEXT NOT NULL DEFAULT '',
  accelerator_model TEXT NOT NULL,
  hourly_rate_usd_micros INTEGER NOT NULL,
  managed_request_id TEXT NOT NULL DEFAULT '',
  address           TEXT NOT NULL,
  cert_path         TEXT NOT NULL,
  state             TEXT NOT NULL,
  hub               TEXT NOT NULL,
  rented_at         TEXT NOT NULL,
  media_address     TEXT NOT NULL DEFAULT '',
  expected_worker_id         TEXT NOT NULL DEFAULT '',
  expected_worker_boot_id    TEXT NOT NULL DEFAULT ''
);

CREATE TABLE request_events (
  seq        INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id TEXT    NOT NULL REFERENCES requests(id),
  type       TEXT    NOT NULL,
  attempt    INTEGER NOT NULL DEFAULT 0,
  payload    TEXT    NOT NULL,
  at         TEXT    NOT NULL
);

CREATE TABLE request_model_transfer_files (
  request_id          TEXT NOT NULL REFERENCES requests(id),
  member              TEXT NOT NULL,
  object_id           TEXT NOT NULL,
  length              INTEGER NOT NULL CHECK(length>0),
  capability_revision INTEGER NOT NULL DEFAULT 0,
  state               TEXT NOT NULL DEFAULT 'pending',
  transferred         INTEGER NOT NULL DEFAULT 0,
  safe_code           TEXT NOT NULL DEFAULT '',
  safe_detail         TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(request_id,member)
);

CREATE TABLE request_model_transfer_objects (
  request_id       TEXT NOT NULL,
  attempt          INTEGER NOT NULL,
  output_slot      TEXT NOT NULL,
  object_id        TEXT NOT NULL,
  length           INTEGER NOT NULL CHECK(length>0),
  source_ref       TEXT NOT NULL,
  operation_id     TEXT NOT NULL DEFAULT '',
  grant_revision   INTEGER NOT NULL DEFAULT 0,
  update_sequence  INTEGER NOT NULL DEFAULT 0,
  state            TEXT NOT NULL DEFAULT 'pending',
  transferred      INTEGER NOT NULL DEFAULT 0,
  safe_code        TEXT NOT NULL DEFAULT '',
  safe_detail      TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(request_id,attempt,output_slot,object_id),
  FOREIGN KEY(request_id,attempt,output_slot)
    REFERENCES request_model_transfer_outputs(request_id,attempt,output_slot)
);

CREATE TABLE request_model_transfer_outputs (
  request_id       TEXT NOT NULL REFERENCES requests(id),
  output_slot      TEXT NOT NULL,
  manifest_id      TEXT NOT NULL,
  manifest_length  INTEGER NOT NULL CHECK(manifest_length>0),
  evidence         BLOB NOT NULL,
  attempt           INTEGER NOT NULL,
  invocation_digest TEXT NOT NULL,
  transaction_id    TEXT NOT NULL,
  receipt_digest    TEXT NOT NULL,
  receipt           BLOB NOT NULL,
  final_id          TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(request_id,attempt,output_slot)
);

CREATE TABLE request_model_transfers (
  request_id       TEXT PRIMARY KEY REFERENCES requests(id),
  intent           TEXT NOT NULL,
  state            TEXT NOT NULL DEFAULT 'pending',
  models           TEXT NOT NULL DEFAULT '[]',
  checkpoints      TEXT NOT NULL DEFAULT '{}',
  error_code       TEXT NOT NULL DEFAULT '',
  safe_error       TEXT NOT NULL DEFAULT '',
  updated_at       TEXT NOT NULL,
  CHECK (state IN ('pending','materializing','materialized','finalizing','completed','failed','canceled'))
);

CREATE TABLE request_output_exports (
  request_id       TEXT PRIMARY KEY REFERENCES requests(id),
  directory        TEXT    NOT NULL,
  payload_hash     TEXT    NOT NULL,
  outputs          TEXT    NOT NULL,
  state            TEXT    NOT NULL,
  attempts         INTEGER NOT NULL DEFAULT 0,
  error_code       TEXT    NOT NULL DEFAULT '',
  safe_error       TEXT    NOT NULL DEFAULT '',
  published_paths  TEXT    NOT NULL DEFAULT '[]',
  updated_at       TEXT    NOT NULL,
  CHECK (state IN ('pending','exporting','published','failed','skipped'))
);

CREATE TABLE requests (
  id           TEXT PRIMARY KEY,
  idem_key     TEXT    NOT NULL UNIQUE,
  body_digest  TEXT    NOT NULL,
  package     TEXT    NOT NULL,
  entrypoint   TEXT    NOT NULL,
  plan_id      TEXT    NOT NULL,
  package_release TEXT NOT NULL DEFAULT '',
  package_revision_digest TEXT NOT NULL DEFAULT '',
  private_package_digest TEXT NOT NULL DEFAULT '',
  private_package_uploaded_boot_id TEXT NOT NULL DEFAULT '',
  environment_digest TEXT NOT NULL DEFAULT '',
  config_digest TEXT NOT NULL DEFAULT '',
  payload      BLOB    NOT NULL,
  outputs      TEXT    NOT NULL DEFAULT '',
  state        TEXT    NOT NULL,
  ordinal      INTEGER NOT NULL DEFAULT 0,
  requeues     INTEGER NOT NULL DEFAULT 0,
  created_at   TEXT    NOT NULL,
  kind         TEXT    NOT NULL DEFAULT 'serving',
  needs_accelerator INTEGER NOT NULL DEFAULT 0,
  org          TEXT    NOT NULL DEFAULT '',
  trees        TEXT    NOT NULL DEFAULT '',
  worker       TEXT    NOT NULL DEFAULT '',
  rental       INTEGER NOT NULL DEFAULT 0,
  rental_required INTEGER NOT NULL DEFAULT 0,
  install_id   TEXT    REFERENCES install_generations(id),
  assets       TEXT    NOT NULL DEFAULT '[]',
  models       TEXT    NOT NULL DEFAULT '[]',
  weights_outputs TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE weights_finalizations (
  request_id         TEXT    NOT NULL,
  attempt            INTEGER NOT NULL,
  instance_id        TEXT    NOT NULL,
  owner_scope        TEXT    NOT NULL,
  invocation_digest  TEXT    NOT NULL,
  output_slot        TEXT    NOT NULL,
  disposition        TEXT    NOT NULL,
  receipt_digest     TEXT    NOT NULL DEFAULT '',
  scratch_root_id    TEXT    NOT NULL DEFAULT '',
  result_outcome     TEXT    NOT NULL DEFAULT '',
  result_receipt_digest TEXT NOT NULL DEFAULT '',
  result_receipt_bytes  BLOB NOT NULL DEFAULT x'',
  recorded_at        TEXT    NOT NULL,
  completed_at       TEXT    NOT NULL DEFAULT '',
  PRIMARY KEY (request_id, invocation_digest, output_slot),
  FOREIGN KEY (request_id, attempt) REFERENCES attempts(request_id, attempt)
);

CREATE TABLE worker_processes (
  instance_id     TEXT PRIMARY KEY,
  package        TEXT    NOT NULL,
  generation      TEXT    REFERENCES install_generations(id),
  package_revision_digest      TEXT    NOT NULL,
  worker_id       TEXT    NOT NULL,
  devices         TEXT    NOT NULL,
	pid             INTEGER NOT NULL,
	birth           TEXT    NOT NULL,
	session_id      TEXT,
	state           TEXT    NOT NULL,
  opened_at       TEXT    NOT NULL,
  closed_at       TEXT    NOT NULL DEFAULT ''
);

CREATE UNIQUE INDEX rental_operation_remote
  ON rental_operations(rental_id) WHERE rental_id <> '';

CREATE UNIQUE INDEX rentals_machine_name
  ON rentals(machine_name) WHERE machine_name<>'';

CREATE INDEX request_events_by_request ON request_events(request_id, seq);

CREATE UNIQUE INDEX worker_session ON worker_processes(session_id)
  WHERE session_id IS NOT NULL;
