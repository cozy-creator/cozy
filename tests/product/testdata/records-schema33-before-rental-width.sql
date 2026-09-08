-- Exact schema exported from the retained PR400 task database; no user rows.
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
  triage_bundle    BLOB    NOT NULL DEFAULT x'',
  terminal_body    BLOB,
  dispatched_at    TEXT    NOT NULL,
  accepted_at      TEXT    NOT NULL DEFAULT '',
  closed_at        TEXT    NOT NULL DEFAULT '',
  media_cleaned    INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (request_id, attempt)
);

CREATE TABLE byte_outputs (
 request_id TEXT NOT NULL, attempt INTEGER NOT NULL, output_id TEXT NOT NULL,
 digest TEXT NOT NULL,length INTEGER NOT NULL,mime_type TEXT NOT NULL,
 producer_root_id TEXT NOT NULL,receipt_digest TEXT NOT NULL,manifest_id TEXT NOT NULL,
 manifest_length INTEGER NOT NULL,content_bytes INTEGER NOT NULL,
 PRIMARY KEY(request_id,attempt,output_id),
 FOREIGN KEY(request_id,attempt) REFERENCES attempts(request_id,attempt)
);

CREATE TABLE installs (
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
  package_interface TEXT    NOT NULL,
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

CREATE TABLE native_artifact_retentions (
 artifact_kind TEXT NOT NULL DEFAULT 'derived' CHECK(artifact_kind IN ('derived','tree')),
 producer_attempt INTEGER NOT NULL DEFAULT 0, producer_output_id TEXT NOT NULL DEFAULT '',content_bytes INTEGER NOT NULL DEFAULT 0,
 consumer_id TEXT NOT NULL, parent_request_id TEXT NOT NULL REFERENCES requests(id),
 kind TEXT NOT NULL CHECK(kind IN ('input','result','effect')),slot TEXT NOT NULL,
 producer_id TEXT NOT NULL, manifest_id TEXT NOT NULL, manifest_length INTEGER NOT NULL,
 receipt_digest TEXT NOT NULL, transaction_id TEXT NOT NULL, owner_request_id TEXT NOT NULL, owner_worker TEXT NOT NULL,
 retention_id TEXT NOT NULL UNIQUE, instance_id TEXT NOT NULL DEFAULT '',worker_boot_id TEXT NOT NULL DEFAULT '',
 state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','held','releasing','released')),
 PRIMARY KEY(consumer_id,kind,slot)
);

CREATE TABLE native_calls (
 id TEXT PRIMARY KEY, parent_request_id TEXT NOT NULL REFERENCES requests(id),
 call_index INTEGER NOT NULL CHECK(call_index>=0 AND call_index<32),
 kind TEXT NOT NULL CHECK(kind IN ('source','effect')), operation TEXT NOT NULL,
 intent_digest TEXT NOT NULL, request BLOB NOT NULL, frozen BLOB NOT NULL DEFAULT x'',
 state TEXT NOT NULL CHECK(state IN ('accepted','frozen','executing','succeeded','failed','canceled','stopped')),
 result BLOB NOT NULL DEFAULT x'', native_receipt BLOB NOT NULL DEFAULT x'',
 safe_code TEXT NOT NULL DEFAULT '', worker TEXT NOT NULL DEFAULT '',
 instance_id TEXT NOT NULL DEFAULT '', worker_boot_id TEXT NOT NULL DEFAULT '',
 parent_attempt INTEGER NOT NULL DEFAULT 0,
 cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)),
 UNIQUE(parent_request_id,call_index)
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

CREATE TABLE package_events (
  seq     INTEGER PRIMARY KEY AUTOINCREMENT,
  package TEXT    NOT NULL,
  type    TEXT    NOT NULL,
  payload TEXT    NOT NULL,
  at      TEXT    NOT NULL
);

CREATE TABLE pins (
  package     TEXT    NOT NULL,
  major        INTEGER NOT NULL,
  install_id   TEXT    NOT NULL REFERENCES installs(id),
  activated_at TEXT    NOT NULL,
  PRIMARY KEY (package)
);

CREATE TABLE private_child_bindings (
 parent_install_id TEXT NOT NULL REFERENCES installs(id) ON DELETE CASCADE,
 interface_digest TEXT NOT NULL,
 module TEXT NOT NULL,
 export TEXT NOT NULL,
 child_install_id TEXT NOT NULL REFERENCES installs(id),
 local_revision_digest TEXT NOT NULL,
 entrypoint TEXT NOT NULL,
 PRIMARY KEY(parent_install_id,interface_digest,module,export)
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
  expected_worker_boot_id    TEXT NOT NULL DEFAULT '',
  ready_at          TEXT NOT NULL DEFAULT '',
  failure_code                 TEXT NOT NULL DEFAULT '',
  failure_image_digest         TEXT NOT NULL DEFAULT '',
  failure_provider             TEXT NOT NULL DEFAULT '',
  failure_provider_resource_id TEXT NOT NULL DEFAULT '',
  failure_provider_host_id     TEXT NOT NULL DEFAULT '',
  failure_provider_state       TEXT NOT NULL DEFAULT '',
  failure_container_state      TEXT NOT NULL DEFAULT ''
);

CREATE TABLE request_events (
  seq        INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id TEXT    NOT NULL REFERENCES requests(id),
  type       TEXT    NOT NULL,
  attempt    INTEGER NOT NULL DEFAULT 0,
  payload    TEXT    NOT NULL,
  at         TEXT    NOT NULL
);

CREATE TABLE request_model_checkpoint_publications (
  request_id TEXT NOT NULL REFERENCES requests(id),
  operation TEXT NOT NULL,
  objects BLOB NOT NULL,
  opened INTEGER NOT NULL DEFAULT 0,
  released INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(request_id,operation)
);

CREATE TABLE request_model_checkpoints (
  request_id TEXT NOT NULL REFERENCES requests(id),
  kind TEXT NOT NULL DEFAULT 'source' CHECK(kind IN ('source','weights')),
  slot TEXT NOT NULL,
  subject BLOB NOT NULL DEFAULT x'',
  attempt INTEGER NOT NULL DEFAULT 0 CHECK(attempt>=0),
  worker_boot_id TEXT NOT NULL,
  observed TEXT NOT NULL DEFAULT '',
  acknowledged TEXT NOT NULL DEFAULT '',
  grant_revision INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(request_id,kind,slot)
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
  worker_boot_id      TEXT NOT NULL DEFAULT '',
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
  models_worker_boot_id TEXT NOT NULL DEFAULT '',
  CHECK (state IN ('pending','materializing','materialized','finalizing','completed','failed','canceling','canceled'))
);

CREATE TABLE request_operation_lookups (
 request_id TEXT PRIMARY KEY REFERENCES requests(id),
 computation_digest TEXT NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('pending','miss','hit'))
);

CREATE TABLE request_output_exports (
  request_id       TEXT PRIMARY KEY REFERENCES requests(id),
  directory        TEXT    NOT NULL,
  outputs          TEXT    NOT NULL,
  state            TEXT    NOT NULL,
  attempts         INTEGER NOT NULL DEFAULT 0,
  error_code       TEXT    NOT NULL DEFAULT '',
  safe_error       TEXT    NOT NULL DEFAULT '',
  published_paths  TEXT    NOT NULL DEFAULT '[]',
  updated_at       TEXT    NOT NULL,
  CHECK (state IN ('pending','exporting','published','failed','skipped'))
);

CREATE TABLE request_weights_retentions (
 request_id TEXT NOT NULL REFERENCES requests(id),
 kind TEXT NOT NULL CHECK(kind IN ('input','result')),
 slot TEXT NOT NULL,
 producer_request_id TEXT NOT NULL,
 producer_attempt INTEGER NOT NULL,
 producer_output_slot TEXT NOT NULL,
 retention_id TEXT NOT NULL UNIQUE,
 instance_id TEXT NOT NULL DEFAULT '',
 worker_boot_id TEXT NOT NULL DEFAULT '',
 state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','held','releasing','released')),
 PRIMARY KEY(request_id,kind,slot),
 FOREIGN KEY(producer_request_id,producer_attempt,producer_output_slot) REFERENCES request_model_transfer_outputs(request_id,attempt,output_slot)
);

CREATE TABLE requests (
  id           TEXT PRIMARY KEY,
  idem_key     TEXT    NOT NULL UNIQUE,
  body_digest  TEXT    NOT NULL,
  package     TEXT    NOT NULL,
  entrypoint   TEXT    NOT NULL,
  plan_id      TEXT    NOT NULL,
  package_release TEXT NOT NULL DEFAULT '',
  local_package_digest TEXT NOT NULL DEFAULT '',
  local_package_uploaded_boot_id TEXT NOT NULL DEFAULT '',
  environment_digest TEXT NOT NULL DEFAULT '',
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
  machine      TEXT    NOT NULL DEFAULT '',
  rental       INTEGER NOT NULL DEFAULT 0,
  rental_required INTEGER NOT NULL DEFAULT 0,
  install_id   TEXT    REFERENCES installs(id),
  assets       TEXT    NOT NULL DEFAULT '[]',
  capture      TEXT    NOT NULL DEFAULT '',
  models       TEXT    NOT NULL DEFAULT '[]',
  weights_outputs TEXT NOT NULL DEFAULT '[]',
  retain_work INTEGER NOT NULL DEFAULT 0 CHECK(retain_work IN (0,1)),
  retry_of TEXT NOT NULL DEFAULT '',
  reuse_scope TEXT NOT NULL DEFAULT '',
  control_revision INTEGER NOT NULL DEFAULT 0 CHECK(control_revision>=0),
  parent_request_id TEXT NOT NULL DEFAULT '',
  parent_call_index INTEGER NOT NULL DEFAULT -1 CHECK(parent_call_index>=-1 AND parent_call_index<32),
  child_intent_digest TEXT NOT NULL DEFAULT '',
  child_target_digest TEXT NOT NULL DEFAULT '',
  child_reusable INTEGER NOT NULL DEFAULT 0 CHECK(child_reusable IN (0,1)),
  reused_from TEXT NOT NULL DEFAULT '',
  orchestration_directive BLOB NOT NULL DEFAULT x'',
  child_artifacts INTEGER NOT NULL DEFAULT 0 CHECK(child_artifacts IN (0,1))
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
  install_id      TEXT    REFERENCES installs(id),
  worker_id       TEXT    NOT NULL,
  devices         TEXT    NOT NULL,
	pid             INTEGER NOT NULL,
	birth           TEXT    NOT NULL,
	session_id      TEXT,
	state           TEXT    NOT NULL,
  opened_at       TEXT    NOT NULL,
  closed_at       TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX package_events_by_package ON package_events(package, seq);

CREATE UNIQUE INDEX rental_operation_remote
  ON rental_operations(rental_id) WHERE rental_id <> '';

CREATE UNIQUE INDEX rentals_machine_name
  ON rentals(machine_name) WHERE machine_name<>'';

CREATE INDEX request_events_by_request ON request_events(request_id, seq);

CREATE UNIQUE INDEX requests_parent_call ON requests(parent_request_id,parent_call_index) WHERE parent_request_id!='';

CREATE UNIQUE INDEX worker_session ON worker_processes(session_id)
  WHERE session_id IS NOT NULL;
PRAGMA user_version=33;
