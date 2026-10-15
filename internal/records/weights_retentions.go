package records

const weightsRetentionsDDL = `
CREATE TABLE IF NOT EXISTS request_weights_retentions (
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
)`
