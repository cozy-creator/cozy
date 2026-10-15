package records

// A numerical context is bound once after selecting the callee worker. It is
// distinct from the captured implementation and survives pending lookup replay.
const operationContextsDDL = `CREATE TABLE IF NOT EXISTS request_operation_contexts (
 request_id TEXT PRIMARY KEY REFERENCES requests(id),
 numerical_environment_digest TEXT NOT NULL,
 computation_digest TEXT NOT NULL
)`

type OperationContext struct{ NumericalEnvironment, Key string }
