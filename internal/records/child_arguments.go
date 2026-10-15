package records

import (
	"bytes"
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

const childArgumentsDDL = `CREATE TABLE IF NOT EXISTS request_child_arguments (
  request_id TEXT PRIMARY KEY REFERENCES requests(id) ON DELETE CASCADE,
  body BLOB NOT NULL CHECK(length(body)>0 AND length(body)<=49152)
)`

// ServingCallArguments separates immutable call options from the unchanged inference payload.
func ServingCallArguments(body []byte) ([]byte, map[string]json.RawMessage, *exit.Error) {
	normalized, err := canonical.NormalizeJCS(body)
	if err != nil || !bytes.Equal(body, normalized) || len(body) > 48*1024 {
		return nil, nil, exit.New(exit.Validation, "serving call must be bounded canonical JSON")
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(body, &fields) != nil || len(fields) != 2 || fields["payload"] == nil || fields["models"] == nil {
		return nil, nil, exit.New(exit.Validation, "serving call has exactly payload and models")
	}
	payload := fields["payload"]
	var models map[string]json.RawMessage
	if len(payload) == 0 || payload[0] != '{' || json.Unmarshal(fields["models"], &models) != nil || models == nil {
		return nil, nil, exit.New(exit.Validation, "serving payload and model arguments must be objects")
	}
	return payload, models, nil
}
