package api

import (
	"bytes"
	"encoding/json"
)

// Authored results must not pass through float64 while a controller reads and
// reprints state. Other advisory/status fields retain their existing typed readers.
func (s *JobState) UnmarshalJSON(data []byte) error {
	type plain JobState
	var document struct {
		plain
		Result json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(data, &document); err != nil {
		return err
	}
	result, err := applicationResult(document.Result)
	if err != nil {
		return err
	}
	*s = JobState(document.plain)
	s.Result = result
	return nil
}

func (s *Lifecycle) UnmarshalJSON(data []byte) error {
	type plain Lifecycle
	var document struct {
		plain
		Result json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(data, &document); err != nil {
		return err
	}
	result, err := applicationResult(document.Result)
	if err != nil {
		return err
	}
	*s = Lifecycle(document.plain)
	s.Result = result
	return nil
}

func applicationResult(raw json.RawMessage) (any, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	err := decoder.Decode(&value)
	return value, err
}
