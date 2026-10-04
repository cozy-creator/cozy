package api

import "encoding/json"

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
	*s = JobState(document.plain)
	if len(document.Result) > 0 && string(document.Result) != "null" {
		s.Result = document.Result
	}
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
	*s = Lifecycle(document.plain)
	if len(document.Result) > 0 && string(document.Result) != "null" {
		s.Result = document.Result
	}
	return nil
}
