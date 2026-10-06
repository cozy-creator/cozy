package archive

import (
	"encoding/base64"
	"fmt"
	"google.golang.org/protobuf/encoding/protowire"
)

// Only the fields actually read from retained records are decoded. Unknown fields are skipped.
type fields struct {
	bytes    map[protowire.Number][]byte
	integers map[protowire.Number]uint64
}

func record(data []byte, limit int, wanted ...protowire.Number) (fields, error) {
	out := fields{map[protowire.Number][]byte{}, map[protowire.Number]uint64{}}
	if len(data) == 0 || len(data) > limit {
		return out, fmt.Errorf("archived record size is invalid")
	}
	for len(data) > 0 {
		field, kind, n := protowire.ConsumeTag(data)
		if n < 0 {
			return out, protowire.ParseError(n)
		}
		data = data[n:]
		keep := false
		for _, tag := range wanted {
			keep = keep || field == tag
		}
		if !keep {
			n := protowire.ConsumeFieldValue(field, kind, data)
			if n < 0 {
				return out, protowire.ParseError(n)
			}
			data = data[n:]
			continue
		}
		switch kind {
		case protowire.BytesType:
			value, n := protowire.ConsumeBytes(data)
			if n < 0 {
				return out, protowire.ParseError(n)
			}
			out.bytes[field] = value
			data = data[n:]
		case protowire.VarintType:
			value, n := protowire.ConsumeVarint(data)
			if n < 0 {
				return out, protowire.ParseError(n)
			}
			out.integers[field] = value
			data = data[n:]
		default:
			n := protowire.ConsumeFieldValue(field, kind, data)
			if n < 0 {
				return out, protowire.ParseError(n)
			}
			data = data[n:]
		}
	}
	return out, nil
}

type Receipt struct {
	RequestID, WorkerID string
	AcceptedMS, Number  uint64
}

func ReadReceipt(raw []byte) (Receipt, error) {
	value, err := record(raw, 64<<10, 1, 5, 6, 10)
	return Receipt{string(value.bytes[1]), string(value.bytes[6]), value.integers[5], value.integers[10]}, err
}

type Terminal struct {
	Attempt        uint64
	Status         int64
	Message, Cause string
	Result         []byte
	RuntimeMS      uint64
	Unverified     []string
}

func ReadTerminal(data []byte) (*Terminal, error) {
	doc, err := Read(data, TerminalBody)
	if err != nil {
		return nil, err
	}
	if doc.Int("attempt_ordinal") < 0 {
		return nil, fmt.Errorf("archived attempt is negative")
	}
	out := &Terminal{Attempt: uint64(doc.Int("attempt_ordinal")), Status: doc.Int("status"), Message: doc.Str("safe_message")}
	code := doc.Sub("cause").Int("code")
	if len(doc.Sub("cause")) > 0 {
		if name, ok := causeNames[code]; ok {
			out.Cause = name
		} else if code != 0 {
			out.Cause = fmt.Sprintf("%d", code)
		}
	}
	if result := doc.Sub("result").Str("inline_result"); result != "" {
		out.Result, err = base64.StdEncoding.Strict().DecodeString(result)
		if err != nil {
			return nil, err
		}
	}
	metrics := doc.Sub("metrics")
	if metrics.Int("runtime_ms") < 0 {
		return nil, fmt.Errorf("archived runtime is negative")
	}
	out.RuntimeMS = uint64(metrics.Int("runtime_ms"))
	out.Unverified = metrics.Strs("unverified_fields")
	return out, nil
}
func ReadOutcome(data []byte) (*Terminal, error) {
	fields, err := record(data, 8<<20, 10)
	if err != nil {
		return nil, err
	}
	return ReadTerminal(fields.bytes[10])
}

type Release struct{ Package, Release, InstallationID string }
type Submission struct {
	Capture []byte
	Root    *Release
}

func ReadSubmission(data []byte) (Submission, error) {
	fields, err := record(data, 8<<20, 4, 13)
	if err != nil {
		return Submission{}, err
	}
	out := Submission{Capture: fields.bytes[4]}
	if raw := fields.bytes[13]; len(raw) > 0 {
		release, err := record(raw, 8<<20, 1, 2, 16)
		if err != nil {
			return out, err
		}
		out.Root = &Release{string(release.bytes[1]), string(release.bytes[2]), string(release.bytes[16])}
	}
	return out, nil
}

var causeNames = map[int64]string{
	0:  "UNSPECIFIED",
	1:  "INVALID_REQUEST",
	2:  "UNSUPPORTED_INPUT",
	3:  "LOCAL_SAFETY",
	4:  "PROTOCOL",
	5:  "CONSTRAINT_INFEASIBLE",
	6:  "AUTHOR_EXCEPTION",
	7:  "EXECUTOR_FAULT",
	8:  "GRANT_EXPIRED",
	9:  "ARTIFACT_UNFETCHABLE",
	10: "CAPABILITY_UNAVAILABLE",
	11: "CLIENT_CANCEL",
	12: "DEADLINE_EXPIRED",
	13: "DRAIN_CANCEL",
	14: "POLICY_CANCEL",
	15: "SUPERSEDED_CANCEL",
	16: "EXECUTOR_INVALIDATED",
	17: "NO_CAPACITY",
	18: "ADMISSION_EPOCH_STALE",
	19: "UNKNOWN_PLACEMENT",
	20: "PLACEMENT_NOT_DISPATCHABLE",
}
