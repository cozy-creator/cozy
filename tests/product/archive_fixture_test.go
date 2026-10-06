package producttest

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
)

// Fixture documents describe stored data directly. They do not recreate an old RPC graph.
func fixtureDigest(value []byte) string {
	spelled, err := canonical.Spell(value)
	if err != nil {
		panic(err)
	}
	return spelled
}

func fixtureBytes(value map[string]any) ([]byte, error) {
	raw, err := json.Marshal(fixtureValue(value, false))
	if err != nil {
		return nil, err
	}
	return canonical.NormalizeJCS(raw)
}

func fixtureIdentity(value map[string]any) ([]byte, []byte, error) {
	raw, err := fixtureBytes(value)
	return raw, canonical.Digest(raw), err
}

func fixtureValue(value any, nested bool) any {
	switch value := value.(type) {
	case map[string]any:
		out := map[string]any{}
		format, _ := value["_format"].(string)
		for key, field := range value {
			if key == "_format" {
				continue
			}
			// Default scalar fields were absent in the stored profile. Required repeated
			// fields are explicit, including when empty.
			if field == nil || fmt.Sprint(field) == "" || fmt.Sprint(field) == "0" || field == false {
				continue
			}
			if values, ok := field.([]any); ok && len(values) == 0 &&
				!(strings.HasSuffix(format, ".Placement/1") && (key == "entrypoints" || key == "models") ||
					strings.HasSuffix(format, ".Slot/1") && (key == "components" || key == "stamps") ||
					strings.HasSuffix(format, ".Stamp/1") && key == "values") {
				continue
			}
			out[key] = fixtureValue(field, true)
		}
		if !nested && format != "" {
			out["format"] = format
		}
		return out
	case []any:
		out := make([]any, len(value))
		for i, entry := range value {
			out[i] = fixtureValue(entry, true)
		}
		return out
	case []byte:
		return base64.StdEncoding.EncodeToString(value)
	}
	return value
}
