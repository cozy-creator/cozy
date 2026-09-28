package orchestrator

import (
	"encoding/json"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// VerifyByteResultRow binds a native output receipt to its typed result position.
func VerifyByteResultRow(result any, b records.ByteOutput) *exit.Error {
	value := result
	for _, part := range strings.Split(b.OutputID, ".") {
		switch current := value.(type) {
		case map[string]any:
			value = current[part]
		case []any:
			index, err := strconv.Atoi(part)
			if err != nil || index < 0 || index >= len(current) {
				return exit.New(exit.Validation, "byte result path is absent")
			}
			value = current[index]
		default:
			return exit.New(exit.Validation, "byte result path is not declared metadata")
		}
	}
	row, ok := value.(map[string]any)
	if !ok {
		return exit.New(exit.Validation, "byte result has no final asset row")
	}
	size, err := strconv.ParseInt(stringNumber(row["size_bytes"]), 10, 64)
	if err != nil || row["asset_ref"] != b.Digest || row["digest"] != b.Digest || size != b.ContentBytes {
		return exit.New(exit.Validation, "byte result row changed its final native identity")
	}
	if row["kind"] == "tree" {
		if b.MimeType != "application/vnd.cozy.tree-manifest" || b.Digest != b.ManifestID || b.Length != b.ManifestLength {
			return exit.New(exit.Validation, "Tree result confuses manifest length with content size")
		}
	} else {
		if row["media_type"] != b.MimeType || b.ContentBytes != b.Length {
			return exit.New(exit.Validation, "asset result differs from encoded file facts")
		}
	}
	return nil
}
func stringNumber(value any) string {
	if number, ok := value.(json.Number); ok {
		return string(number)
	}
	return ""
}
