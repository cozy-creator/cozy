package canonical

import (
	"encoding/json"
	"math"
	"strconv"
	"strings"
)

// NormalizeApplication preserves authored application values, including exact integer
// digits and integer/float kind. Object order and insignificant whitespace do not
// affect intent; array order does. This is not the JCS protocol profile.
func NormalizeApplication(data []byte) ([]byte, error) {
	return normalizeJSON(data, applicationNumber)
}

func applicationNumber(raw string) (string, error) {
	if !strings.ContainsAny(raw, ".eE") {
		// The JSON token reader already enforced integer grammar. Do not convert
		// through float64 or impose a protocol bound on a typed Python integer.
		if raw == "-0" {
			return "0", nil
		}
		return raw, nil
	}
	value, err := strconv.ParseFloat(raw, 64)
	if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
		return "", refuse("number_range", "%s has no finite IEEE-754 spelling", raw)
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", refuse("number_range", "%s: %v", raw, err)
	}
	spelling := string(encoded)
	if !strings.ContainsAny(spelling, ".eE") {
		spelling += ".0"
	}
	return spelling, nil
}
