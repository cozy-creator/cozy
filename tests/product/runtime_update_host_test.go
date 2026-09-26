package producttest

import (
	"github.com/cozy-creator/cozy/internal/rental"
	"strings"
	"testing"
)

func TestRuntimeUpdateHostKeepsHostBoundary(t *testing.T) {
	for _, row := range []struct {
		host   uint32
		target *rental.RuntimeWire
		code   string
	}{
		{60, &rental.RuntimeWire{WireMinor: 62, MinimumWireMinor: 62}, "rental.runtime_update_host_too_old"},
		{60, &rental.RuntimeWire{WireMinor: 61, MinimumWireMinor: 60}, ""},
		{62, &rental.RuntimeWire{WireMinor: 62, MinimumWireMinor: 62}, ""},
		{60, nil, "rental.runtime_update_wire_unknown"},
	} {
		problem := rental.RuntimeUpdateHost("candidate", row.target, row.host)
		if row.code == "" {
			if problem != nil {
				t.Fatal(problem)
			}
			continue
		}
		if problem == nil || problem.ErrName() != row.code {
			t.Fatalf("host %d target %+v: %v", row.host, row.target, problem)
		}
		if row.code == "rental.runtime_update_host_too_old" && !strings.Contains(problem.Message, "do not replace the pod Host") {
			t.Fatal("refusal did not explain Host replacement boundary")
		}
	}
}
