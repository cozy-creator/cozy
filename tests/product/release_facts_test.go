package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/rental"
)

// `cozy rental update` reads the pinned PodHost's protocol range over the real TLS
// probe and refuses a Runtime whose declared wire minimum that host does not reach, or
// that declares no readable range.
func TestRuntimeUpdateRefusesRuntimeAheadOfHost(t *testing.T) {
	for _, test := range []struct {
		host   uint32
		target *rental.RuntimeWire
		code   string
	}{
		{60, &rental.RuntimeWire{WireMinor: 61, MinimumWireMinor: 61}, "rental.runtime_update_host_too_old"},
		{60, &rental.RuntimeWire{WireMinor: 61, MinimumWireMinor: 60}, ""},
		{61, &rental.RuntimeWire{WireMinor: 61, MinimumWireMinor: 61}, ""},
		{61, &rental.RuntimeWire{WireMinor: 62, MinimumWireMinor: 62}, "rental.runtime_update_host_too_old"},
		{61, nil, "rental.runtime_update_range_unreadable"},
		{61, &rental.RuntimeWire{WireMinor: 60, MinimumWireMinor: 61}, "rental.runtime_update_range_unreadable"},
	} {
		public, _, err := ed25519.GenerateKey(rand.Reader)
		must(t, err)
		connection, _ := startFakePod(t, t.TempDir(), &fakePod{controlKey: public, wireMinor: test.host})
		info, problem := orchestrator.RentalProtocolInfo(context.Background(), connection)
		fatal(t, problem)
		problem = rental.RuntimeUpdateHost("0.18.99", test.target, info.WireMinor)
		got := ""
		if problem != nil {
			got = problem.ErrName()
		}
		if got != test.code {
			t.Fatalf("host %d target %+v: %v, want %q", test.host, test.target, problem, test.code)
		}
		if got == "rental.runtime_update_host_too_old" && !strings.Contains(problem.Message, "do not replace the pod Host") {
			t.Fatal("refusal did not explain the Host replacement boundary")
		}
	}
}
