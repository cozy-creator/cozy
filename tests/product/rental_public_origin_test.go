package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/rental"
)

func TestRentalPublicOriginUsesOnlyItsHubPackageIndex(t *testing.T) {
	for _, test := range []struct{ value, want string }{
		{"--index-url https://pypi.org/simple\n--extra-index-url https://public.example/v1/index/paul/simple/\nmodel @ https://bytes.example/file.whl\n", "https://public.example"},
		{"--extra-index-url http://public.example/v1/index/paul/simple/\n", ""},
		{"--extra-index-url https://secret@public.example/v1/index/paul/simple/\n", ""},
		{"--extra-index-url https://public.example/v1/index/other/simple/\n", ""},
		{"--extra-index-url https://public.example/v1/index/paul/simple/?token=x\n", ""},
		{"--extra-index-url https://public.example/v1/index/paul/simple/#fragment\n", ""},
		{"--extra-index-url https://public.example/elsewhere\n", ""},
		{"--extra-index-url https://one.example/v1/index/paul/simple/\n--extra-index-url https://two.example/v1/index/paul/simple/\n", ""},
	} {
		if got := rental.PublicOrigin([]byte(test.value), "paul/minimax-h3"); got != test.want {
			t.Fatalf("public index origin %q want%q", got, test.want)
		}
	}
}
