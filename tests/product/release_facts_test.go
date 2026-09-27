package producttest

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

const releaseFactsInterface = `{"application":"h3:app","entrypoints":[{"name":"generate","models":[{"class":"H3","component_use":{"condition_text":["text_encoder"],"sample_fl2va":["fl2va_dit"]},"path":"generate.models.model"}],"request":{"fields":[{"name":"steps","type":"int"}]},"result":{"fields":[]}}],"format":"cozy.package.interface/1","jobs":[{"name":"lane","models":[{"class":"Source","component_use":{},"path":"lane.models.pruned"}],"publishes":false,"request":{"fields":[]},"result":{"fields":[]},"weights_outputs":[{"max_bytes":65536,"mime_type":"application/vnd.cozy.model-manifest","output_id":"attn8"}]}]}`

// releaseFactsHub answers the two routes the real PrepareFactsSource reads: the
// rental-scoped facts and the public release record carrying the interface.
func releaseFactsHub(t *testing.T, application string) (*httptest.Server, []byte) {
	t.Helper()
	normalized, err := canonical.NormalizeJCS([]byte(releaseFactsInterface))
	must(t, err)
	digest, err := canonical.Spell(canonical.Digest(normalized))
	must(t, err)
	var detail hub.PackageReleaseDetail
	detail.PackageInterface = []byte(releaseFactsInterface)
	detail.Release.Release = "1.0.0"
	detail.Release.PackageInterfaceDigest = digest
	detail.Release.PackageInterfaceLength = int64(len(normalized))
	detail.ExecutionRequirements = []string{"cozy-runtime>=0.18.21"}
	facts := testPrepareFacts("proof/h3", "1.0.0")
	inventory, err := json.Marshal(map[string]any{"format": "tensorhub.image_inventory/1",
		"profile": facts.ImageInventory.Profile, "python": facts.ImageInventory.Python,
		"distributions": []map[string]string{{"name": "numpy", "version": "2.1.0"}}})
	must(t, err)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/rentals/"+podRental+"/prepare-facts", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("package") != "proof/h3" || r.URL.Query().Get("release") != "1.0.0" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(hub.PrepareFactsView{Application: application,
			ModelSlotPaths: []string{"generate.models.model", "lane.models.pruned"},
			ImageInventory: inventory, LockedRequirements: string(facts.LockedRequirements)})
	})
	mux.HandleFunc("GET /v1/packages/proof/h3/releases/1.0.0", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(detail)
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	return server, normalized
}

// The owner's PodHost prepare carries the hub release's exact PackageInterface bytes
// (wire 61 field 10), fetched by the production facts source. Agreement between the
// interface and the assembled facts is the preparing Runtime's check, not Creator's:
// a fact the hub spells differently still reaches the pod that owns the decision.
func TestPrepareFactsCarryReleaseInterfaceToPodHost(t *testing.T) {
	for _, application := range []string{"h3:app", "other:app"} {
		t.Run(application, func(t *testing.T) {
			server, normalized := releaseFactsHub(t, application)
			client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("fixture")}, "")
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			pod := &fakePod{controlKey: public}
			connection, _ := startFakePod(t, t.TempDir(), pod)
			o := hostOwner(t, "release-facts-"+strings.ReplaceAll(application, ":", "-"), rentalWiring(connection, private),
				func(options *orchestrator.Options) { options.RentalPrepareFacts = rental.PrepareFactsSource(client) })
			instance, _, _, problem := o.c.EnsureRental(podRental)
			fatal(t, problem)
			fatal(t, o.c.ConvergePackageSet(instance, []*pb.DownloadPackageRef{{Package: "proof/h3", Release: "1.0.0"}}, nil))
			waitUntil(t, "the PodHost prepare", func() bool {
				pod.mu.Lock()
				defer pod.mu.Unlock()
				return len(pod.prepares) > 0
			})
			pod.mu.Lock()
			defer pod.mu.Unlock()
			if got := pod.prepares[0].PackageInterface; !bytes.Equal(got, normalized) {
				t.Fatalf("PreparePackageSetCall.package_interface = %q, want the release's canonical bytes %q", got, normalized)
			}
		})
	}
}

// `cozy rental update` reads the pinned PodHost's protocol range over the real TLS
// probe and refuses a Runtime whose declared wire minimum that host does not reach. A
// Runtime that declares no readable range is attempted; each operation gates itself.
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
		{61, nil, ""},
		{61, &rental.RuntimeWire{WireMinor: 60, MinimumWireMinor: 61}, ""},
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
	}
}
