package rental

// The prepare facts (wire 31, xs-019). PreparePackageSetCall fields 3-6 are the
// record owner's word: `application`, the interface's model-slot paths, the
// placed image's measured inventory, and the hash-pinned locked requirements.
// The hub assembles them ONCE — packagerelease.PrepareFacts joined with the
// rental's registered image row — and this side fetches that assembly verbatim
// over the renter's authenticated edge rather than re-deriving any of it.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

const inventoryFormat = "tensorhub.image_inventory/1"

// PublicOrigin reads the Hub-owned package index endpoint already embedded in
// authenticated prepare facts. It never changes the control URL or sends a
// credential to that endpoint. Ambiguous or non-HTTPS facts supply no override.
func PublicOrigin(requirements []byte, pkg string) string {
	ref, problem := hub.ParseRef(pkg)
	if problem != nil {
		return ""
	}
	wanted := "/v1/index/" + ref.Org + "/simple/"
	origin := ""
	for _, line := range strings.Split(string(requirements), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 || fields[0] != "--extra-index-url" {
			continue
		}
		value, err := url.Parse(fields[1])
		if err != nil || value.Scheme != "https" || value.Host == "" || value.User != nil || value.RawQuery != "" || value.Fragment != "" || value.RawPath != "" || value.Path != wanted {
			continue
		}
		selected := value.Scheme + "://" + value.Host
		if origin != "" && origin != selected {
			return ""
		}
		origin = selected
	}
	return origin
}

// PrepareFactsSource fetches one package release's facts for one rental and
// answers them in the call's own shape.
func PrepareFactsSource(client *hub.Client) orchestrator.RentalPrepareFactsSource {
	return func(ctx context.Context, connection *orchestrator.WorkerConnection,
		ref *pb.DownloadPackageRef) (orchestrator.PrepareFacts, *exit.Error) {
		if connection == nil || connection.RentalID == "" || ref == nil ||
			ref.Package == "" || ref.Release == "" {
			return orchestrator.PrepareFacts{}, exit.Internalf(
				"prepare facts need a rental connection and one exact package release")
		}
		view, problem := client.PrepareFacts(ctx, connection.RentalID, ref.Package, ref.Release)
		if problem != nil {
			return orchestrator.PrepareFacts{}, problem
		}
		return PrepareFactsFromView(view, ref.Package, ref.Release)
	}
}

// PrepareFactsFromView admits the hub's answer onto the wire shape. Bounds are
// the wire's; the pod host and the Runtime re-verify everything downstream.
func PrepareFactsFromView(view hub.PrepareFactsView, pkg, release string,
) (orchestrator.PrepareFacts, *exit.Error) {
	refuse := func(format string, args ...any) (orchestrator.PrepareFacts, *exit.Error) {
		return orchestrator.PrepareFacts{}, exit.Named(exit.Structural, "rental.prepare_facts_invalid",
			"the hub's prepare facts for %s %s are unusable: %s",
			pkg, release, fmt.Sprintf(format, args...))
	}
	if view.Application == "" {
		return refuse("no application")
	}
	if len(view.ModelSlotPaths) > pb.MaxModelSlotPaths {
		return refuse("%d model-slot paths exceed the wire bound %d",
			len(view.ModelSlotPaths), pb.MaxModelSlotPaths)
	}
	if view.LockedRequirements == "" || len(view.LockedRequirements) > pb.MaxLockedRequirementsBytes {
		return refuse("locked requirements are absent or exceed the %d-byte wire bound",
			pb.MaxLockedRequirementsBytes)
	}
	inventory, err := ImageInventory(view.ImageInventory)
	if err != nil {
		return refuse("%s", err)
	}
	return orchestrator.PrepareFacts{
		Application:        view.Application,
		ModelSlotPaths:     append([]string(nil), view.ModelSlotPaths...),
		ImageInventory:     inventory,
		LockedRequirements: []byte(view.LockedRequirements),
	}, nil
}

// imageInventory converts the registered tensorhub.image_inventory/1 document
// into the wire message the Runtime range-checks its base against. The Runtime
// refuses a preparation without one, so an image registered without an
// inventory is refused HERE, with the fact named, not on the pod.
func ImageInventory(raw json.RawMessage) (*pb.ImageInventory, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, fmt.Errorf("the rental's image has no registered inventory")
	}
	var doc struct {
		Format        string `json:"format"`
		Profile       string `json:"profile"`
		Python        string `json:"python"`
		Distributions []struct {
			Name    string `json:"name"`
			Version string `json:"version"`
		} `json:"distributions"`
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&doc); err != nil {
		return nil, fmt.Errorf("image inventory unreadable: %w", err)
	}
	if next := decoder.Decode(new(any)); next != io.EOF {
		return nil, fmt.Errorf("image inventory has trailing JSON")
	}
	if doc.Format != inventoryFormat {
		return nil, fmt.Errorf("image inventory format %q is not %q", doc.Format, inventoryFormat)
	}
	if doc.Profile == "" || doc.Python == "" {
		return nil, fmt.Errorf("image inventory names no profile or interpreter")
	}
	if len(doc.Distributions) > pb.MaxImageInventoryDistributions {
		return nil, fmt.Errorf("image inventory's %d distributions exceed the wire bound %d",
			len(doc.Distributions), pb.MaxImageInventoryDistributions)
	}
	inventory := &pb.ImageInventory{Profile: doc.Profile, Python: doc.Python}
	for _, row := range doc.Distributions {
		if row.Name == "" || row.Version == "" {
			return nil, fmt.Errorf("image inventory carries an unnamed or unversioned distribution")
		}
		inventory.Distributions = append(inventory.Distributions,
			&pb.ImageDistribution{Distribution: row.Name, Version: row.Version})
	}
	return inventory, nil
}
