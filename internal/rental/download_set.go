package rental

import (
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// DownloadSet authors the DESIRED DOWNLOAD SET for one logical package/model selection:
// which packages and which models a rental's pod must hold, sorted and unique.
//
// It authorizes NOTHING. The signed DownloadDelegation this used to be -- an Ed25519
// document binding the rental, worker, boot and pinned TLS leaf, with a one-hour expiry --
// is deleted (owner ruling 2026-09-03). The hub's closure and presign routes are public
// and packages and repos are publicly readable, so the pod fetches with no credential and
// this document is desired state alone. It resolves no platform, wheel, CUDA build, or
// storage URL; pod-supervisor and Tensorhub own that work.
//
// Because the document is now a pure function of its content, the same selection authors
// the same bytes every time. The pod's preparation ledger keys on those bytes, so an
// unchanged selection is recognised as unchanged rather than re-prepared.
func DownloadSet(packages []*pb.DownloadPackageRef, models []*pb.DownloadModelRef,
) ([]byte, *exit.Error) {
	if len(packages) > 32 || len(models) > 32 || len(packages)+len(models) == 0 {
		return nil, exit.Named(exit.Validation, "rental.download_set_too_large",
			"a download set names at most 32 packages and 32 models")
	}
	packages = append([]*pb.DownloadPackageRef(nil), packages...)
	models = append([]*pb.DownloadModelRef(nil), models...)
	sort.Slice(packages, func(i, j int) bool {
		return packages[i].Package+"\x00"+packages[i].Release < packages[j].Package+"\x00"+packages[j].Release
	})
	sort.Slice(models, func(i, j int) bool {
		return modelKey(models[i]) < modelKey(models[j])
	})
	prior := ""
	selectedPackages := map[string]bool{}
	for _, row := range packages {
		key := ""
		if row != nil {
			key = row.Package + "\x00" + row.Release
		}
		if row == nil || strings.TrimSpace(row.Package) != row.Package || row.Package == "" ||
			strings.TrimSpace(row.Release) != row.Release || row.Release == "" || key <= prior {
			return nil, exit.Named(exit.Validation, "rental.download_set_package_invalid",
				"download packages must be complete, unique logical refs")
		}
		selectedPackages[row.Package] = true
		prior = key
	}
	prior = ""
	modelOnlyPackage := ""
	for _, row := range models {
		key := modelKey(row)
		_, digestErr := canonical.Raw(row.GetManifest())
		standalone := len(packages) == 0 && row.GetPackage() == "" && row.GetSlot() == ""
		packageSelected := standalone || selectedPackages[row.GetPackage()]
		if len(packages) == 0 && row != nil {
			if _, problem := hub.ParseRef(row.Package); problem == nil {
				if modelOnlyPackage == "" {
					modelOnlyPackage = row.Package
				}
				packageSelected = row.Package == modelOnlyPackage
			}
		}
		if row == nil || strings.TrimSpace(row.Model) != row.Model || row.Model == "" ||
			strings.TrimSpace(row.Release) != row.Release ||
			strings.TrimSpace(row.Package) != row.Package || !packageSelected ||
			strings.TrimSpace(row.Slot) != row.Slot || row.Slot == "" && !standalone ||
			strings.TrimSpace(row.Lane) != row.Lane || (row.Release == "") != (row.Lane == "") ||
			digestErr != nil || key <= prior {
			return nil, exit.Named(exit.Validation, "rental.download_set_model_invalid",
				"download models need unique exact manifests and a release/lane pair or neither")
		}
		prior = key
	}
	document, err := canonical.Bytes(&pb.DownloadDelegation{Models: models, Packages: packages})
	if err != nil {
		return nil, exit.Internalf("cannot author the download set: %s", err)
	}
	return document, nil
}

func modelKey(row *pb.DownloadModelRef) string {
	if row == nil {
		return ""
	}
	return row.Package + "\x00" + row.Slot + "\x00" + row.Model + "\x00" +
		row.Release + "\x00" + row.Manifest
}

func PackageSetSource() orchestrator.RentalPackageSetSource {
	return func(packages []*pb.DownloadPackageRef, models []*pb.DownloadModelRef) ([]byte, *exit.Error) {
		return DownloadSet(packages, models)
	}
}
