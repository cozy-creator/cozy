package rental

import (
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// SignDownloadDelegation binds one logical package/model set to the exact
// rental, worker, boot, and pinned TLS leaf. It does not resolve a platform,
// wheel, CUDA build, or storage URL; pod-supervisor and Tensorhub own that work.
func SignDownloadDelegation(l home.Layout, connection *orchestrator.WorkerConnection,
	packages []*pb.DownloadPackageRef, models []*pb.DownloadModelRef, expiresAt time.Time,
) ([]byte, []byte, *exit.Error) {
	if connection == nil || connection.RentalID == "" || connection.WorkerID == "" ||
		connection.WorkerBootID == "" {
		return nil, nil, exit.Named(exit.Credential, "rental.delegation_identity_missing",
			"download delegation requires exact rental, worker, boot, and Creator identity")
	}
	now := time.Now()
	if !expiresAt.After(now) || expiresAt.After(now.Add(time.Hour)) {
		return nil, nil, exit.Named(exit.Validation, "rental.delegation_expiry_invalid",
			"download delegation expiry must be in the next hour")
	}
	if len(packages) > 32 || len(models) > 32 {
		return nil, nil, exit.Named(exit.Validation, "rental.delegation_too_large",
			"download delegation admits at most 32 packages and 32 models")
	}
	packages = append([]*pb.DownloadPackageRef(nil), packages...)
	models = append([]*pb.DownloadModelRef(nil), models...)
	sort.Slice(packages, func(i, j int) bool {
		return packages[i].Package+"\x00"+packages[i].Release < packages[j].Package+"\x00"+packages[j].Release
	})
	sort.Slice(models, func(i, j int) bool {
		left := modelKey(models[i])
		right := modelKey(models[j])
		return left < right
	})
	prior := ""
	delegatedPackages := map[string]bool{}
	for _, row := range packages {
		key := ""
		if row != nil {
			key = row.Package + "\x00" + row.Release
		}
		if row == nil || strings.TrimSpace(row.Package) != row.Package || row.Package == "" ||
			strings.TrimSpace(row.Release) != row.Release || row.Release == "" || key <= prior {
			return nil, nil, exit.Named(exit.Validation, "rental.delegation_package_invalid",
				"download packages must be complete, unique logical refs")
		}
		delegatedPackages[row.Package] = true
		prior = key
	}
	prior = ""
	for _, row := range models {
		key := modelKey(row)
		_, digestErr := canonical.Raw(row.GetManifest())
		if row == nil || strings.TrimSpace(row.Model) != row.Model || row.Model == "" ||
			strings.TrimSpace(row.Release) != row.Release || row.Release == "" ||
			strings.TrimSpace(row.Package) != row.Package || !delegatedPackages[row.Package] ||
			strings.TrimSpace(row.Slot) != row.Slot || row.Slot == "" ||
			digestErr != nil || key <= prior {
			return nil, nil, exit.Named(exit.Validation, "rental.delegation_model_invalid",
				"download models must be complete, unique logical refs with exact manifests")
		}
		prior = key
	}
	pin, err := workertls.LoadPin(connection.CACert)
	if err != nil {
		return nil, nil, exit.New(exit.Credential, "the worker certificate pin is unreadable: %s", err)
	}
	document, err := canonical.Bytes(&pb.DownloadDelegation{
		ExpiresAtUnix: uint64(expiresAt.Unix()), Models: models, Packages: packages,
		RentalId: connection.RentalID, WorkerBootId: connection.WorkerBootID,
		WorkerId: connection.WorkerID, WorkerTlsCertificateDigest: pin.Digest(),
	})
	if err != nil {
		return nil, nil, exit.Internalf("cannot author the download delegation: %s", err)
	}
	identity, problem := CreatorIdentityFor(l, connection.RentalID)
	if problem != nil {
		return nil, nil, problem
	}
	return document, identity.Sign(document), nil
}

func modelKey(row *pb.DownloadModelRef) string {
	if row == nil {
		return ""
	}
	return row.Package + "\x00" + row.Slot + "\x00" + row.Model + "\x00" +
		row.Release + "\x00" + row.Manifest
}

func PackageSetSigner(l home.Layout) orchestrator.RentalPackageSetSource {
	return func(connection *orchestrator.WorkerConnection, packages []*pb.DownloadPackageRef,
		models []*pb.DownloadModelRef) ([]byte, []byte, *exit.Error) {
		return SignDownloadDelegation(l, connection, packages, models, time.Now().Add(30*time.Minute))
	}
}
