package cli

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// releaseInterfacePath keeps one published release's interface as a machine described it:
// immutable, so a run's results are read against it with no Hub or machine call.
func releaseInterfacePath(root, pkg, release string) string {
	name := sha256.Sum256([]byte(pkg + "@" + release))
	return filepath.Join(root, "releases", "interfaces", hex.EncodeToString(name[:16])+".json")
}

func keepReleaseInterface(root, pkg, release string, raw []byte) {
	path := releaseInterfacePath(root, pkg, release)
	if os.MkdirAll(filepath.Dir(path), 0o700) == nil && os.WriteFile(path+".tmp", raw, 0o600) == nil {
		_ = os.Rename(path+".tmp", path)
	}
}

func (r *Resolver) capturedResultInterface(request records.Request) (*launch.PackageInterface, *exit.Error) {
	link, problem := r.store.MachineExecution(request.ID)
	if problem != nil {
		return nil, problem
	}
	if request.InstallID != "" && (link == nil || len(link.Submission) == 0 || !strings.HasPrefix(request.Package, "local/")) {
		_, surface, problem := r.installPackageInterface(request.InstallID)
		return surface, problem
	}
	var submission pb.MachineExecutionSubmit
	var capture pb.MachineExecutionCapture
	if link != nil && proto.Unmarshal(link.Submission, &submission) == nil && submission.ReleaseRoot != nil {
		// The machine installed the committed release: its interface as a machine described
		// it here, else as the Hub read that chose a machine to rent kept it.
		root := submission.ReleaseRoot
		if raw, err := os.ReadFile(releaseInterfacePath(home.Paths(r.cfg.Home).Root, root.Package, root.Release)); err == nil {
			return launch.DecodePackageInterface(raw)
		}
		ref, problem := hub.ParseRef(root.Package)
		if problem != nil {
			return nil, problem
		}
		ctx, cancel := hub.Context()
		defer cancel()
		detail, problem := r.catalog(request.Hub).PackageRelease(ctx, ref, root.Release)
		if problem != nil {
			return nil, problem
		}
		return launch.DecodePackageInterface(detail.PackageInterface)
	}
	if link == nil || proto.Unmarshal(link.Submission, &submission) != nil || canonical.Unmarshal(submission.CaptureCanonicalBytes, &capture) != nil {
		return nil, exit.New(exit.Conflict, "accepted execution metadata is unavailable")
	}
	for _, installed := range capture.InstalledPackages {
		if installed.InstallationId == capture.RootInstallationId {
			return launch.DecodePackageInterface(installed.PackageInterface)
		}
	}
	return nil, exit.New(exit.NotFound, "worker interface is absent from accepted execution")
}
