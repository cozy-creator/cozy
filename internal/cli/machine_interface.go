package cli

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
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

// keptRelease is one published release as this computer read it: immutable, so a run's
// results are read against its interface with no Hub or machine call. Requirements, its
// dependency closure, decide the machine class a rental needs; a machine's own description
// carries none, and never erases the closure an install or the Hub gave.
type keptRelease struct {
	Interface    json.RawMessage `json:"package_interface"`
	Requirements []string        `json:"requirements,omitempty"`
}

func releaseInterfacePath(root, pkg, release string) string {
	name := sha256.Sum256([]byte(pkg + "@" + release))
	return filepath.Join(root, "releases", "interfaces", hex.EncodeToString(name[:16])+".json")
}

func keepReleaseInterface(root, pkg, release string, raw []byte, requirements []string) {
	if requirements == nil {
		requirements = readKeptRelease(root, pkg, release).Requirements
	}
	doc, err := json.Marshal(keptRelease{Interface: raw, Requirements: requirements})
	path := releaseInterfacePath(root, pkg, release)
	if err == nil && os.MkdirAll(filepath.Dir(path), 0o700) == nil && os.WriteFile(path+".tmp", doc, 0o600) == nil {
		_ = os.Rename(path+".tmp", path)
	}
}

// readKeptRelease is the kept release, or none; a file holding only an interface has no
// closure.
func readKeptRelease(root, pkg, release string) keptRelease {
	raw, err := os.ReadFile(releaseInterfacePath(root, pkg, release))
	if err != nil {
		return keptRelease{}
	}
	var kept keptRelease
	if json.Unmarshal(raw, &kept) != nil || len(kept.Interface) == 0 {
		return keptRelease{Interface: raw}
	}
	return kept
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
		if root.InstallationId != "" {
			// A root naming unpublished code: the interface is that code's own.
			_, surface, problem := r.installPackageInterface(request.InstallID)
			return surface, problem
		}
		if kept := readKeptRelease(home.Paths(r.cfg.Home).Root, root.Package, root.Release); kept.Interface != nil {
			return launch.DecodePackageInterface(kept.Interface)
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
