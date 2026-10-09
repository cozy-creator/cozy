package cli

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"github.com/cozy-creator/cozy/internal/archive"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// keptRelease is one published release as this computer read it: immutable, so a run's
// results are read against its interface with no Hub or machine call. Requirements, its
// dependency closure, decide the machine class a rental needs; a machine's own description
// carries none, and never erases the closure an install or the Hub gave.
type keptRelease struct {
	Interface    json.RawMessage `json:"package_interface"`
	Requirements []string        `json:"requirements,omitempty"`
}

func releaseInterfacePath(root, hubURL, pkg, release string) string {
	name := sha256.Sum256([]byte(hubURL + "\x00" + pkg + "@" + release))
	return filepath.Join(root, "releases", "interfaces", hex.EncodeToString(name[:16])+".json")
}

func keepReleaseInterface(root, hubURL, pkg, release string, raw []byte, requirements []string) {
	if requirements == nil {
		requirements = readKeptRelease(root, hubURL, pkg, release).Requirements
	}
	doc, err := json.Marshal(keptRelease{Interface: raw, Requirements: requirements})
	path := releaseInterfacePath(root, hubURL, pkg, release)
	if err == nil && os.MkdirAll(filepath.Dir(path), 0o700) == nil && os.WriteFile(path+".tmp", doc, 0o600) == nil {
		_ = os.Rename(path+".tmp", path)
	}
}

// newestReleasePath names the release of pkg at hub this client last read, and the catalog
// revision it read it under, so a later run under that revision names it with no Hub or
// machine call.
func newestReleasePath(root, hubURL, pkg string) string {
	name := sha256.Sum256([]byte(hubURL + "\x00" + pkg))
	return filepath.Join(root, "releases", "newest", hex.EncodeToString(name[:16]))
}

func keepNewestRelease(root, hubURL, pkg, revision, release string) {
	path := newestReleasePath(root, hubURL, pkg)
	if os.MkdirAll(filepath.Dir(path), 0o700) == nil && os.WriteFile(path+".tmp", []byte(revision+"\n"+release), 0o600) == nil {
		_ = os.Rename(path+".tmp", path)
	}
}

// keptNewestRelease is the release keepNewestRelease named under revision and its kept
// interface, or "".
func keptNewestRelease(root, hubURL, pkg, revision string) (string, *launch.PackageInterface) {
	raw, err := os.ReadFile(newestReleasePath(root, hubURL, pkg))
	under, release, _ := strings.Cut(string(raw), "\n")
	if err != nil || under != revision || release == "" {
		return "", nil
	}
	kept := readKeptRelease(root, hubURL, pkg, release)
	if kept.Interface == nil {
		return "", nil
	}
	surface, problem := launch.DecodePackageInterface(kept.Interface)
	if problem != nil {
		return "", nil
	}
	return release, surface
}

// readKeptRelease is the kept release, or none; a file holding only an interface has no
// closure.
func readKeptRelease(root, hubURL, pkg, release string) keptRelease {
	raw, err := os.ReadFile(releaseInterfacePath(root, hubURL, pkg, release))
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
	var submission archive.Submission
	if link != nil {
		submission, _ = archive.ReadSubmission(link.Submission)
	}
	if submission.Root != nil {
		// The machine installed the committed release: its interface as a machine described
		// it here, else as the Hub read that chose a machine to rent kept it.
		root := submission.Root
		if root.InstallationID != "" {
			// A root naming unpublished code: the interface is that code's own.
			_, surface, problem := r.installPackageInterface(request.InstallID)
			return surface, problem
		}
		if kept := readKeptRelease(home.Paths(r.cfg.Home).Root, request.Hub, root.Package, root.Release); kept.Interface != nil {
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
	if link == nil || len(submission.Capture) == 0 {
		return nil, exit.New(exit.Conflict, "accepted execution metadata is unavailable")
	}
	capture, err := canonical.ReadObject(submission.Capture)
	if err != nil || capture.Str("format") != "cozy.worker.v1.MachineExecutionCapture/2" {
		return nil, exit.New(exit.Conflict, "accepted execution metadata is unavailable")
	}
	for _, installed := range capture.List("installed_packages") {
		if installed.Str("installation_id") == capture.Str("root_installation_id") {
			raw, err := base64.StdEncoding.Strict().DecodeString(installed.Str("package_interface"))
			if err != nil {
				return nil, exit.New(exit.Conflict, "accepted package interface is unreadable")
			}
			return launch.DecodePackageInterface(raw)
		}
	}
	return nil, exit.New(exit.NotFound, "package interface is absent from accepted execution")
}
