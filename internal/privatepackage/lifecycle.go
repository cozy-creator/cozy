package privatepackage

import (
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

var ownership sync.Mutex

// Guard closes the filesystem/SQLite ownership gap between staging one exact revision and
// recording the request that owns it. Terminal cleanup and restart sweep take the same guard.
func Guard() func() {
	ownership.Lock()
	return ownership.Unlock
}

type durableIdentity struct {
	digest, packageName, release, sourceDigest string
}

func readDurableIdentity(root, digest string) (durableIdentity, *exit.Error) {
	var out durableIdentity
	if !validDigest(digest) {
		return out, exit.New(exit.Validation, "private package revision digest %q is invalid", digest)
	}
	info, err := os.Lstat(root)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return out, exit.Named(exit.Structural, "private_package_revision_invalid",
			"private package revision root is not one real directory")
	}
	raw, err := os.ReadFile(filepath.Join(root, privateRevisionFile))
	if err != nil || len(raw) == 0 || len(raw) > canonical.DocMax {
		return out, exit.Named(exit.Structural, "private_package_revision_invalid",
			"private package revision document is absent or over its bound")
	}
	spelled, err := canonical.Spell(canonical.Digest(raw))
	if err != nil || spelled != digest {
		return out, exit.Named(exit.Conflict, "private_package_revision_changed",
			"private package revision document no longer matches %s", digest)
	}
	doc, err := canonical.Read(raw, &pb.PrivatePackageRevision{})
	if err != nil {
		return out, exit.Named(exit.Structural, "private_package_revision_invalid", "%s", err)
	}
	out.digest, out.packageName, out.release = digest, doc.Str("package"), doc.Str("release")
	out.sourceDigest = doc.Str("source_digest")
	if _, err := canonical.Raw(out.sourceDigest); err != nil {
		return durableIdentity{}, exit.Named(exit.Structural,
			"private_package_revision_invalid", "source digest: %s", err)
	}
	if out.packageName == "" || out.release == "" {
		return durableIdentity{}, exit.Named(exit.Structural,
			"private_package_revision_invalid", "private package revision identity is incomplete")
	}
	return out, nil
}

// Drop removes one exact, self-identifying revision directory. RemoveAll never follows child
// symlinks, and the root itself is Lstat-fenced before deletion.
func Drop(layout home.Layout, digest string) *exit.Error {
	root := filepath.Join(layout.PrivatePackages, strings.TrimPrefix(digest, "sha256:"))
	if _, err := os.Lstat(root); os.IsNotExist(err) {
		return nil
	} else if err != nil {
		return exit.Internalf("cannot inspect private package revision %s: %s", digest, err)
	}
	if _, problem := readDurableIdentity(root, digest); problem != nil {
		return problem
	}
	if err := os.RemoveAll(root); err != nil {
		return exit.Internalf("cannot remove private package revision %s: %s", digest, err)
	}
	if err := syncDirectory(layout.PrivatePackages); err != nil {
		return exit.Internalf("cannot sync private package cleanup: %s", err)
	}
	return nil
}

// DropUnowned removes a staged revision only after neither a live request nor the current
// editable pin names its exact source declaration.
func DropUnowned(layout home.Layout, store *records.Store, revision Revision) *exit.Error {
	return DropDigestUnowned(layout, store, revision.Digest)
}

// DropDigestUnowned is the terminal/restart form: its canonical revision document supplies the
// source declaration even if the mutable checkout or install row has already disappeared.
func DropDigestUnowned(layout home.Layout, store *records.Store, digest string) *exit.Error {
	root := filepath.Join(layout.PrivatePackages, strings.TrimPrefix(digest, "sha256:"))
	if _, err := os.Lstat(root); os.IsNotExist(err) {
		return nil
	} else if err != nil {
		return exit.Internalf("cannot inspect private package revision %s: %s", digest, err)
	}
	identity, problem := readDurableIdentity(root, digest)
	if problem != nil {
		return problem
	}
	used, problem := store.PrivatePackageInUse(identity.digest, identity.packageName,
		identity.release, identity.sourceDigest)
	if problem != nil || used {
		return problem
	}
	return Drop(layout, digest)
}

// Sweep removes dead staging directories and committed revisions left without a live request or
// current editable declaration. Unrecognized paths are never guessed to be ours.
func Sweep(layout home.Layout, store *records.Store) *exit.Error {
	entries, err := os.ReadDir(layout.PrivatePackages)
	if err != nil {
		return exit.Internalf("cannot scan private package revisions: %s", err)
	}
	for _, entry := range entries {
		name := entry.Name()
		path := filepath.Join(layout.PrivatePackages, name)
		if strings.HasPrefix(name, ".stage-") {
			if err := os.RemoveAll(path); err != nil {
				return exit.Internalf("cannot remove orphan private package staging %s: %s", name, err)
			}
			continue
		}
		digest := "sha256:" + name
		if len(name) != 64 || !validDigest(digest) {
			continue
		}
		identity, problem := readDurableIdentity(path, digest)
		if problem != nil {
			return problem
		}
		used, problem := store.PrivatePackageInUse(identity.digest, identity.packageName,
			identity.release, identity.sourceDigest)
		if problem != nil {
			return problem
		}
		if !used {
			if problem := Drop(layout, digest); problem != nil {
				return problem
			}
		}
	}
	return nil
}

func validDigest(digest string) bool {
	_, err := canonical.Raw(digest)
	return err == nil
}
