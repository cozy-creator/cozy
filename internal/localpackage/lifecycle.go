package localpackage

import (
	"errors"
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
		return out, exit.New(exit.Validation, "local package revision digest %q is invalid", digest)
	}
	info, err := os.Lstat(root)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return out, exit.Named(exit.Structural, "local_package_revision_invalid",
			"local package revision root is not one real directory")
	}
	raw, err := os.ReadFile(filepath.Join(root, localRevisionFile))
	if err != nil || len(raw) == 0 || len(raw) > canonical.DocMax {
		return out, exit.Named(exit.Structural, "local_package_revision_invalid",
			"local package revision document is absent or over its bound")
	}
	spelled, err := canonical.Spell(canonical.Digest(raw))
	if err != nil || spelled != digest {
		return out, exit.Named(exit.Conflict, "local_package_revision_changed",
			"local package revision document no longer matches %s", digest)
	}
	doc, err := canonical.Read(raw, &pb.LocalPackageRevision{})
	if err != nil {
		var refusal *canonical.Error
		if errors.As(err, &refusal) && refusal.Code == "unknown_format" {
			// The bytes hash to the directory's name, so this is one of ours — written under
			// a document format this build no longer reads. Nothing can use it again.
			return out, exit.Named(exit.Structural, "local_package_revision_retired", "%s", err)
		}
		return out, exit.Named(exit.Structural, "local_package_revision_invalid", "%s", err)
	}
	out.digest, out.packageName, out.release = digest, doc.Str("package"), doc.Str("release")
	out.sourceDigest = doc.Str("source_digest")
	if _, err := canonical.Raw(out.sourceDigest); err != nil {
		return durableIdentity{}, exit.Named(exit.Structural,
			"local_package_revision_invalid", "source digest: %s", err)
	}
	if out.packageName == "" || out.release == "" {
		return durableIdentity{}, exit.Named(exit.Structural,
			"local_package_revision_invalid", "local package revision identity is incomplete")
	}
	return out, nil
}

// Drop removes one exact, self-identifying revision directory. RemoveAll never follows child
// symlinks, and the root itself is Lstat-fenced before deletion.
func Drop(layout home.Layout, digest string) *exit.Error {
	root := filepath.Join(layout.LocalPackages, strings.TrimPrefix(digest, "sha256:"))
	if _, err := os.Lstat(root); os.IsNotExist(err) {
		return nil
	} else if err != nil {
		return exit.Internalf("cannot inspect local package revision %s: %s", digest, err)
	}
	if _, problem := readDurableIdentity(root, digest); problem != nil {
		return problem
	}
	if err := os.RemoveAll(root); err != nil {
		return exit.Internalf("cannot remove local package revision %s: %s", digest, err)
	}
	if err := syncDirectory(layout.LocalPackages); err != nil {
		return exit.Internalf("cannot sync local package cleanup: %s", err)
	}
	return nil
}

// DropDigestUnowned is the terminal/restart form: its canonical revision document supplies the
// source declaration even if the mutable checkout or install row has already disappeared.
func DropDigestUnowned(layout home.Layout, store *records.Store, digest string) *exit.Error {
	root := filepath.Join(layout.LocalPackages, strings.TrimPrefix(digest, "sha256:"))
	if _, err := os.Lstat(root); os.IsNotExist(err) {
		return nil
	} else if err != nil {
		return exit.Internalf("cannot inspect local package revision %s: %s", digest, err)
	}
	identity, problem := readDurableIdentity(root, digest)
	if problem != nil {
		return problem
	}
	used, problem := store.LocalPackageInUse(identity.digest, identity.packageName,
		identity.release, identity.sourceDigest)
	if problem != nil || used {
		return problem
	}
	return Drop(layout, digest)
}

// Sweep removes dead staging directories and committed revisions left without a live request or
// current editable declaration. Unrecognized paths are never guessed to be ours.
func Sweep(layout home.Layout, store *records.Store) *exit.Error {
	entries, err := os.ReadDir(layout.LocalPackages)
	if err != nil {
		return exit.Internalf("cannot scan local package revisions: %s", err)
	}
	for _, entry := range entries {
		name := entry.Name()
		path := filepath.Join(layout.LocalPackages, name)
		if strings.HasPrefix(name, ".stage-") {
			if err := os.RemoveAll(path); err != nil {
				return exit.Internalf("cannot remove orphan local package staging %s: %s", name, err)
			}
			continue
		}
		digest := "sha256:" + name
		if len(name) != 64 || !validDigest(digest) {
			continue
		}
		identity, problem := readDurableIdentity(path, digest)
		if problem != nil && problem.Name == "local_package_revision_retired" {
			if err := os.RemoveAll(path); err != nil {
				return exit.Internalf("cannot retire local package revision %s: %s", name, err)
			}
			continue
		}
		if problem != nil {
			return problem
		}
		used, problem := store.LocalPackageInUse(identity.digest, identity.packageName,
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
