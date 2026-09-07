package install

import (
	"io"
	"os"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

// snapshotSource copies only the existing bounded, admissible package source set.
// Its identity is checked after copying so an edit cannot mix two revisions.
func snapshotSource(installDir string, local LocalSource) (string, *exit.Error) {
	pack, problem := packagepublish.PrepareLocalFrom(local.Tree)
	if problem != nil {
		return "", problem
	}
	defer pack.Close()
	root := filepath.Join(installDir, "source")
	for _, name := range packagepublish.Paths(pack.Files) {
		to := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(to), 0o700); err != nil {
			return "", exit.Internalf("cannot create invocation source snapshot: %s", err)
		}
		if problem := copySnapshotFile(pack.Files[name], to); problem != nil {
			return "", problem
		}
	}
	frozen, problem := packagepublish.PrepareLocalFrom(root)
	if problem != nil {
		return "", problem
	}
	defer frozen.Close()
	digest, _, _, problem := frozen.SourceIdentity()
	if problem != nil {
		return "", problem
	}
	if digest != local.SourceDigest || "local/"+frozen.Name != local.Package || frozen.Release != local.Release {
		return "", exit.Named(exit.Conflict, "local_package_source_changed",
			"local package changed while capturing its invocation snapshot").
			WithRemedy("retry after the source has stopped changing")
	}
	return root, nil
}

func copySnapshotFile(from, to string) *exit.Error {
	in, err := os.Open(from)
	if err != nil {
		return exit.Internalf("cannot read invocation source: %s", err)
	}
	defer in.Close()
	out, err := os.OpenFile(to, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return exit.Internalf("cannot create invocation source: %s", err)
	}
	n, copyErr := io.Copy(out, io.LimitReader(in, packagepublish.MaxSourceFileBytes+1))
	syncErr := out.Sync()
	closeErr := out.Close()
	if copyErr != nil || syncErr != nil || closeErr != nil || n > packagepublish.MaxSourceFileBytes {
		return exit.Named(exit.Conflict, "local_package_source_changed",
			"cannot capture a bounded complete invocation source file")
	}
	return nil
}
