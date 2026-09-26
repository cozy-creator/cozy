package install

import (
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func snapshotSource(installDir string, local *LocalSource) (string, *exit.Error) {
	root := filepath.Join(installDir, "source")
	frozen, problem := packagepublish.SnapshotSource(local.Tree, root)
	if problem != nil {
		return "", problem
	}
	defer frozen.Close()
	if "local/"+frozen.Name != local.Package || frozen.Release != local.Release {
		return "", exit.New(exit.Conflict, "copied project declares a different package name or version")
	}
	local.Tree = root
	return root, nil
}
