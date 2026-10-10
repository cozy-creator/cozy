package packagepublish

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
)

// PrepareUnpublishedFrom resolves a unpublished package in an owned copy. New editable
// invocable dependencies need no author-maintained lock; existing locks seed uv's
// resolution without changing either the author's source or their lock file.
// Published package intake keeps its strict source/lock requirements.
func PrepareUnpublishedFrom(ctx context.Context, projectDir string, namespace NamespaceSource, extras ...string) (*Package, *exit.Error) {
	author, problem := AuthorTree(projectDir)
	if problem != nil {
		return nil, problem
	}
	return prepareUnpublishedCopy(ctx, author, nil, namespace, extras...)
}

// AuthorTree reads an authored package directory as it stands, its uv.lock optional, with no
// copy, lock or build: a run's held snapshot of it is matched against its SourceStats.
func AuthorTree(projectDir string) (*Package, *exit.Error) {
	tree, files, problem := boundedSourceTree(projectDir, []string{"package.toml", "pyproject.toml"})
	if problem != nil {
		return nil, problem
	}
	document, problem := readProjectDocument(files["pyproject.toml"])
	if problem != nil {
		return nil, problem
	}
	metadata, problem := document.projectIdentity()
	if problem != nil {
		return nil, problem
	}
	return &Package{Files: files, Tree: tree, Name: normalizedProjectName(metadata.Name), Release: metadata.Version,
		PythonRequires: document.Project.RequiresPython}, nil
}
