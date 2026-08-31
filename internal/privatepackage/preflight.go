package privatepackage

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
)

type BaseManifest struct {
	Digest string
	Bytes  []byte
}

// StagePreflight creates one disposable, read-only request over exact already-staged revision
// bytes and Hub-authoritative base manifests. Cleanup removes only this temporary directory.
func StagePreflight(layout home.Layout, revision Revision,
	bases []BaseManifest,
) (string, func(), *exit.Error) {
	if len(bases) == 0 || len(bases) > 32 {
		return "", func() {}, exit.Named(exit.Conflict, "private_preflight_bases_invalid",
			"Tensorhub returned %d active base manifests", len(bases))
	}
	root := filepath.Join(layout.PrivatePackages, strings.TrimPrefix(revision.Digest, "sha256:"))
	work, err := os.MkdirTemp(layout.PrivatePackages, ".preflight-")
	if err != nil {
		return "", func() {}, exit.Internalf("cannot create private preflight staging: %s", err)
	}
	cleanup := func() { _ = os.RemoveAll(work) }
	type baseRow struct {
		Path   string `json:"path"`
		Digest string `json:"digest"`
	}
	rows := make([]baseRow, 0, len(bases))
	seen := map[string]bool{}
	for index, base := range bases {
		measured, spellErr := canonical.Spell(canonical.Digest(base.Bytes))
		if spellErr != nil || measured != base.Digest || seen[base.Digest] {
			cleanup()
			return "", func() {}, exit.Named(exit.Conflict,
				"private_preflight_base_identity_mismatch",
				"active base manifest %d does not match %s", index, base.Digest)
		}
		seen[base.Digest] = true
		path := filepath.Join(work, "base-"+strings.TrimPrefix(base.Digest, "sha256:")+".json")
		if problem := copyDescriptor(base.Bytes, path); problem != nil {
			cleanup()
			return "", func() {}, problem
		}
		rows = append(rows, baseRow{Path: path, Digest: base.Digest})
	}
	wheels := make([]string, len(revision.Files))
	for index, file := range revision.Files {
		wheels[index] = file.Path
	}
	request := struct {
		Bases      []baseRow `json:"base_manifests"`
		Descriptor string    `json:"descriptor"`
		Revision   string    `json:"revision"`
		Wheels     []string  `json:"wheels"`
	}{Bases: rows, Descriptor: filepath.Join(root, privateDescriptorFile),
		Revision: filepath.Join(root, privateRevisionFile), Wheels: wheels}
	raw, err := json.Marshal(request)
	if err != nil {
		cleanup()
		return "", func() {}, exit.Internalf("cannot encode private preflight request: %s", err)
	}
	requestPath := filepath.Join(work, "request.json")
	if problem := copyDescriptor(raw, requestPath); problem != nil {
		cleanup()
		return "", func() {}, problem
	}
	if err := syncDirectory(work); err != nil {
		cleanup()
		return "", func() {}, exit.Internalf("cannot sync private preflight request: %s", err)
	}
	return requestPath, cleanup, nil
}
