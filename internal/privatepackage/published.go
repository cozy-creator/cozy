package privatepackage

import (
	"bytes"
	"os"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
)

type PublishedWheel struct {
	Digest, Filename, Path, Kind string // kind: project | dependency
	Length                       int64
}

// StagePublishedPreflight adapts exact downloaded Hub release carriers to Runtime's existing
// portable overlay classifier. The synthetic private-revision document is disposable plumbing;
// it is never persisted as request or execution identity.
func StagePublishedPreflight(layout home.Layout, packageName, release, releaseDigest string,
	descriptor []byte, wheels []PublishedWheel,
) (Revision, home.Layout, func(), *exit.Error) {
	if len(descriptor) == 0 || len(descriptor) > canonical.DocMax {
		return Revision{}, layout, func() {}, exit.Named(exit.Structural,
			"package_preflight_descriptor_invalid", "published descriptor is absent or oversized")
	}
	normalized, err := canonical.NormalizeJCS(descriptor)
	if err != nil || !bytes.Equal(normalized, descriptor) {
		return Revision{}, layout, func() {}, exit.Named(exit.Structural,
			"package_preflight_descriptor_invalid", "published descriptor is not canonical")
	}
	root, err := os.MkdirTemp(layout.PrivatePackages, ".published-preflight-")
	if err != nil {
		return Revision{}, layout, func() {}, exit.Internalf(
			"cannot create published preflight staging: %s", err)
	}
	cleanup := func() { _ = os.RemoveAll(root) }
	stagedLayout := layout
	stagedLayout.PrivatePackages = root
	files := make([]File, 0, len(wheels))
	for _, wheel := range wheels {
		if (wheel.Kind != "project" && wheel.Kind != "dependency") || wheel.Length <= 0 ||
			filepath.Base(wheel.Filename) != wheel.Filename || filepath.Base(wheel.Path) != wheel.Filename {
			cleanup()
			return Revision{}, layout, func() {}, exit.Named(exit.Structural,
				"package_preflight_wheel_invalid", "published wheel kind is invalid")
		}
		if _, err := canonical.Raw(wheel.Digest); err != nil {
			cleanup()
			return Revision{}, layout, func() {}, exit.Named(exit.Structural,
				"package_preflight_wheel_invalid", "published wheel digest is invalid")
		}
		files = append(files, File{Digest: wheel.Digest, Filename: wheel.Filename,
			Kind: wheel.Kind, Length: wheel.Length, Path: wheel.Path})
	}
	descriptorDigest, err := canonical.Spell(canonical.Digest(descriptor))
	if err != nil {
		cleanup()
		return Revision{}, layout, func() {}, exit.Internalf("cannot spell descriptor digest: %s", err)
	}
	revision, revisionBytes, problem := identity(packageName, release, releaseDigest,
		descriptorDigest, int64(len(descriptor)), files)
	if problem != nil {
		cleanup()
		return Revision{}, layout, func() {}, problem
	}
	final := filepath.Join(root, revision.Digest[7:])
	if err := os.Mkdir(final, 0o700); err != nil {
		cleanup()
		return Revision{}, layout, func() {}, exit.Internalf("cannot stage published preflight: %s", err)
	}
	if problem := copyDescriptor(descriptor, filepath.Join(final, privateDescriptorFile)); problem != nil {
		cleanup()
		return Revision{}, layout, func() {}, problem
	}
	if problem := copyDescriptor(revisionBytes, filepath.Join(final, privateRevisionFile)); problem != nil {
		cleanup()
		return Revision{}, layout, func() {}, problem
	}
	return revision, stagedLayout, cleanup, nil
}
