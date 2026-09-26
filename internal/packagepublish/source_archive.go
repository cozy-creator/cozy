package packagepublish

import (
	"archive/tar"
	"io"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
)

// WriteSourceArchive transports one invocation's source tree. The filename and
// operation identify its destination; its bytes are not a package identity.
func WriteSourceArchive(tree, destination string) (int64, *exit.Error) {
	output, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return 0, exit.Internalf("cannot create source archive: %s", err)
	}
	archive := tar.NewWriter(output)
	var total int64
	count := 0
	err = filepath.WalkDir(tree, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return fs.ErrInvalid
		}
		name, err := filepath.Rel(tree, path)
		if err != nil {
			return err
		}
		count++
		total += info.Size()
		if count > MaxSourceFiles || total > 1<<30 || info.Size() > SourceFileLimit(name) {
			return fs.ErrInvalid
		}
		header := &tar.Header{Name: filepath.ToSlash(name), Mode: int64(info.Mode().Perm()), Size: info.Size(), Typeflag: tar.TypeReg}
		if err := archive.WriteHeader(header); err != nil {
			return err
		}
		input, err := os.Open(path)
		if err != nil {
			return err
		}
		_, copyErr := io.CopyN(archive, input, info.Size())
		closeErr := input.Close()
		if copyErr != nil {
			return copyErr
		}
		return closeErr
	})
	archiveErr := archive.Close()
	syncErr := output.Sync()
	info, statErr := output.Stat()
	closeErr := output.Close()
	if err != nil || archiveErr != nil || syncErr != nil || statErr != nil || closeErr != nil || info.Size() > 1<<30 {
		_ = os.Remove(destination)
		return 0, exit.New(exit.Validation, "cannot create a bounded regular-file source archive")
	}
	return info.Size(), nil
}
