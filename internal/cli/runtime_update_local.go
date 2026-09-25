package cli

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"
	"regexp"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Freeze before creating the maintenance row: a retry never reopens the caller's
// mutable build output, and a daemon crash cannot change a local ask into PyPI.
func freezeRuntimeWheel(root, source string) (_ *runtimeUpdateWheel, problem *exit.Error) {
	const limit = 128 << 20
	name := filepath.Base(source)
	if !filepath.IsAbs(source) || !regexp.MustCompile(`^[A-Za-z0-9_.+-]+\.whl$`).MatchString(name) {
		return nil, exit.New(exit.Validation, "runtime_wheel must be an absolute local wheel path")
	}
	info, err := os.Stat(source)
	if err != nil || !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > limit {
		return nil, exit.New(exit.Validation, "local Runtime wheel must be a regular file of at most 128 MiB")
	}
	input, err := os.Open(source)
	if err != nil {
		return nil, exit.New(exit.Validation, "cannot read local Runtime wheel: %s", err)
	}
	defer input.Close()
	info, err = input.Stat()
	if err != nil || !info.Mode().IsRegular() {
		return nil, exit.New(exit.Validation, "local Runtime wheel is not a regular file")
	}
	root = filepath.Join(root, "runtime-update-wheels")
	if err = os.MkdirAll(root, 0700); err != nil {
		return nil, exit.Internalf("cannot create local wheel staging: %s", err)
	}
	directory, err := os.MkdirTemp(root, "candidate-")
	if err != nil {
		return nil, exit.Internalf("cannot create local wheel snapshot: %s", err)
	}
	defer func() {
		if problem != nil {
			_ = os.RemoveAll(directory)
		}
	}()
	path := filepath.Join(directory, name)
	output, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return nil, exit.Internalf("cannot retain local Runtime wheel: %s", err)
	}
	defer output.Close()
	digest := sha256.New()
	length, err := io.Copy(io.MultiWriter(output, digest), io.LimitReader(input, limit+1))
	if err != nil || length != info.Size() || length <= 0 || length > limit {
		return nil, exit.New(exit.Validation, "local Runtime wheel changed size or could not be captured")
	}
	if err = output.Sync(); err != nil {
		return nil, exit.Internalf("cannot persist local Runtime wheel: %s", err)
	}
	for _, directory := range []string{directory, root} {
		handle, err := os.Open(directory)
		if err != nil {
			return nil, exit.Internalf("cannot open local wheel directory: %s", err)
		}
		err = handle.Sync()
		_ = handle.Close()
		if err != nil {
			return nil, exit.Internalf("cannot persist local wheel directory: %s", err)
		}
	}
	return &runtimeUpdateWheel{Filename: name, Path: path, Digest: "sha256:" + hex.EncodeToString(digest.Sum(nil)), Length: length}, nil
}
