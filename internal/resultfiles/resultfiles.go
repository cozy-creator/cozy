// Package resultfiles publishes daemon-owned verified media into a durable directory:
// the package's own store under outputs/ or a caller's --out. It owns filenames and the
// no-overwrite boundary; execution never composes a path from a terminal document.
package resultfiles

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Entry is one already-mirrored terminal asset and its pre-authorized destination name.
type Entry struct {
	OutputID  string
	MediaType string
	Filename  string
	Source    string
	Digest    string
	Length    int64
}

// Filename names one verified file by its own content digest and media type:
// `<sha256 hex>.<ext>`. The same bytes always get the same name, so a regenerated file
// lands on itself and two different files can never collide.
func Filename(digest, mediaType string) (string, *exit.Error) {
	hex, ok := strings.CutPrefix(digest, "sha256:")
	if !ok || len(hex) != sha256.Size*2 || strings.Trim(hex, "0123456789abcdef") != "" {
		return "", exit.Named(exit.Validation, "output_export_digest_malformed",
			"output digest %q is not sha256 over 64 lowercase hexadecimal characters", digest)
	}
	return hex + Extension(mediaType), nil
}

// Extension is the closed media-to-filename projection. Unknown bytes get no guessed
// suffix and therefore cannot enter a durable output intent.
func Extension(mediaType string) string {
	switch mediaType {
	case "image/png":
		return ".png"
	case "image/jpeg":
		return ".jpg"
	case "image/webp":
		return ".webp"
	case "video/mp4":
		return ".mp4"
	case "audio/wav":
		return ".wav"
	case "application/json":
		return ".json"
	default:
		return ""
	}
}

// Publish copies one complete verified set without overwriting an existing destination.
// An exact existing file is an idempotent replay; different bytes are a typed conflict.
func Publish(directory string, entries []Entry) ([]string, *exit.Error) {
	if !filepath.IsAbs(directory) || filepath.Clean(directory) != directory {
		return nil, exit.Named(exit.Validation, "output_export_directory_malformed",
			"output directory %q is not one canonical absolute path", directory)
	}
	if len(entries) == 0 {
		return nil, nil
	}
	if err := os.MkdirAll(directory, 0o755); err != nil {
		return nil, exportIO("cannot create output directory %s: %s", directory, err)
	}
	info, err := os.Lstat(directory)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return nil, exit.Named(exit.Conflict, "output_export_directory_substituted",
			"output directory %s is not a real directory", directory)
	}
	root, err := os.OpenRoot(directory)
	if err != nil {
		return nil, exportIO("cannot open output directory %s: %s", directory, err)
	}
	defer root.Close()

	paths := make([]string, 0, len(entries))
	for _, entry := range entries {
		if filepath.Base(entry.Filename) != entry.Filename || strings.ContainsAny(entry.Filename, `/\\`) {
			return nil, exit.Named(exit.Validation, "output_export_path_escape",
				"output filename %q is not one path element", entry.Filename)
		}
		replayed, problem := existing(root, entry)
		if problem != nil {
			return nil, problem
		}
		if !replayed {
			if problem := publishOne(root, entry); problem != nil {
				return nil, problem
			}
		}
		paths = append(paths, filepath.Join(directory, entry.Filename))
	}
	if handle, err := os.Open(directory); err == nil {
		err = handle.Sync()
		_ = handle.Close()
		if err != nil {
			return nil, exportIO("cannot make output directory durable: %s", err)
		}
	}
	return paths, nil
}

func existing(root *os.Root, entry Entry) (bool, *exit.Error) {
	info, err := root.Lstat(entry.Filename)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, exportIO("cannot inspect output %s: %s", entry.Filename, err)
	}
	if !info.Mode().IsRegular() {
		return false, exit.Named(exit.Conflict, "output_export_destination_substituted",
			"output destination %s is not a regular file", entry.Filename)
	}
	if problem := verifyRoot(root, entry.Filename, entry.Length, entry.Digest); problem != nil {
		return false, exit.Named(exit.Conflict, "output_export_destination_conflict",
			"output destination %s already contains different bytes", entry.Filename).
			WithRemedy("move or remove the conflicting file, then repeat the same run idempotency key")
	}
	return true, nil
}

func publishOne(root *os.Root, entry Entry) *exit.Error {
	sourceInfo, err := os.Lstat(entry.Source)
	if err != nil || !sourceInfo.Mode().IsRegular() {
		return exportIO("verified source for %s is unavailable", entry.OutputID)
	}
	source, err := os.Open(entry.Source)
	if err != nil {
		return exportIO("cannot open verified source for %s: %s", entry.OutputID, err)
	}
	defer source.Close()

	temporary := ".cozy-" + randomSuffix()
	staged, err := root.OpenFile(temporary, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return exportIO("cannot stage output %s: %s", entry.OutputID, err)
	}
	defer root.Remove(temporary)
	hash := sha256.New()
	written, copyErr := io.Copy(io.MultiWriter(staged, hash), io.LimitReader(source, entry.Length+1))
	if copyErr == nil {
		copyErr = staged.Sync()
	}
	if closeErr := staged.Close(); copyErr == nil {
		copyErr = closeErr
	}
	if copyErr != nil {
		return exportIO("cannot stage output %s: %s", entry.OutputID, copyErr)
	}
	digest := "sha256:" + hex.EncodeToString(hash.Sum(nil))
	if written != entry.Length || digest != entry.Digest {
		return exit.Named(exit.Validation, "output_export_source_changed",
			"verified source for %s changed while exporting", entry.OutputID)
	}
	if err := root.Link(temporary, entry.Filename); err != nil {
		if replayed, problem := existing(root, entry); replayed && problem == nil {
			return nil
		} else if problem != nil {
			return problem
		}
		return exportIO("cannot publish output %s: %s", entry.Filename, err)
	}
	return nil
}

func verifyRoot(root *os.Root, name string, length int64, digest string) *exit.Error {
	file, err := root.Open(name)
	if err != nil {
		return exportIO("cannot open output %s: %s", name, err)
	}
	defer file.Close()
	hash := sha256.New()
	n, err := io.Copy(hash, io.LimitReader(file, length+1))
	if err != nil {
		return exportIO("cannot verify output %s: %s", name, err)
	}
	if n != length || "sha256:"+hex.EncodeToString(hash.Sum(nil)) != digest {
		return exit.Named(exit.Conflict, "output_export_destination_conflict",
			"output %s differs from the accepted media manifest", name)
	}
	return nil
}

func randomSuffix() string {
	var raw [12]byte
	if _, err := rand.Read(raw[:]); err != nil {
		panic(fmt.Sprintf("crypto/rand: %s", err))
	}
	return hex.EncodeToString(raw[:])
}

func exportIO(format string, args ...any) *exit.Error {
	return exit.Named(exit.Unavailable, "output_export_io", format, args...).
		WithRemedy("free space or restore access to the recorded output directory; Cozy retries the durable export")
}

// Preflight proves the destination can accept a published file BEFORE a request is
// queued: create-if-absent, then a touch-and-unlink probe in the final directory. It
// runs in the daemon — the process that later publishes — so it proves the writer's own
// access, not a client's. A probe that passes is not a reservation; the durable export
// still handles a destination that dies later.
func Preflight(directory string) *exit.Error {
	if !filepath.IsAbs(directory) || filepath.Clean(directory) != directory {
		return exit.Named(exit.Validation, "output_export_directory_malformed",
			"output directory %q is not one canonical absolute path", directory)
	}
	if err := os.MkdirAll(directory, 0o755); err != nil {
		return unwritable(directory, err)
	}
	info, err := os.Lstat(directory)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return exit.Named(exit.Conflict, "output_export_directory_substituted",
			"output directory %s is not a real directory", directory)
	}
	root, err := os.OpenRoot(directory)
	if err != nil {
		return unwritable(directory, err)
	}
	defer root.Close()
	name := ".cozy-probe-" + randomSuffix()
	probe, err := root.OpenFile(name, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return unwritable(directory, err)
	}
	problem := probe.Close()
	if err := root.Remove(name); problem == nil {
		problem = err
	}
	if problem != nil {
		return unwritable(directory, problem)
	}
	return nil
}

func unwritable(directory string, err error) *exit.Error {
	return exit.Named(exit.Validation, "output_destination_unwritable",
		"output destination %s is not writable: %s", directory, ioCause(err)).
		WithRemedy("fix permissions on %s (or pick another --out) and resubmit; nothing was queued", directory)
}

// ioCause strips the probe's own random filename out of the refusal: the caller's
// remediable fact is the directory and the errno, not a name that never existed.
func ioCause(err error) error {
	var pathError *fs.PathError
	if errors.As(err, &pathError) {
		return pathError.Err
	}
	return err
}
