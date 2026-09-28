package resultfiles

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"math"
	"os"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Materialize exports already accepted bytes from internal publication custody.
// Keep that custody independent of the user-editable copy, including across mounts.
// The destination name comes only from the accepted digest and media type.
func Materialize(source, directory, digest, mediaType string, length int64) (string, *exit.Error) {
	name, problem := Filename(digest, mediaType)
	if problem != nil {
		return "", problem
	}
	return MaterializeParts([]string{source}, directory, name, digest, length)
}

// MaterializeParts writes `name` in `directory` as the sources concatenated, verified
// against `digest` and `length`, and replaces any file of that name atomically: a reader
// opening it sees the whole previous bytes or the whole new ones.
func MaterializeParts(sources []string, directory, name, digest string, length int64) (string, *exit.Error) {
	if length < 0 || length == math.MaxInt64 {
		return "", exit.Named(exit.Validation, "output_export_length_invalid", "output length is invalid")
	}
	if problem := Preflight(directory); problem != nil {
		return "", problem
	}
	root, err := os.OpenRoot(directory)
	if err != nil {
		return "", exportWriteFailure(directory, err)
	}
	defer root.Close()
	temporary := ".cozy-export-" + randomSuffix()
	output, err := root.OpenFile(temporary, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return "", exportWriteFailure(directory, err)
	}
	defer func() { output.Close(); _ = root.Remove(temporary) }()
	hash := sha256.New()
	var written int64
	for _, source := range sources {
		input, err := os.Open(source)
		if err != nil {
			return "", exit.Internalf("cannot open accepted output: %s", err)
		}
		copied, err := io.Copy(io.MultiWriter(output, hash), io.LimitReader(input, length-written+1))
		input.Close()
		if err != nil {
			return "", exit.Internalf("cannot export accepted output: %s", err)
		}
		written += copied
	}
	if written != length || "sha256:"+hex.EncodeToString(hash.Sum(nil)) != digest {
		return "", exit.Named(exit.Conflict, "output_export_source_changed", "accepted output changed before export")
	}
	if err := output.Sync(); err != nil {
		return "", exportWriteFailure(directory, err)
	}
	if err := output.Close(); err != nil {
		return "", exportWriteFailure(directory, err)
	}
	if err := root.Rename(temporary, name); err != nil {
		return "", exportWriteFailure(directory, err)
	}
	parent, err := root.Open(".")
	if err != nil {
		return "", exportWriteFailure(directory, err)
	}
	defer parent.Close()
	if err := parent.Sync(); err != nil {
		return "", exportWriteFailure(directory, err)
	}
	return filepath.Join(directory, name), nil
}

func exportWriteFailure(directory string, err error) *exit.Error {
	return exit.Named(exit.Failed, "output_export_write_failed",
		"cannot export accepted output under %s: %s", directory, ioCause(err))
}
