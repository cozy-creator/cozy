package modelsource

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
)

// StageLocal freezes one mutable user file into the private operation tree in
// the same pass that measures its source identity. TensorFS never reopens the
// user path, and the original is never removed or changed.
func StageLocal(source Source, root string) (Plan, StagedFile, *exit.Error) {
	if source.Kind != LocalFile {
		return Plan{}, StagedFile{}, exit.Internalf("local stager received a provider source")
	}
	if err := os.MkdirAll(root, 0o700); err != nil {
		return Plan{}, StagedFile{}, exit.Internalf("cannot create local model staging: %s", err)
	}
	input, err := openLocalNoFollow(source.Path)
	if err != nil {
		return Plan{}, StagedFile{}, exit.Named(exit.Validation, "model_source_file_refused",
			"local model source is not a readable regular non-symlink file: %s", err)
	}
	before, err := input.Stat()
	if err != nil || !before.Mode().IsRegular() || before.Size() <= 0 {
		input.Close()
		return Plan{}, StagedFile{}, exit.Named(exit.Validation, "model_source_file_refused",
			"local model source is not one nonempty regular file")
	}
	target := filepath.Join(root, "source.safetensors")
	temporary := target + ".part"
	output, err := os.OpenFile(temporary, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		input.Close()
		return Plan{}, StagedFile{}, exit.Internalf("cannot create local model staging file: %s", err)
	}
	hash := sha256.New()
	written, copyErr := io.Copy(io.MultiWriter(output, hash), input)
	after, statErr := input.Stat()
	inputClose := input.Close()
	syncErr := output.Sync()
	outputClose := output.Close()
	if copyErr != nil || statErr != nil || inputClose != nil || syncErr != nil || outputClose != nil ||
		!os.SameFile(before, after) || before.Size() != after.Size() || before.ModTime() != after.ModTime() ||
		written != before.Size() {
		_ = os.Remove(temporary)
		return Plan{}, StagedFile{}, exit.Named(exit.Conflict, "model_source_changed",
			"local model source changed while it was being copied")
	}
	if err := os.Rename(temporary, target); err != nil {
		_ = os.Remove(temporary)
		return Plan{}, StagedFile{}, exit.Internalf("cannot commit local model staging: %s", err)
	}
	sha := hex.EncodeToString(hash.Sum(nil))
	resolved := source
	resolved.Canonical = "sha256:" + sha
	plan := Plan{Source: resolved, Canonical: resolved.Canonical, SelectionSHA256: sha,
		Files: []File{{Member: "source.safetensors", SHA256: sha, Length: written, Carrier: true}},
		Bytes: written}
	return plan, StagedFile{Path: target, Carrier: true}, nil
}
