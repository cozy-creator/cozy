package modelsource

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
)

// StageLocal freezes one mutable user file into the private operation tree in
// the same pass that measures its source identity. TensorFS never reopens the
// user path, and the original is never removed or changed.
func StageLocal(ctx context.Context, source Source, root string) (Plan, StagedFile, *exit.Error) {
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
	written, copyErr := io.Copy(io.MultiWriter(output, hash), contextReader{ctx: ctx, reader: input})
	after, statErr := input.Stat()
	inputClose := input.Close()
	syncErr := output.Sync()
	outputClose := output.Close()
	if ctx.Err() != nil {
		_ = os.Remove(temporary)
		return Plan{}, StagedFile{}, exit.New(exit.Canceled, "local model staging canceled")
	}
	if copyErr != nil || statErr != nil || inputClose != nil || syncErr != nil || outputClose != nil ||
		!os.SameFile(before, after) || before.Size() != after.Size() || before.ModTime() != after.ModTime() ||
		written != before.Size() {
		_ = os.Remove(temporary)
		return Plan{}, StagedFile{}, exit.Named(exit.Conflict, "model_source_changed",
			"local model source changed while it was being copied")
	}
	_ = os.Remove(target)
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
	return plan, StagedFile{Path: target}, nil
}

// StageLocalHeader creates only the sparse header view required by TensorFS
// source-plan. It never reads or copies the model body.
func StageLocalHeader(ctx context.Context, source Source, root string) (Plan, StagedFile, *exit.Error) {
	if source.Kind != LocalFile {
		return Plan{}, StagedFile{}, exit.Internalf("local header stager received a provider source")
	}
	if err := os.MkdirAll(root, 0o700); err != nil {
		return Plan{}, StagedFile{}, exit.Internalf("cannot create local header staging: %s", err)
	}
	input, err := openLocalNoFollow(source.Path)
	if err != nil {
		return Plan{}, StagedFile{}, exit.Named(exit.Validation, "model_source_file_refused",
			"local model source is not a readable regular non-symlink file: %s", err)
	}
	before, err := input.Stat()
	if err != nil || !before.Mode().IsRegular() || before.Size() <= 8 {
		input.Close()
		return Plan{}, StagedFile{}, exit.Named(exit.Validation, "model_source_file_refused",
			"local model source is not one nonempty regular file")
	}
	prefix := make([]byte, 8)
	if _, err := io.ReadFull(contextReader{ctx: ctx, reader: input}, prefix); err != nil {
		input.Close()
		return Plan{}, StagedFile{}, importReadProblem(ctx, "local model header prefix", err)
	}
	headerLength := int64(binary.LittleEndian.Uint64(prefix))
	if headerLength <= 1 || headerLength > maxCarrierHeader || 8+headerLength > before.Size() {
		input.Close()
		return Plan{}, StagedFile{}, exit.Named(exit.Validation, "model_source_header_invalid",
			"local model source declares invalid header length %d", headerLength)
	}
	header := make([]byte, headerLength)
	if _, err := io.ReadFull(contextReader{ctx: ctx, reader: input}, header); err != nil {
		input.Close()
		return Plan{}, StagedFile{}, importReadProblem(ctx, "local model header", err)
	}
	after, statErr := input.Stat()
	closeErr := input.Close()
	if statErr != nil || closeErr != nil || !os.SameFile(before, after) || before.Size() != after.Size() ||
		before.ModTime() != after.ModTime() {
		return Plan{}, StagedFile{}, exit.Named(exit.Conflict, "model_source_changed",
			"local model source changed while its header was being inspected")
	}
	target := filepath.Join(root, "source.safetensors")
	temporary := target + ".part"
	output, err := os.OpenFile(temporary, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return Plan{}, StagedFile{}, exit.Internalf("cannot create sparse local header: %s", err)
	}
	_, writeErr := output.Write(prefix)
	if writeErr == nil {
		_, writeErr = output.Write(header)
	}
	if writeErr == nil {
		writeErr = output.Truncate(before.Size())
	}
	if writeErr == nil {
		writeErr = output.Sync()
	}
	closeErr = output.Close()
	if writeErr != nil || closeErr != nil {
		_ = os.Remove(temporary)
		return Plan{}, StagedFile{}, exit.Internalf("cannot finish sparse local header: %v %v", writeErr, closeErr)
	}
	_ = os.Remove(target)
	if err := os.Rename(temporary, target); err != nil {
		_ = os.Remove(temporary)
		return Plan{}, StagedFile{}, exit.Internalf("cannot commit sparse local header: %s", err)
	}
	plan := Plan{Source: source, Canonical: source.Canonical,
		Files: []File{{Member: "source.safetensors", Length: before.Size(), Carrier: true}},
		Bytes: before.Size(), InspectedBytes: 8 + headerLength}
	return plan, StagedFile{Path: target}, nil
}

func importReadProblem(ctx context.Context, what string, err error) *exit.Error {
	if ctx.Err() != nil {
		return exit.New(exit.Canceled, "%s canceled", what)
	}
	return exit.Named(exit.Validation, "model_source_header_short", "%s is unreadable: %s", what, err)
}

type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (r contextReader) Read(buffer []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.reader.Read(buffer)
}
