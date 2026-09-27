package cli

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type machineByteStream interface {
	Recv() (*pb.NativeByteReadChunk, error)
}

// machineFilePlan is what one outcome's file receipts leave to collect. The machine is an
// independently upgraded peer: an output it returns without a declaration is ignored, and a
// declared output it does not return, or returns unusably, fails alone. Each is one warning.
type machineFilePlan struct {
	files    []records.MachineFileResult
	warnings []string
}

func (m *machineRuns) planMachineFiles(request records.Request, outcome *pb.AttemptOutcome, body *pb.AttemptOutcomeBody) (*machineFilePlan, *exit.Error) {
	entries := body.GetOutputManifest().GetOutputs()
	if len(entries) == 0 && (request.Outputs == "" || body.Status != pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED) {
		return nil, nil
	}
	if len(entries) > pb.MaxChildArtifactGrants || body.Result == nil || body.Result.ResultBlob != nil {
		return nil, exit.New(exit.Conflict, "file result exceeds the captured control bounds")
	}
	surface, problem := m.resolver.capturedResultInterface(request)
	if problem != nil {
		return nil, problem
	}
	entrypoint, problem := surface.Function(request.Entrypoint)
	if problem != nil {
		return nil, problem
	}
	if len(entries) == 0 && len(launch.AssetPaths(entrypoint.Result)) == 0 {
		return nil, nil
	}
	schema, problem := machineResultSchema(surface, request.Entrypoint)
	if problem != nil {
		return nil, problem
	}
	if problem := launch.ValidateMachineResult(schema, body.Result); problem != nil {
		return nil, problem
	}
	var result any
	decoder := json.NewDecoder(bytes.NewReader(body.Result.InlineResult))
	decoder.UseNumber()
	if decoder.Decode(&result) != nil {
		return nil, exit.New(exit.Conflict, "file result is unreadable")
	}
	declared := map[string]bool{}
	for _, name := range launch.AssetPaths(entrypoint.Result) {
		declared[name] = true
	}
	plan := &machineFilePlan{files: make([]records.MachineFileResult, 0, len(entries))}
	for _, entry := range entries {
		if entry == nil || !declared[entry.OutputId] {
			plan.warnings = append(plan.warnings, fmt.Sprintf("the machine returned output %q, which the package does not declare; ignored", entry.GetOutputId()))
			continue
		}
		delete(declared, entry.OutputId)
		file, problem := m.machineFile(request, outcome, result, entry)
		if problem != nil {
			plan.warnings = append(plan.warnings, fmt.Sprintf("output %q failed: %s", entry.OutputId, problem.Message))
			continue
		}
		plan.files = append(plan.files, file)
	}
	if body.Status == pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED {
		for _, name := range slices.Sorted(maps.Keys(declared)) {
			plan.warnings = append(plan.warnings, fmt.Sprintf("output %q failed: the machine did not return it", name))
		}
	}
	return plan, nil
}

func (m *machineRuns) machineFile(request records.Request, outcome *pb.AttemptOutcome, result any, entry *pb.OutputEntry) (records.MachineFileResult, *exit.Error) {
	if len(entry.Digest) != 32 || entry.Length > math.MaxInt64 || records.ValidateByteRef(entry.NativeTree) != nil {
		return records.MachineFileResult{}, exit.New(exit.Conflict, "its native receipt is incomplete")
	}
	maximum, problem := m.resolver.CapturedByteOutputBound(request, entry.OutputId, entry.MimeType)
	if problem != nil {
		return records.MachineFileResult{}, problem
	}
	if entry.NativeTree.ContentBytes > uint64(maximum) || entry.MimeType != resultfiles.TreeMediaType && entry.NativeTree.ContentBytes != entry.Length {
		return records.MachineFileResult{}, exit.New(exit.Conflict, "it exceeds its declared size or changes its native content length")
	}
	digest, _ := canonical.Spell(entry.Digest)
	receipt, _ := canonical.Spell(entry.NativeTree.ReceiptDigest)
	manifest, _ := canonical.Spell(entry.NativeTree.Manifest.Digest)
	source := records.ByteOutput{RequestID: request.ID, Attempt: int64(outcome.AttemptOrdinal), OutputID: entry.OutputId, Digest: digest, Length: int64(entry.Length), MimeType: entry.MimeType, ProducerRootID: entry.NativeTree.ProducerRootId, ReceiptDigest: receipt, ManifestID: manifest, ManifestLength: int64(entry.NativeTree.Manifest.Length), ContentBytes: int64(entry.NativeTree.ContentBytes)}
	if problem := orchestrator.VerifyByteResultRow(result, source); problem != nil {
		return records.MachineFileResult{}, problem
	}
	name, problem := resultfiles.Filename(digest, entry.MimeType)
	if problem != nil {
		return records.MachineFileResult{}, problem
	}
	return records.MachineFileResult{OutcomeID: outcome.OutcomeId, RetentionID: records.ByteRetentionID(request.ID, "machine-result", entry.OutputId, source), Source: source, Output: records.Output{OutputID: entry.OutputId, MediaID: records.NewID("med"), Path: filepath.Join(m.layout.PublicationRoot(request.Org, request.ID), "received", name), Digest: digest, Length: int64(entry.Length), MimeType: entry.MimeType}}, nil
}

func (m *machineRuns) collectMachineFiles(ctx context.Context, request records.Request, connection *machineConnection, plan *machineFilePlan) (bool, *exit.Error) {
	if plan == nil {
		return false, nil
	}
	for _, file := range plan.files {
		file, problem := m.store.FreezeMachineFileResult(request.ID, file)
		if problem != nil {
			return false, problem
		}
		if file.State == "pending" {
			retained, err := connection.retainBytes(ctx, fileRetentionRequest(file))
			if err != nil {
				return false, machineTransport(err)
			}
			if problem := verifyFileRetention(file, retained, false); problem != nil {
				return false, problem
			}
			if problem := receiveMachineFile(ctx, connection, file); problem != nil {
				return false, problem
			}
			if problem := m.store.AdvanceMachineFileResult(request.ID, file.RetentionID, "copied"); problem != nil {
				return false, problem
			}
		} else if problem := verifyMachineResultCopy(file); problem != nil {
			// The machine still holds bytes it has not released: receive them again.
			if file.State != "copied" {
				return false, problem
			}
			if problem := receiveMachineFile(ctx, connection, file); problem != nil {
				return false, problem
			}
		}
		if file.State != "released" {
			released, err := connection.releaseBytes(ctx, fileRetentionRequest(file))
			if err != nil {
				return false, machineTransport(err)
			}
			if problem := verifyFileRetention(file, released, true); problem != nil {
				return false, problem
			}
			if problem := m.store.AdvanceMachineFileResult(request.ID, file.RetentionID, "released"); problem != nil {
				return false, problem
			}
		}
	}
	export, problem := m.store.OutputExportOf(request.ID)
	if problem != nil {
		return false, problem
	}
	if export != nil && export.State != "published" {
		var paths []string
		for _, declared := range export.Outputs {
			for _, file := range plan.files {
				if file.Output.OutputID == declared.OutputID {
					path, problem := materializeMachineResult(file, export.Directory)
					if problem != nil {
						_ = m.store.FailOutputExport(request.ID, problem.ErrName(), problem.Message)
						return false, problem
					}
					paths = append(paths, path)
				}
			}
		}
		if problem := m.store.CompleteOutputExport(request.ID, paths); problem != nil {
			return false, problem
		}
	}
	return true, nil
}

func verifyMachineFileCopy(output records.Output) *exit.Error {
	info, err := os.Lstat(output.Path)
	if err != nil || !info.Mode().IsRegular() || info.Size() != output.Length {
		return exit.New(exit.Conflict, "received file custody is absent or changed")
	}
	file, err := os.Open(output.Path)
	if err != nil {
		return exit.Internalf("cannot open received file: %s", err)
	}
	defer file.Close()
	hash := sha256.New()
	written, err := io.Copy(hash, io.LimitReader(file, output.Length+1))
	if err != nil || written != output.Length || "sha256:"+hex.EncodeToString(hash.Sum(nil)) != output.Digest {
		return exit.New(exit.Conflict, "received file custody no longer matches its digest")
	}
	return nil
}

func fileRetentionRequest(file records.MachineFileResult) *pb.NativeByteRetentionRequest {
	return &pb.NativeByteRetentionRequest{RetentionId: file.RetentionID, Source: file.Source.NativeRef()}
}
func verifyFileRetention(file records.MachineFileResult, result *pb.NativeByteRetentionResult, released bool) *exit.Error {
	want, got := file.Source.NativeRef(), result.GetSource()
	if result == nil || result.RetentionId != file.RetentionID || result.Released != released ||
		got.GetProducerRootId() != want.ProducerRootId || !bytes.Equal(got.GetReceiptDigest(), want.ReceiptDigest) ||
		!bytes.Equal(got.GetManifest().GetDigest(), want.Manifest.Digest) || got.GetManifest().GetLength() != want.Manifest.Length ||
		got.GetContentBytes() != want.ContentBytes {
		return exit.New(exit.Conflict, "machine returned a different native file custody receipt")
	}
	return nil
}

func receiveMachineBlob(ctx context.Context, connection *machineConnection, retention *pb.NativeByteRetentionRequest, descriptor records.Output) *exit.Error {
	directory := filepath.Dir(descriptor.Path)
	if err := os.MkdirAll(directory, 0700); err != nil {
		return exit.Internalf("cannot create received file custody: %s", err)
	}
	root, err := os.OpenRoot(directory)
	if err != nil {
		return exit.Internalf("cannot open received file custody: %s", err)
	}
	defer root.Close()
	name := ".receive-" + records.NewID("file")
	output, err := root.OpenFile(name, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return exit.Internalf("cannot stage received file: %s", err)
	}
	defer func() { output.Close(); _ = root.Remove(name) }()
	digest, _ := canonical.Raw(descriptor.Digest)
	stream, err := connection.readBytes(ctx, retention, &pb.Ref{Digest: digest, Length: uint64(descriptor.Length)})
	if err != nil {
		return machineTransport(err)
	}
	hash := sha256.New()
	var offset uint64
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return machineTransport(err)
		}
		if chunk == nil || chunk.Offset != offset || len(chunk.Data) == 0 || len(chunk.Data) > pb.MaxNativeByteReadChunkBytes || uint64(len(chunk.Data)) > uint64(descriptor.Length)-offset {
			return exit.New(exit.Conflict, "received file stream changed its exact object bounds")
		}
		if _, err := io.MultiWriter(output, hash).Write(chunk.Data); err != nil {
			return exit.Internalf("cannot write received file: %s", err)
		}
		connection.progress.Advance(int64(len(chunk.Data)))
		offset += uint64(len(chunk.Data))
	}
	if offset != uint64(descriptor.Length) || "sha256:"+hex.EncodeToString(hash.Sum(nil)) != descriptor.Digest {
		return exit.New(exit.Conflict, "received file differs from its immutable digest")
	}
	if err := output.Sync(); err != nil {
		return exit.Internalf("cannot sync received file: %s", err)
	}
	if err := output.Close(); err != nil {
		return exit.Internalf("cannot close received file: %s", err)
	}
	if err := root.Rename(name, filepath.Base(descriptor.Path)); err != nil {
		return exit.Internalf("cannot commit received file: %s", err)
	}
	parent, err := root.Open(".")
	if err != nil {
		return exit.Internalf("cannot open received file directory: %s", err)
	}
	defer parent.Close()
	if err := parent.Sync(); err != nil {
		return exit.Internalf("cannot sync received file directory: %s", err)
	}
	return nil
}

func (m *machineRuns) releaseMachineFiles(ctx context.Context, request string, connection *machineConnection) *exit.Error {
	files, problem := m.store.MachineFileResults(request)
	if problem != nil {
		return problem
	}
	for _, file := range files {
		if file.State == "released" {
			continue
		}
		if problem := m.store.AdvanceMachineFileResult(request, file.RetentionID, "releasing"); problem != nil {
			return problem
		}
		result, err := connection.releaseBytes(ctx, fileRetentionRequest(file))
		if err != nil {
			return machineTransport(err)
		}
		if problem := verifyFileRetention(file, result, true); problem != nil {
			return problem
		}
		if problem := m.store.AdvanceMachineFileResult(request, file.RetentionID, "released"); problem != nil {
			return problem
		}
	}
	return nil
}

func machineResultSchema(surface *launch.PackageInterface, entrypoint string) (json.RawMessage, *exit.Error) {
	var document struct {
		Jobs []struct {
			Name   string          `json:"name"`
			Result json.RawMessage `json:"result"`
		} `json:"jobs"`
	}
	if json.Unmarshal(surface.Raw, &document) != nil {
		return nil, exit.New(exit.Conflict, "captured result schema is unreadable")
	}
	for _, job := range document.Jobs {
		if job.Name == entrypoint {
			return job.Result, nil
		}
	}
	return nil, exit.New(exit.Conflict, "captured result schema is absent for %s", strings.TrimSpace(entrypoint))
}
