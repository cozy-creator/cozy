package cli

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func inputTreeHeader(request records.Request, asset records.AssetBinding, claim *pb.Claim) (*pb.InputTreeImportHeader, []resultfiles.TreeMember, *exit.Error) {
	snapshot := asset.Snapshot
	if snapshot == nil || snapshot.ContentBytes > inputasset.MaxRootInputBytes || snapshot.Manifest.Length != int64(len(snapshot.Body)) || !bytes.Equal(canonical.Digest(snapshot.Body), mustRawInputDigest(snapshot.Manifest.Digest)) {
		return nil, nil, exit.New(exit.Conflict, "root input snapshot changed its manifest identity")
	}
	members, problem := resultfiles.ParseTreeManifest(snapshot.Body, snapshot.ContentBytes)
	if problem != nil {
		return nil, nil, problem
	}
	if asset.MediaType == resultfiles.TreeMediaType {
		if asset.Digest != snapshot.Manifest.Digest || asset.Length != snapshot.Manifest.Length {
			return nil, nil, exit.New(exit.Conflict, "Tree input confuses manifest and content identity")
		}
	} else if len(members) != 1 || members[0].Path != "payload" || members[0].Digest != asset.Digest || members[0].Length != asset.Length {
		return nil, nil, exit.New(exit.Conflict, "file input changed its captured payload")
	}
	return &pb.InputTreeImportHeader{Claim: claim, RequestId: request.ID, InputId: asset.FieldPath, Manifest: &pb.Ref{Digest: mustRawInputDigest(snapshot.Manifest.Digest), Length: uint64(snapshot.Manifest.Length)}, ManifestCanonicalBytes: snapshot.Body, ContentBytes: uint64(snapshot.ContentBytes)}, members, nil
}

func mustRawInputDigest(spelled string) []byte { raw, _ := canonical.Raw(spelled); return raw }

func verifyMachineInput(header *pb.InputTreeImportHeader, result *pb.NativeByteRetentionResult, abort bool) *exit.Error {
	if result == nil || result.Released != abort {
		return exit.New(exit.Conflict, "native input changed its intake state")
	}
	if _, err := canonical.Raw(result.RetentionId); err != nil {
		return exit.New(exit.Conflict, "native input has no exact recipient identity")
	}
	source := result.Source
	if source == nil {
		if abort {
			return nil
		}
		return exit.New(exit.Conflict, "native input has no committed receipt")
	}
	if records.ValidateByteRef(source) != nil || !bytes.Equal(source.Manifest.GetDigest(), header.Manifest.GetDigest()) ||
		source.Manifest.GetLength() != header.Manifest.GetLength() || source.ContentBytes != header.ContentBytes {
		return exit.New(exit.Conflict, "native input changed its captured manifest")
	}
	return nil
}

func (m *machineRuns) stageMachineInputs(ctx context.Context, request records.Request, connection *machineConnection) ([]*pb.InputAccess, *exit.Error) {
	if len(request.Assets) == 0 {
		return nil, nil
	}
	type staged struct {
		header  *pb.InputTreeImportHeader
		members []resultfiles.TreeMember
		input   records.MachineInput
		held    *pb.NativeByteRetentionResult
		problem *exit.Error
	}
	rows := make([]staged, len(request.Assets))
	for index, asset := range request.Assets {
		header, members, problem := inputTreeHeader(request, asset, connection.Claim)
		if problem != nil {
			return nil, problem
		}
		input, problem := m.store.BeginMachineInput(request.ID, asset)
		if problem != nil {
			return nil, problem
		}
		if input.State == "released" {
			return nil, exit.New(exit.Canceled, "native input was already released")
		}
		rows[index] = staged{header: header, members: members, input: input}
		if len(input.Receipt) > 0 {
			rows[index].held = &pb.NativeByteRetentionResult{}
			if proto.Unmarshal(input.Receipt, rows[index].held) != nil {
				return nil, exit.New(exit.Conflict, "recorded input receipt is unreadable")
			}
		}
	}
	// Every input moves at once. One the machine already took is committed without its
	// bytes; the machine reuses the objects it holds, or refuses and the bytes follow.
	var wait sync.WaitGroup
	for index := range rows {
		row, asset := &rows[index], request.Assets[index]
		if row.held != nil {
			continue
		}
		wait.Add(1)
		go func() {
			defer wait.Done()
			if m.store.MachineInputSent(connection.Name, asset.Snapshot.Manifest.Digest) {
				if held, problem := m.importMachineInput(ctx, request.ID, asset, row.header, nil, connection, false); problem == nil {
					row.held = held
					return
				}
			}
			row.held, row.problem = m.importMachineInput(ctx, request.ID, asset, row.header, row.members, connection, false)
		}()
	}
	wait.Wait()
	access := make([]*pb.InputAccess, 0, len(rows))
	for index, row := range rows {
		if row.problem != nil {
			return nil, row.problem
		}
		if len(row.input.Receipt) == 0 {
			if problem := m.store.RecordMachineInput(request.ID, row.input, row.held); problem != nil {
				return nil, problem
			}
		}
		if problem := verifyMachineInput(row.header, row.held, false); problem != nil {
			return nil, problem
		}
		access = append(access, &pb.InputAccess{InputId: request.Assets[index].FieldPath, NativeTree: &pb.NativeByteRetentionRequest{Source: row.held.Source, RetentionId: row.held.RetentionId}})
	}
	return access, nil
}

func (m *machineRuns) importMachineInput(ctx context.Context, request string, asset records.AssetBinding, header *pb.InputTreeImportHeader, members []resultfiles.TreeMember, connection *machineConnection, abort bool) (*pb.NativeByteRetentionResult, *exit.Error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	stream, err := connection.importInputTree(ctx)
	if err != nil {
		return nil, machineTransport(err)
	}
	defer stream.CloseSend()
	send := func(frame *pb.InputTreeImportFrame) *exit.Error {
		if err := stream.Send(frame); err != nil {
			if err == io.EOF {
				if _, remote := stream.CloseAndRecv(); remote != nil {
					return machineTransport(remote)
				}
			}
			return machineTransport(err)
		}
		return nil
	}
	if problem := send(&pb.InputTreeImportFrame{Body: &pb.InputTreeImportFrame_Header{Header: header}}); problem != nil {
		return nil, problem
	}
	if !abort {
		seen := map[string]bool{}
		for _, member := range members {
			if seen[member.Digest] {
				continue
			}
			seen[member.Digest] = true
			file, err := os.Open(filepath.Join(asset.Snapshot.Path+".files", strings.TrimPrefix(member.Digest, "sha256:")))
			if err != nil {
				return nil, exit.New(exit.Conflict, "captured input bytes are unavailable: %s", err)
			}
			problem := func() *exit.Error {
				defer file.Close()
				info, err := file.Stat()
				if err != nil || !info.Mode().IsRegular() || info.Size() != member.Length {
					return exit.New(exit.Conflict, "captured input object changed")
				}
				hash := sha256.New()
				buffer := make([]byte, pb.MaxInputTreeChunkBytes)
				var offset int64
				for offset < member.Length || member.Length == 0 && offset == 0 {
					current, problem := m.store.RequestRow(request)
					if problem != nil {
						return problem
					}
					if current == nil || current.State == "canceled" {
						return exit.New(exit.Canceled, "root input upload was canceled")
					}
					count, err := io.ReadFull(file, buffer[:min(int64(len(buffer)), member.Length-offset)])
					if err != nil {
						return exit.New(exit.Conflict, "captured input object changed during upload")
					}
					hash.Write(buffer[:count])
					frame := &pb.InputTreeImportFrame{Body: &pb.InputTreeImportFrame_Blob{Blob: &pb.InputTreeImportBlob{Object: &pb.Ref{Digest: mustRawInputDigest(member.Digest), Length: uint64(member.Length)}, Offset: uint64(offset), Data: buffer[:count]}}}
					if problem := send(frame); problem != nil {
						return problem
					}
					offset += int64(count)
					if member.Length == 0 {
						break
					}
				}
				if "sha256:"+hex.EncodeToString(hash.Sum(nil)) != member.Digest {
					return exit.New(exit.Conflict, "captured input failed its content digest")
				}
				return nil
			}()
			if problem != nil {
				return nil, problem
			}
		}
	}
	if !abort {
		current, problem := m.store.RequestRow(request)
		if problem != nil {
			return nil, problem
		}
		if current == nil || current.State == "canceled" {
			return nil, exit.New(exit.Canceled, "root input upload was canceled")
		}
	}
	if problem := send(&pb.InputTreeImportFrame{Body: &pb.InputTreeImportFrame_Commit{Commit: &pb.InputTreeImportCommit{Abort: abort}}}); problem != nil {
		return nil, problem
	}
	result, err := stream.CloseAndRecv()
	if err != nil {
		return nil, machineTransport(err)
	}
	if problem := verifyMachineInput(header, result, abort); problem != nil {
		return nil, problem
	}
	return result, nil
}

func (m *machineRuns) releaseMachineInputs(ctx context.Context, request records.Request, connection *machineConnection) *exit.Error {
	if !request.IsJob() {
		// Runtime adopts a job root's staged bytes; an inference root's stay held here
		// until its execution has ended.
		link, problem := m.store.MachineExecution(request.ID)
		if problem != nil {
			return problem
		}
		var state pb.MachineExecutionState
		if link != nil && len(link.Receipt) > 0 && !link.Collected &&
			(proto.Unmarshal(link.ObservedState, &state) != nil || !machineEnded(state.State)) {
			return nil
		}
	}
	inputs, problem := m.store.MachineInputs(request.ID)
	if problem != nil {
		return problem
	}
	for _, input := range inputs {
		if input.State == "released" {
			continue
		}
		var selected *records.AssetBinding
		for index := range request.Assets {
			if request.Assets[index].FieldPath == input.InputID {
				selected = &request.Assets[index]
				break
			}
		}
		if selected == nil {
			return exit.New(exit.Conflict, "native input lost its frozen capture")
		}
		header, members, problem := inputTreeHeader(request, *selected, connection.Claim)
		if problem != nil {
			return problem
		}
		result, problem := m.importMachineInput(ctx, request.ID, *selected, header, members, connection, true)
		if problem != nil {
			return problem
		}
		if problem := m.store.RecordMachineInput(request.ID, input, result); problem != nil {
			return problem
		}
	}
	return nil
}

func machineEnded(state string) bool {
	return state == "succeeded" || state == "failed" || state == "canceled"
}
