package cli

import (
	"context"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
)

func machineTreeMembers(file records.MachineFileResult) ([]resultfiles.TreeMember, *exit.Error) {
	return resultfiles.ReadTreeManifest(file.Output.Path, file.Output.Digest, file.Output.Length, file.Source.ContentBytes)
}

func machineTreeMemberOutput(file records.MachineFileResult, member resultfiles.TreeMember) records.Output {
	return records.Output{Path: filepath.Join(file.Output.Path+".files", strings.TrimPrefix(member.Digest, "sha256:")), Digest: member.Digest, Length: member.Length}
}

func receiveMachineFile(ctx context.Context, connection *machineConnection, file records.MachineFileResult) *exit.Error {
	if problem := receiveMachineBlob(ctx, connection, fileRetentionRequest(file), file.Output); problem != nil {
		return problem
	}
	if file.Output.MimeType != resultfiles.TreeMediaType {
		return nil
	}
	members, problem := machineTreeMembers(file)
	if problem != nil {
		return problem
	}
	for _, member := range members {
		if problem := receiveMachineBlob(ctx, connection, fileRetentionRequest(file), machineTreeMemberOutput(file, member)); problem != nil {
			return problem
		}
	}
	directory, err := os.Open(filepath.Dir(file.Output.Path))
	if err != nil {
		return exit.Internalf("cannot open received tree directory: %s", err)
	}
	defer directory.Close()
	if err := directory.Sync(); err != nil {
		return exit.Internalf("cannot sync received tree directory: %s", err)
	}
	return nil
}

func verifyMachineResultCopy(file records.MachineFileResult) *exit.Error {
	if problem := verifyMachineFileCopy(file.Output); problem != nil {
		return problem
	}
	if file.Output.MimeType != resultfiles.TreeMediaType {
		return nil
	}
	members, problem := machineTreeMembers(file)
	if problem != nil {
		return problem
	}
	for _, member := range members {
		if problem := verifyMachineFileCopy(machineTreeMemberOutput(file, member)); problem != nil {
			return problem
		}
	}
	return nil
}

func materializeMachineResult(file records.MachineFileResult, directory string) (string, *exit.Error) {
	if file.Output.MimeType == resultfiles.TreeMediaType {
		return resultfiles.MaterializeTree(file.Output.Path, directory, file.Output.Digest, file.Output.Length, file.Source.ContentBytes)
	}
	return resultfiles.Materialize(file.Output.Path, directory, file.Output.Digest, file.Output.MimeType, file.Output.Length)
}
