package localpackage

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"os"

	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

const uploadChunkBytes = 1 << 20
const uploadWindow = 4

// UploadFile pipelines at most four chunks beyond the Host's durable prefix.
// Only a matching VERIFIED acknowledgement for the entire file completes it.
// The callback sees validated acknowledgements, never speculative sent progress.
func UploadFile(ctx context.Context, host pb.PodHostClient, header *pb.LocalPackageUploadHeader,
	path string, onStatus func(*pb.LocalPackageFileStatus),
) *exit.Error {
	if header == nil || header.Claim == nil || header.File == nil ||
		header.OperationId == "" ||
		(len(header.File.Digest) != 32 && !(len(header.File.Digest) == 0 && header.File.Filename == "source.tar")) || header.File.Filename == "" ||
		header.File.Length == 0 || header.File.Length > math.MaxInt64 {
		return exit.Named(exit.Structural, "local_package_upload_header_invalid", "captured wheel upload has no complete identity")
	}
	// SendMsg may retain its message after Send returns. The caller's header and
	// every chunk remain immutable, including while acknowledgements arrive.
	header = proto.Clone(header).(*pb.LocalPackageUploadHeader)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	stream, err := host.LocalPackageUpload(ctx)
	if err != nil {
		return uploadProblem(ctx, err)
	}
	defer stream.CloseSend()
	if err := stream.Send(&pb.LocalPackageUploadFrame{Body: &pb.LocalPackageUploadFrame_Header{Header: header}}); err != nil {
		if errors.Is(err, io.EOF) {
			_, err = stream.Recv() // preserve a server refusal hidden behind Send's EOF
		}
		return uploadProblem(ctx, err)
	}
	initial, err := stream.Recv()
	if err != nil {
		return uploadProblem(ctx, err)
	}
	verified, problem := uploadStatus(header, initial, nil)
	if problem != nil {
		return problem
	}
	if onStatus != nil {
		onStatus(initial)
	}
	if verified {
		return nil // replay needs no local carrier once the Host has verified it
	}
	file, err := os.Open(path)
	if err != nil {
		return exit.Named(exit.Structural, "local_package_wheel_unreadable", "cannot read captured wheel: %s", err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() != int64(header.File.Length) {
		return exit.Named(exit.Conflict, "local_package_wheel_changed", "captured wheel changed before upload")
	}
	if _, err := file.Seek(int64(initial.ReceivedBytes), io.SeekStart); err != nil {
		return exit.Internalf("cannot resume captured wheel: %s", err)
	}

	// One Send owner and one Recv owner avoid gRPC stream races. Credits are
	// returned only after the matching durable ACK, so buffers and speculative
	// work stay bounded even if the receiver stops making progress.
	credits := make(chan struct{}, uploadWindow)
	for range uploadWindow {
		credits <- struct{}{}
	}
	ends := make(chan uint64, uploadWindow)
	sent := make(chan error, 1)
	done := make(chan struct{})
	go func() {
		defer close(done)
		defer close(ends)
		sent <- sendUpload(ctx, stream, file, initial.ReceivedBytes, header.File.Length, credits, ends)
	}()
	defer func() {
		cancel()
		<-done // do not close the stream or file while Send still owns them
	}()
	for end := range ends {
		ack, err := stream.Recv()
		if err != nil {
			return uploadProblem(ctx, err)
		}
		verified, problem := uploadStatus(header, ack, &end)
		if problem != nil {
			return problem
		}
		if onStatus != nil {
			onStatus(ack)
		}
		if verified {
			return nil
		}
		credits <- struct{}{}
	}
	err = <-sent
	if errors.Is(err, io.EOF) {
		// There is no other receiver now. Read the server's terminal status rather
		// than turning FailedPrecondition/PermissionDenied into a retryable EOF.
		_, err = stream.Recv()
	}
	return uploadProblem(ctx, err)
}

func sendUpload(ctx context.Context, stream pb.PodHost_LocalPackageUploadClient, file io.Reader,
	offset, length uint64, credits <-chan struct{}, ends chan<- uint64,
) error {
	for offset < length {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-credits:
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		chunk := make([]byte, min(uint64(uploadChunkBytes), length-offset))
		if _, err := io.ReadFull(file, chunk); err != nil {
			return exit.Named(exit.Conflict, "local_package_wheel_changed", "captured wheel changed during upload: %s", err)
		}
		if err := stream.Send(&pb.LocalPackageUploadFrame{Body: &pb.LocalPackageUploadFrame_Chunk{
			Chunk: &pb.LocalPackageUploadChunk{Offset: offset, Data: chunk},
		}}); err != nil {
			return err
		}
		offset += uint64(len(chunk))
		select {
		case ends <- offset:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

func uploadStatus(header *pb.LocalPackageUploadHeader, ack *pb.LocalPackageFileStatus, expected *uint64) (bool, *exit.Error) {
	file := header.File
	if ack == nil || ack.OperationId != header.OperationId ||
		!bytes.Equal(ack.Digest, file.Digest) || ack.Filename != file.Filename || ack.Length != file.Length ||
		ack.ReceivedBytes > file.Length || (expected != nil && ack.ReceivedBytes != *expected) {
		return false, exit.Named(exit.Conflict, "local_package_upload_identity_changed", "worker returned another captured file or unexpected upload offset")
	}
	if ack.State == pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_REFUSED {
		return false, exit.Named(exit.Failed, ack.SafeCode, "worker refused captured wheel %s: %s", file.Filename, ack.SafeDetail)
	}
	if ack.State == pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED {
		if ack.ReceivedBytes != file.Length {
			return false, exit.Named(exit.Conflict, "local_package_upload_incomplete", "worker verified an incomplete captured wheel")
		}
		if ack.SafeCode == "" {
			return true, nil
		}
	}
	if ack.State != pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_RECEIVING || ack.SafeCode != "" {
		return false, exit.Named(exit.Unavailable, "local_package_upload_stopped", "worker stopped captured wheel upload: %s", ack.SafeDetail)
	}
	if ack.ReceivedBytes == file.Length {
		return false, exit.Named(exit.Conflict, "local_package_upload_unverified", "worker has all bytes but did not verify the captured wheel")
	}
	return false, nil
}

func uploadProblem(ctx context.Context, err error) *exit.Error {
	var problem *exit.Error
	if errors.As(err, &problem) {
		return problem
	}
	if errors.Is(ctx.Err(), context.Canceled) {
		return exit.New(exit.Canceled, "unpublished package upload was cancelled")
	}
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return exit.New(exit.Deadline, "unpublished package upload deadline expired")
	}
	switch status.Code(err) {
	case codes.OK, codes.Unavailable, codes.DeadlineExceeded, codes.Canceled,
		codes.ResourceExhausted, codes.Aborted, codes.Internal, codes.Unknown:
		return exit.Named(exit.Unavailable, "local_package_upload_interrupted", "unpublished upload interrupted; acknowledged bytes can be resumed: %s", err)
	default:
		return exit.Named(exit.Structural, "local_package_upload_refused", "worker refused unpublished wheel upload: %s", status.Convert(err).Message())
	}
}
