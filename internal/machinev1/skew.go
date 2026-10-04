package machinev1

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy/internal/build"
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// A cozy and a machine a release apart can share no call for some verb. Each way round is one
// message naming the side to upgrade, never the transport's own words.

// Newer is a cozy.worker.v1 call a machine serving cozy.machine.v1 does not serve: the machine
// is newer than this cozy. Its gRPC status stays the machine's UNIMPLEMENTED.
type Newer struct{ cause error }

func (n *Newer) Error() string              { return n.cause.Error() }
func (n *Newer) Unwrap() error              { return n.cause }
func (n *Newer) GRPCStatus() *status.Status { return status.Convert(n.cause) }

// SkewInterceptors mark a worker.v1 call that answers UNIMPLEMENTED as Newer when the same
// connection serves cozy.machine.v1 Status. The probe runs only on that failure.
func SkewInterceptors() []grpc.DialOption {
	unary := func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		return skew(ctx, cc, method, invoker(ctx, method, req, reply, cc, opts...))
	}
	stream := func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
		s, err := streamer(ctx, desc, cc, method, opts...)
		if err != nil {
			return nil, skew(ctx, cc, method, err)
		}
		return &skewStream{ClientStream: s, ctx: ctx, cc: cc, method: method}, nil
	}
	return []grpc.DialOption{grpc.WithChainUnaryInterceptor(unary), grpc.WithChainStreamInterceptor(stream)}
}

type skewStream struct {
	grpc.ClientStream
	ctx    context.Context
	cc     *grpc.ClientConn
	method string
}

func (s *skewStream) RecvMsg(m any) error {
	return skew(s.ctx, s.cc, s.method, s.ClientStream.RecvMsg(m))
}

func skew(ctx context.Context, cc *grpc.ClientConn, method string, err error) error {
	if status.Code(err) != codes.Unimplemented || !strings.HasPrefix(method, "/cozy.worker.v1.") || !servesV1(ctx, cc) {
		return err
	}
	return &Newer{cause: err}
}

// servesV1 asks the machine's Status without a capability, which answers its identity.
func servesV1(ctx context.Context, cc *grpc.ClientConn) bool {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	stream, err := pb.NewMachineClient(cc).Status(ctx, &pb.StatusRequest{})
	if err == nil {
		_, err = stream.Recv()
	}
	return err == nil || status.Code(err) == codes.Unauthenticated || status.Code(err) == codes.PermissionDenied
}

// NewerThanCozy names this cozy as the side to upgrade.
func NewerThanCozy(format string, args ...any) *exit.Error {
	return exit.Named(exit.Structural, "cozy.upgrade_required", "the machine is newer than this cozy (%s): %s", build.Version, fmt.Sprintf(format, args...)).
		WithRemedy("upgrade cozy: curl -fsSL https://github.com/cozy-creator/cozy/releases/latest/download/install.sh | sh")
}

// Skew reads a failed call as a peer a release apart: Newer is a machine newer than this cozy;
// any other UNIMPLEMENTED is a machine older than this cozy. nil for every other failure.
func Skew(err error) *exit.Error {
	var newer *Newer
	if errors.As(err, &newer) {
		return NewerThanCozy("it no longer serves this call (%s)", status.Convert(err).Message())
	}
	if status.Code(err) != codes.Unimplemented {
		return nil
	}
	return exit.Named(exit.Structural, "machine.upgrade_required",
		"the machine is older than this cozy (%s) and does not serve this call (%s)", build.Version, status.Convert(err).Message()).
		WithRemedy("update the machine: `cozy machine install` on this computer; a rental on an older image is replaced by a new one (`cozy rental new`)")
}
