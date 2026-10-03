package machinev1

import (
	"context"
	"crypto/tls"
	"errors"
	"io"

	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

// Status is the machine's current picture: the first frame of its Status stream. An open
// Status stream is not activity, so reading it never keeps a rental alive.
func (c *Client) Status(ctx context.Context) (*pb.StatusFrame, error) {
	return first(ctx, c.Machine, false)
}

// Keepalive resets the machine's idle deadline once; the frame carries the new deadline.
func (c *Client) Keepalive(ctx context.Context) (*pb.StatusFrame, error) {
	return first(ctx, c.Machine, true)
}

// Watch calls each with the machine's picture and again whenever it changes, until ctx ends,
// the machine ends the stream, or each returns an error.
func (c *Client) Watch(ctx context.Context, each func(*pb.StatusFrame) error) error {
	stream, err := c.Machine.Status(ctx, &pb.StatusRequest{})
	if err != nil {
		return err
	}
	for {
		frame, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if err := each(frame); err != nil {
			return err
		}
	}
}

// Identity is what a machine answers anyone, without a capability: worker and boot id,
// version, capabilities and its sealed readiness receipt (empty until sealed).
func Identity(ctx context.Context, addr string, tlsConfig *tls.Config) (*pb.StatusFrame, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(credentials.NewTLS(tlsConfig)))
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	return first(ctx, pb.NewMachineClient(conn), false)
}

func first(ctx context.Context, machine pb.MachineClient, keepalive bool) (*pb.StatusFrame, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	stream, err := machine.Status(ctx, &pb.StatusRequest{Keepalive: keepalive})
	if err != nil {
		return nil, err
	}
	return stream.Recv()
}
