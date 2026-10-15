// Package machinev1 is the CLI's client of a machine's `cozy.machine.v1` API: one pinned TLS
// connection, every call authorized by a short machine-scope Cozy-Cap the owner key signs.
package machinev1

import (
	"context"
	"crypto/ed25519"
	"crypto/tls"
	"errors"
	"io"
	"net"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/backoff"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/protobuf/types/known/emptypb"
)

// ScopeMachine is the cap action that authorizes every call as its signer.
const ScopeMachine = "machine"

// Signer is the owner key: its public half and a signature over a message.
type Signer struct {
	Public ed25519.PublicKey
	Sign   func([]byte) []byte
}

// Client is one machine. Its HTTP/2 receive windows are fixed at 16 MiB per stream and 32 MiB
// per connection: BDP probing from 64 KiB lost 16% to a fixed window on a lossy 160 ms link
// (tensord read-bench); 16 MiB covers 150 Mbit/s at 800 ms.
type Client struct {
	conn    *grpc.ClientConn
	Machine pb.MachineClient
	worker  string
	signer  Signer
	// dial opens another connection to the machine: a lane of ReadRanges. dialUploads opens
	// the one Writes ride.
	dial, dialUploads func() (*grpc.ClientConn, error)
	lanesMu           sync.Mutex
	lanes             []*grpc.ClientConn
	upload            *grpc.ClientConn
	ranges            atomic.Int32
	stop              context.CancelFunc
}

// Dial connects to the machine at addr whose pinned leaf tlsConfig trusts; worker is the
// machine's worker id, which every cap names.
func Dial(addr string, tlsConfig *tls.Config, worker string, signer Signer) (*Client, error) {
	c := &Client{worker: worker, signer: signer}
	options := append(dialing(tlsConfig),
		grpc.WithPerRPCCredentials(caps{c}),
		grpc.WithInitialWindowSize(16<<20), grpc.WithInitialConnWindowSize(32<<20),
		// A ping every 20 s keeps NAT mappings on the path alive through a quiet run, and an
		// unanswered one ends a dead connection so the run is attached again at once.
		grpc.WithKeepaliveParams(keepalive.ClientParameters{Time: 20 * time.Second, Timeout: 20 * time.Second, PermitWithoutStream: true}),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(16<<20), grpc.MaxCallSendMsgSize(16<<20)))
	// Read lanes close after gRPC's idle time. The control and upload connections never close
	// for want of calls: a run's first write after a pause would pay a new TCP and TLS handshake.
	reading := slices.Clone(options)
	c.dial = func() (*grpc.ClientConn, error) { return grpc.NewClient(addr, reading...) }
	options = append(options, grpc.WithIdleTimeout(0))
	conn, err := grpc.NewClient(addr, options...)
	if err != nil {
		return nil, err
	}
	c.conn, c.Machine = conn, pb.NewMachineClient(conn)
	// Uploads ride their own connection, its socket holding little unsent: over a slow uplink a
	// keepalive ping then waits behind kilobytes, not seconds, of an upload, and a connection an
	// upload loses takes no run or status with it.
	uploading := slices.Concat(options, []grpc.DialOption{grpc.WithContextDialer(func(ctx context.Context, addr string) (net.Conn, error) {
		conn, err := raceDial(ctx, addr)
		if err == nil {
			boundUnsent(conn)
		}
		return conn, err
	})})
	c.dialUploads = func() (*grpc.ClientConn, error) { return grpc.NewClient(addr, uploading...) }
	warming, stop := context.WithCancel(context.Background())
	c.stop = stop
	go c.warm(warming)
	return c, nil
}

// warmEvery paces what an open upload connection carries while no write uses it: less than
// TCP's retransmission timeout (200 ms at least), so the kernel never finds it idle and the
// next run's inputs leave at the rate the last ones reached, not from slow start. Over 150 ms,
// a 475 KB write after a 2-minute pause took 6 round trips idle and 1 kept warm.
const warmEvery = 100 * time.Millisecond

// warm sends a call the machine answers at once (it serves no such method) on the upload
// connection while it is open and ready. Nothing is dialed for it.
func (c *Client) warm(ctx context.Context) {
	tick := time.NewTicker(warmEvery)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
		c.lanesMu.Lock()
		upload := c.upload
		c.lanesMu.Unlock()
		if upload != nil && upload.GetState() == connectivity.Ready {
			_ = upload.Invoke(ctx, "/cozy.machine.v1.Machine/Warm", &emptypb.Empty{}, &emptypb.Empty{})
		}
	}
}

// synWait is how long a connection attempt waits for the machine's SYN-ACK before it also
// sends one on a new socket: the kernel's first SYN retransmission comes after as long.
const synWait = time.Second

// dialing is how every connection to a machine is made. A fresh pod's public port can stop
// answering SYNs for a moment (seen for 2 s about 23 s after readiness, as its provider
// published the ports), and a socket whose SYN went unanswered then may stay unanswered. So
// an attempt that hears no SYN-ACK within synWait races a new socket, and another each
// synWait after, taking whichever the machine answers first. It gives up only as gRPC's
// connect did, after 20 s with no SYN-ACK on any socket; a refusal ends it at once, and the
// next attempt follows within a second. An answered connection keeps gRPC's 20 s for TLS
// and HTTP/2, and an established one its keepalive.
func dialing(tlsConfig *tls.Config) []grpc.DialOption {
	return []grpc.DialOption{
		grpc.WithTransportCredentials(credentials.NewTLS(tlsConfig)),
		grpc.WithContextDialer(raceDial),
		grpc.WithConnectParams(grpc.ConnectParams{
			Backoff:           backoff.Config{BaseDelay: 200 * time.Millisecond, Multiplier: 1.6, Jitter: 0.2, MaxDelay: time.Second},
			MinConnectTimeout: 20 * time.Second,
		}),
	}
}

// raceDial connects on the first socket the machine answers; see dialing.
func raceDial(ctx context.Context, addr string) (net.Conn, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	type dialed struct {
		conn net.Conn
		err  error
	}
	results := make(chan dialed)
	race := func() {
		go func() {
			conn, err := dialSocket(ctx, addr)
			select {
			case results <- dialed{conn, err}:
			case <-ctx.Done():
				if conn != nil {
					conn.Close()
				}
			}
		}()
	}
	race()
	another := time.NewTicker(synWait)
	defer another.Stop()
	for {
		select {
		case r := <-results:
			// A socket ends before ctx only when the machine is reached or something answered
			// for it (a refusal, an unreachable route): that answer ends the attempt.
			return r.conn, r.err
		case <-another.C:
			race()
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

var dialSocket = func(ctx context.Context, addr string) (net.Conn, error) {
	return (&net.Dialer{}).DialContext(ctx, "tcp", addr)
}

// Close ends the connection and its lanes.
func (c *Client) Close() error {
	if c.stop != nil {
		c.stop()
	}
	c.lanesMu.Lock()
	for i, lane := range c.lanes {
		if lane != nil {
			_ = lane.Close()
			c.lanes[i] = nil
		}
	}
	if c.upload != nil {
		_ = c.upload.Close()
		c.upload = nil
	}
	c.lanesMu.Unlock()
	return c.conn.Close()
}

// Broken is a connection gRPC has given up on for now, or one closed: a caller dials anew
// rather than wait out its reconnect backoff.
func (c *Client) Broken() bool { return broken(c.conn) }

func broken(conn *grpc.ClientConn) bool {
	state := conn.GetState()
	return state == connectivity.TransientFailure || state == connectivity.Shutdown
}

// uploads is the connection Writes ride, dialed when first used and again once it broke.
// Without one of its own (a client not made by Dial), the client's.
func (c *Client) uploads() (*grpc.ClientConn, pb.MachineClient) {
	if c.dialUploads == nil {
		return nil, c.Machine
	}
	c.lanesMu.Lock()
	defer c.lanesMu.Unlock()
	if c.upload != nil && broken(c.upload) {
		_ = c.upload.Close()
		c.upload = nil
	}
	if c.upload == nil {
		conn, err := c.dialUploads()
		if err != nil {
			return nil, c.Machine
		}
		c.upload = conn
	}
	return c.upload, pb.NewMachineClient(c.upload)
}

// dropUploads closes conn, the uploads' connection a transfer lost, unless it was replaced.
func (c *Client) dropUploads(conn *grpc.ClientConn) {
	c.lanesMu.Lock()
	defer c.lanesMu.Unlock()
	if conn != nil && c.upload == conn {
		_ = c.upload.Close()
		c.upload = nil
	}
}

// Cap mints a cap for this machine: machine scope (run empty) or one run's outputs.
func (c *Client) Cap(run string, outputs []string, lifetime time.Duration) (string, error) {
	grant := capability.Grant{Machine: c.worker, Run: run, Outputs: outputs, Expires: time.Now().Add(lifetime).Unix()}
	if run == "" {
		grant.Action = ScopeMachine
	}
	return capability.MintSigned(c.signer.Public, c.signer.Sign, grant)
}

type caps struct{ c *Client }

func (k caps) GetRequestMetadata(context.Context, ...string) (map[string]string, error) {
	token, err := k.c.Cap("", nil, 5*time.Minute)
	if err != nil {
		return nil, err
	}
	return map[string]string{"authorization": "Cozy-Cap " + token}, nil
}
func (caps) RequireTransportSecurity() bool { return true }

// Run submits spec under id when id is new (nil spec attaches) and streams the run's log
// after the cursor. Closing the stream never cancels the run.
func (c *Client) Run(ctx context.Context, id string, after uint64, spec *pb.RunSpec) (grpc.ServerStreamingClient[pb.RunEvent], error) {
	return c.Machine.Run(ctx, &pb.RunRequest{Id: id, After: after, Spec: spec})
}

// Control cancels, pauses or resumes a run.
func (c *Client) Control(ctx context.Context, id string, action pb.Action) (*pb.RunState, error) {
	return c.Machine.Control(ctx, &pb.ControlRequest{Id: id, Action: action})
}

// ReadOutput copies an output's bytes from offset into w; with rev set, it is refused
// (FailedPrecondition) once the output has moved past that revision.
func (c *Client) ReadOutput(ctx context.Context, run, output string, index uint32, offset, rev uint64, w io.Writer) (*pb.ReadFrame, int64, error) {
	return c.read(ctx, &pb.ReadRequest{Target: &pb.ReadRequest_Output{Output: &pb.OutputTarget{Run: run, Output: output, Index: index}}, Offset: offset, IfRev: rev}, w)
}

// ReadMember copies one file of a tree output (by its manifest path) into w, at revision rev.
func (c *Client) ReadMember(ctx context.Context, run, output string, index uint32, member string, rev uint64, w io.Writer) (*pb.ReadFrame, int64, error) {
	return c.read(ctx, &pb.ReadRequest{Target: &pb.ReadRequest_Output{Output: &pb.OutputTarget{Run: run, Output: output, Index: index, Member: member}}, IfRev: rev}, w)
}

// ReadTriage copies a failed run's triage bundle into w.
func (c *Client) ReadTriage(ctx context.Context, run string, w io.Writer) (*pb.ReadFrame, int64, error) {
	return c.read(ctx, &pb.ReadRequest{Target: &pb.ReadRequest_Triage{Triage: run}}, w)
}

// ReadLog copies a machine log's newest tail bytes (0: all) into w.
func (c *Client) ReadLog(ctx context.Context, name string, tail uint64, w io.Writer) (*pb.ReadFrame, int64, error) {
	return c.read(ctx, &pb.ReadRequest{Target: &pb.ReadRequest_Log{Log: name}, Tail: tail}, w)
}

// read copies the bytes after the first frame into w and fails when fewer arrive than that
// frame announced or w takes fewer than it was given.
func (c *Client) read(ctx context.Context, request *pb.ReadRequest, w io.Writer) (*pb.ReadFrame, int64, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	stream, err := c.lane(0).Read(ctx, request)
	if err != nil {
		return nil, 0, err
	}
	meta, err := stream.Recv()
	if err != nil {
		return nil, 0, err
	}
	var written int64
	for {
		frame, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			if request.Offset > meta.Length || uint64(written) != meta.Length-request.Offset {
				return meta, written, io.ErrUnexpectedEOF
			}
			return meta, written, nil
		}
		if err != nil {
			return meta, written, err
		}
		n, err := w.Write(frame.GetData())
		written += int64(n)
		if err == nil && n != len(frame.GetData()) {
			err = io.ErrShortWrite
		}
		if err != nil {
			return meta, written, err
		}
	}
}
