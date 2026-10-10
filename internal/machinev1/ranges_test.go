package machinev1

import (
	"bytes"
	"context"
	"crypto/rand"
	"net"
	"slices"
	"sync"
	"testing"

	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

const rangeFrame = 256 << 10

// rangeMachine serves one output at revision 1 the way a machine's Read does. One that is not
// `ranged` predates ranges: it ignores the length asked for and sends to the end.
type rangeMachine struct {
	pb.UnimplementedMachineServer
	data   []byte
	ranged bool
	// together holds every read after the first until this many are open at once.
	together int
	// stall holds the first read that reaches this offset until its caller leaves.
	stall int64

	mu      sync.Mutex
	opened  *sync.Cond
	asked   []span
	open    int
	stalled bool
}

func (m *rangeMachine) Read(request *pb.ReadRequest, stream grpc.ServerStreamingServer[pb.ReadFrame]) error {
	first, end := int64(request.Offset), int64(len(m.data))
	meta := &pb.ReadFrame{Rev: 1, Length: uint64(len(m.data))}
	if m.ranged {
		if request.Length != 0 {
			end = min(first+int64(request.Length), end)
		}
		meta.End = uint64(end)
	}
	m.mu.Lock()
	m.asked = append(m.asked, span{first, first + int64(request.Length)})
	if request.IfRev > 1 {
		m.mu.Unlock()
		return status.Error(codes.FailedPrecondition, "the output is at revision 1")
	}
	later := len(m.asked) > 1
	m.open++
	m.opened.Broadcast()
	for later && m.open < m.together {
		m.opened.Wait()
	}
	m.together = 0
	m.opened.Broadcast()
	m.mu.Unlock()
	defer func() { m.mu.Lock(); m.open--; m.mu.Unlock() }()
	if err := stream.Send(meta); err != nil {
		return err
	}
	for at := first; at < end; at += rangeFrame {
		m.mu.Lock()
		stall := m.stall != 0 && at <= m.stall && m.stall < at+rangeFrame && !m.stalled
		m.stalled = m.stalled || stall
		m.mu.Unlock()
		if stall {
			<-stream.Context().Done()
			return status.FromContextError(stream.Context().Err()).Err()
		}
		if err := stream.Send(&pb.ReadFrame{Data: m.data[at:min(at+rangeFrame, end)]}); err != nil {
			return err
		}
	}
	return nil
}

func serveRanges(t *testing.T, machine *rangeMachine) *Client {
	machine.opened = sync.NewCond(&machine.mu)
	machine.data = make([]byte, 24<<20)
	rand.Read(machine.data)
	listener := bufconn.Listen(1 << 20)
	server := grpc.NewServer()
	pb.RegisterMachineServer(server, machine)
	go server.Serve(listener)
	conn, err := grpc.NewClient("passthrough:///ranges", grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }))
	must(t, err)
	t.Cleanup(func() { conn.Close(); server.Stop() })
	return &Client{Machine: pb.NewMachineClient(conn)}
}

// memory is a file's bytes; it counts the bytes written to it.
type memory struct {
	mu      sync.Mutex
	bytes   []byte
	written int64
}

func (m *memory) WriteAt(p []byte, at int64) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.written += int64(len(p))
	return copy(m.bytes[at:], p), nil
}

func readRanges(t *testing.T, client *Client, machine *rangeMachine, from int64) *memory {
	t.Helper()
	held := &memory{bytes: make([]byte, len(machine.data))}
	copy(held.bytes, machine.data[:from])
	var reported int64
	whole, err := client.ReadRanges(t.Context(), &pb.OutputTarget{Run: "run", Output: "video"}, 1, from, int64(len(machine.data)), held,
		func(n int64) { reported = max(reported, n) })
	if err != nil || whole != int64(len(machine.data))-from {
		t.Fatalf("the read kept %d bytes: %v", whole, err)
	}
	if !bytes.Equal(held.bytes, machine.data) {
		t.Fatal("the bytes read are not the output's")
	}
	if reported != int64(len(machine.data)) {
		t.Fatalf("the read reported %d bytes in hand of %d", reported, len(machine.data))
	}
	return held
}

// A machine that reads ranges answers the first alone, then the rest side by side, each range
// its lanes' share of the bytes; the next output is asked for in ranges from the start.
func TestABigOutputIsReadInRangesSideBySide(t *testing.T) {
	machine := &rangeMachine{ranged: true, together: Lanes - 1}
	client := serveRanges(t, machine)
	held := readRanges(t, client, machine, 0)
	const share = 3 << 20
	if machine.asked[0] != (span{0, share}) || len(machine.asked) != Lanes {
		t.Fatalf("the machine was asked for %v", machine.asked)
	}
	if held.written != int64(len(machine.data)) {
		t.Fatalf("%d bytes were written for an output of %d", held.written, len(machine.data))
	}
	slices.SortFunc(machine.asked, func(a, b span) int { return int(a.first - b.first) })
	for i, asked := range machine.asked {
		if asked != (span{int64(i) * share, int64(i+1) * share}) {
			t.Fatalf("range %d asked for %v", i, asked)
		}
	}

	// What a cut read kept is not asked for again.
	machine.asked, machine.together = nil, Lanes-1
	readRanges(t, client, machine, 8<<20)
	if len(machine.asked) != Lanes || slices.ContainsFunc(machine.asked, func(s span) bool { return s.first < 8<<20 }) {
		t.Fatalf("the rest was asked for as %v", machine.asked)
	}
}

// A machine from before ranges sends to the end whatever length it is asked for: each lane
// is asked for its share once, and its read is cut where the next lane's began.
func TestAMachineWithoutRangesIsReadInShares(t *testing.T) {
	machine := &rangeMachine{together: Lanes - 1}
	client := serveRanges(t, machine)
	held := readRanges(t, client, machine, 0)
	slices.SortFunc(machine.asked, func(a, b span) int { return int(a.first - b.first) })
	for i, asked := range machine.asked {
		if len(machine.asked) != Lanes || asked.first != int64(i)*3<<20 {
			t.Fatalf("the machine was asked from %v", machine.asked)
		}
	}
	if held.written != int64(len(machine.data)) {
		t.Fatalf("%d bytes were written for an output of %d", held.written, len(machine.data))
	}
}

// A range whose bytes stop coming while its peers' flow is dropped and its rest asked again,
// whether it stopped before its first byte or after.
func TestARangeThatStopsIsAskedAgain(t *testing.T) {
	for name, stall := range map[string]int64{"before its first byte": 6 << 20, "partway": 6<<20 + 2*rangeFrame} {
		t.Run(name, func(t *testing.T) {
			machine := &rangeMachine{ranged: true, stall: stall}
			client := serveRanges(t, machine)
			held := readRanges(t, client, machine, 0)
			asks := 0
			for _, asked := range machine.asked {
				if asked.first == stall {
					asks++
				}
			}
			// The range that began there was asked for twice; one that stopped there, once more.
			if want := map[bool]int{true: 2, false: 1}[stall%(3<<20) == 0]; asks != want {
				t.Fatalf("the rest from %d was asked for %d times, want %d: %v", stall, asks, want, machine.asked)
			}
			if held.written != int64(len(machine.data)) {
				t.Fatalf("%d bytes were written for an output of %d", held.written, len(machine.data))
			}
		})
	}
}

// A machine's refusal is its answer: asking again cannot change it.
func TestARefusedRangeEndsTheRead(t *testing.T) {
	machine := &rangeMachine{ranged: true}
	client := serveRanges(t, machine)
	_, err := client.ReadRanges(t.Context(), &pb.OutputTarget{Run: "run", Output: "video"}, 2, 0, int64(len(machine.data)),
		&memory{bytes: make([]byte, len(machine.data))}, nil)
	if status.Code(err) != codes.FailedPrecondition || len(machine.asked) != 1 {
		t.Fatalf("a refused read answered %v after %d asks", err, len(machine.asked))
	}
}
