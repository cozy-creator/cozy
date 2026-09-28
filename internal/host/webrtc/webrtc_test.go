package webrtc_test

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	"github.com/cozy-creator/cozy/internal/host/webrtc/webrtctest"
)

// Every test runs a real listener on loopback and a real pion ICE-TCP client over real TCP;
// outputs are real files, appended in place or replaced by rename, before their entries.

type harness struct {
	t   *testing.T
	ctx context.Context
	m   *webrtctest.Machine
	srv webrtctest.Server
	key ed25519.PrivateKey
}

func newHarness(t *testing.T) *harness {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	t.Cleanup(cancel)
	public, key, _ := ed25519.GenerateKey(rand.Reader)
	m := webrtctest.NewMachine(t.TempDir(), public)
	return &harness{t: t, ctx: ctx, m: m, srv: webrtctest.Serve(t, "127.0.0.1", m), key: key}
}

// grant mints a capability for run 7 of this machine, valid for an hour unless edit says otherwise.
func (h *harness) grant(edit func(*capability.Grant)) string {
	g := capability.Grant{Machine: h.srv.Machine, Run: "7", Expires: time.Now().Add(time.Hour).Unix()}
	if edit != nil {
		edit(&g)
	}
	token, err := capability.Mint(h.key, g)
	if err != nil {
		h.t.Fatal(err)
	}
	return token
}

func (h *harness) dial() *webrtctest.Client {
	c, err := webrtctest.Dial(h.ctx, h.srv.Addr, h.srv.Fingerprint, webrtctest.Options{})
	if err != nil {
		h.t.Fatal(err)
	}
	h.t.Cleanup(func() { c.Close() })
	return c
}

// open is a welcomed session with credit.
func (h *harness) open(credit uint64) *webrtctest.Client {
	c := h.dial()
	h.send(c, map[string]any{"t": "hello", "v": 1, "cap": h.grant(nil)})
	h.expect(c, "welcome")
	h.send(c, map[string]any{"t": "credit", "bytes": credit})
	return c
}

func (h *harness) send(c *webrtctest.Client, record any) {
	if err := c.Send(record); err != nil {
		h.t.Fatal(err)
	}
}

func (h *harness) expect(c *webrtctest.Client, t string) webrtctest.Message {
	h.t.Helper()
	m, _ := c.Recv(h.ctx)
	if m.T != t {
		h.t.Fatalf("expected %s, got %s %s", t, m.T, m.Raw)
	}
	return m
}

// body reads one stream's binary messages from its open to its end, checking every offset.
func (h *harness) body(c *webrtctest.Client) ([]byte, webrtctest.Message) {
	h.t.Helper()
	open := h.expect(c, "open")
	var got []byte
	for {
		m, _ := c.Recv(h.ctx)
		switch m.T {
		case "data":
			if m.Stream != open.Stream || m.Offset != open.Offset+uint64(len(got)) {
				h.t.Fatalf("data at %d of stream %d; expected %d of %d", m.Offset, m.Stream, open.Offset+uint64(len(got)), open.Stream)
			}
			got = append(got, m.Data...)
		case "end":
			if uint64(len(got)) != open.Length {
				h.t.Fatalf("%d bytes before end; open announced %d", len(got), open.Length)
			}
			return got, m
		default:
			h.t.Fatalf("unexpected %s", m.Raw)
		}
	}
}

func segment(n int, fill byte) []byte { return bytes.Repeat([]byte{fill}, n) }

func TestGetServesARangeOfTheCurrentBytes(t *testing.T) {
	h := newHarness(t)
	seg := [][]byte{segment(100_000, 1), segment(150_000, 2), segment(70_000, 3)}
	h.m.Append("7", "video", -1, seg[0], false)
	h.m.Append("7", "video", -1, seg[1], false)
	h.m.Append("7", "video", -1, seg[2], true)
	c := h.open(1 << 20)

	h.send(c, map[string]any{"t": "get", "id": 1, "run": "7", "output": "video", "offset": 100_000, "length": 150_000})
	got, end := h.body(c)
	if !bytes.Equal(got, seg[1]) {
		t.Fatal("the middle segment differs")
	}
	whole := bytes.Join(seg, nil)
	sum := sha256.Sum256(whole)
	if end.Length != uint64(len(whole)) || end.SHA256 != hex.EncodeToString(sum[:]) {
		t.Fatalf("end %s", end.Raw)
	}

	h.send(c, map[string]any{"t": "get", "id": 2, "run": "7", "output": "video", "offset": 0, "etag": "r2"})
	if m := h.expect(c, "error"); m.Code != "changed" {
		t.Fatalf("a stale etag got %s", m.Raw)
	}
	h.send(c, map[string]any{"t": "get", "id": 3, "run": "7", "output": "video", "offset": 0, "etag": "r3"})
	if got, _ := h.body(c); !bytes.Equal(got, whole) {
		t.Fatal("the whole output differs")
	}
}
