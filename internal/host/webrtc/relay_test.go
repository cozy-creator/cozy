package webrtc_test

import (
	"io"
	"net"
	"net/netip"
	"sync"
	"testing"
	"time"
)

// relay is a path between client and machine: it forwards each TCP connection with a
// one-way delay each way (RTT = 2 × delay) and can kill every connection at once.
type relay struct {
	ln    net.Listener
	mu    sync.Mutex
	conns []*net.TCPConn
}

func newRelay(t testing.TB, target netip.AddrPort, delay time.Duration) *relay {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	r := &relay{ln: ln}
	t.Cleanup(func() { ln.Close(); r.kill() })
	go func() {
		for {
			in, err := ln.Accept()
			if err != nil {
				return
			}
			out, err := net.Dial("tcp", target.String())
			if err != nil {
				in.Close()
				continue
			}
			r.mu.Lock()
			r.conns = append(r.conns, in.(*net.TCPConn), out.(*net.TCPConn))
			r.mu.Unlock()
			go forward(out, in, delay)
			go forward(in, out, delay)
		}
	}()
	return r
}

func (r *relay) addr() netip.AddrPort { return r.ln.Addr().(*net.TCPAddr).AddrPort() }

// kill resets every connection, as a dropped path does.
func (r *relay) kill() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, c := range r.conns {
		c.SetLinger(0)
		c.Close()
	}
	r.conns = nil
}

func forward(dst, src net.Conn, delay time.Duration) {
	defer dst.Close()
	if delay == 0 {
		io.Copy(dst, src)
		return
	}
	type chunk struct {
		at   time.Time
		data []byte
	}
	line := make(chan chunk, 1<<16)
	go func() {
		defer close(line)
		buf := make([]byte, 64<<10)
		for {
			n, err := src.Read(buf)
			if err != nil {
				return
			}
			line <- chunk{time.Now().Add(delay), append([]byte(nil), buf[:n]...)}
		}
	}()
	for c := range line {
		time.Sleep(time.Until(c.at))
		if _, err := dst.Write(c.data); err != nil {
			return
		}
	}
}
