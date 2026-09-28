package host

import (
	"context"
	"crypto/tls"
	"fmt"
	"net"
	"strconv"

	"github.com/cozy-creator/cozy/internal/host/outputs"
)

// WebRTCConfig is what the browser media server needs from its machine.
type WebRTCConfig struct {
	Machine string          // the worker id capabilities name
	Leaf    tls.Certificate // the machine leaf, also its DTLS certificate
	Source  outputs.Source
}

// ServeWebRTC serves browser media on the granted port; the WebRTC package sets it. With
// no server built in, or no port granted, the machine serves no WebRTC.
var ServeWebRTC func(ctx context.Context, listener net.Listener, config WebRTCConfig) error

func (m *Machine) serveWebRTC(ctx context.Context) error {
	if m.grant.WebRTCPort == 0 || ServeWebRTC == nil {
		return nil
	}
	listener, err := net.Listen("tcp", net.JoinHostPort(m.grant.ListenHost, strconv.Itoa(m.grant.WebRTCPort)))
	if err != nil {
		return fmt.Errorf("bind the WebRTC port: %w", err)
	}
	go func() {
		defer listener.Close()
		if err := ServeWebRTC(ctx, listener, WebRTCConfig{Machine: m.grant.WorkerID, Leaf: m.id.Leaf, Source: m}); err != nil && ctx.Err() == nil {
			fmt.Fprintln(m.log, "cozy machine: WebRTC stopped serving:", err)
		}
	}()
	return nil
}

// webrtcReceipt is the receipt's additive member naming the WebRTC port, when served.
func (m *Machine) webrtcReceipt() map[string]any {
	if m.grant.WebRTCPort == 0 || ServeWebRTC == nil {
		return nil
	}
	return map[string]any{"port": m.grant.WebRTCPort}
}
