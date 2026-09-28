package host

import "context"

// serveWebRTC serves browser media on the granted port; this build has no WebRTC server
// (cl-633 adds it), so nothing listens and the receipt names no port.
func (m *Machine) serveWebRTC(context.Context) error { return nil }

// webrtcReceipt is the receipt's additive member naming the WebRTC port, when served.
func (m *Machine) webrtcReceipt() map[string]any { return nil }
