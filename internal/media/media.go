// Package media is the OWNER's side of a rented pod's media server (cl-014, ruled #506b):
// the only byte channel there is between this host and a pod, in either direction.
//
// This plane carries per-invocation inputs and outputs only. Package source, plans,
// wheels, and model-object sets are resolved directly by the worker from signed intent;
// there is no package-distribution route on this client or the pod server.
//
// WHY THE POD HOSTS IT and not this host: an owner-side byte plane would have to bind
// off-loopback, which is exactly what `internal/api`'s one-bind-site fence refuses, and it
// would have to mint a bearer for the pod to present, which the no-minted-token fence
// refuses. Putting the plane on the pod resolves both without weakening either — the pod
// already listens off-loopback for its control leg, already holds a provisioned
// credential, and is the machine the bytes have to be on anyway (#506b).
package media

import (
	"github.com/cozy-creator/cozy/internal/secret"
)

// Spec is the pod's media plane as a rental pins it: where it answers, the certificate to
// trust, and the bearer to present. It is the media half of the dial triple — same shape,
// same rules, a different port.
type Spec struct {
	Addr   string       `json:"addr"`
	Token  secret.Value `json:"token"`
	CACert string       `json:"ca_cert"`
}
