package producttest

import pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"

// Legacy dual-service peers implement Control explicitly. Keep its defaults one
// level below their PodHost defaults so new shared unary methods resolve to one
// Unimplemented response instead of an ambiguous Go selector. Tests of machine
// execution implement those unary methods explicitly.
type controlDefaults struct {
	pb.UnimplementedWorkerControlServer
}
