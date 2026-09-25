package rental

import (
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// ExecutionLifecycleRuntime is the first Runtime release that reads a published
// package's interface from PreparePackageSetRequest.package_interface (wire 61)
// instead of describing it. A PodHost older than 61 drops that field.
var ExecutionLifecycleRuntime = [3]int{0, 18, 25}

// RuntimeUpdateHost refuses installing Runtime `target` behind a PodHost that
// reports a wire minor below 61. PodHost reports its intersection with the
// installed Runtime, so an older Runtime also reads as < 61: the refusal is
// conservative, and such a pod gets the new Runtime from a new worker image.
func RuntimeUpdateHost(target string, hostWireMinor uint32) *exit.Error {
	if hostWireMinor >= pb.ExecutionLifecycleWireMinor || releaseBefore(target, ExecutionLifecycleRuntime) {
		return nil
	}
	return exit.Named(exit.Conflict, "rental.runtime_update_host_too_old",
		"Runtime %s needs worker protocol %d on the pod host; this rental reports %d — nothing was changed",
		target, pb.ExecutionLifecycleWireMinor, hostWireMinor).
		WithRemedy("rent a machine on the current worker image, or update to a Runtime before 0.18.25")
}

// releaseBefore reports whether a PEP 440 release's numeric prefix sorts before
// floor. A pre-release of the floor is not before it, and an unreadable version
// is not before anything: the guard applies to both.
func releaseBefore(version string, floor [3]int) bool {
	parts := strings.SplitN(version, ".", 4)
	if len(parts) < 3 {
		return false
	}
	for i, want := range floor {
		digits := strings.TrimLeft(parts[i], "0123456789")
		got, err := strconv.Atoi(strings.TrimSuffix(parts[i], digits))
		if err != nil {
			return false
		}
		if got != want {
			return got < want
		}
	}
	return false
}
