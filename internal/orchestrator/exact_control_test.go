package orchestrator

import (
	"bytes"
	"io"
	"testing"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

func TestRemoteConvergenceRelaysThePersistedPlacementSetBytes(t *testing.T) {
	planDigest := bytes.Repeat([]byte{0x11}, 32)
	setBytes, setDigest, err := canonical.Identity(&pb.PlacementSet{Placements: []*pb.Placement{
		{
			PlacementId: "ra-exact", Spec: &pb.PlacementSpec{
				EndpointReleaseId: "org/model@v1", EnvironmentSpecDigest: bytes.Repeat([]byte{0x22}, 32),
				InstalledEnvironmentReceiptDigest: bytes.Repeat([]byte{0x33}, 32),
				DescriptorDigest:                  bytes.Repeat([]byte{0x44}, 32),
				BindingPlans: []*pb.ArtifactSubject{{Digest: planDigest,
					SubjectId: "sha256:" + bytesToHex(planDigest), Kind: "plan", Length: 17}},
			},
		},
	}})
	if err != nil {
		t.Fatal(err)
	}
	spelled, err := canonical.Spell(setDigest)
	if err != nil {
		t.Fatal(err)
	}
	placement := DesiredPlacement{
		Endpoint: "org/model@rnt-one", InstallID: "ra-exact", PlacementIDValue: "ra-exact",
		ExactPlacementSetDigest: spelled, ExactPlacementSetBytes: setBytes,
	}
	out := make(chan *pb.RecordOwnerFrame, 1)
	s := &session{bootID: "boot-one", generation: 7, out: out}
	w := &worker{instanceID: placement.InstanceID(), spec: WorkerLaunchSpec{Placement: placement},
		subjects: []*pb.ArtifactSubject{{SubjectId: "a local reconstruction would differ"}}}
	c := &Orchestrator{opt: Options{Log: io.Discard}}
	if e := c.converge(s, w, []DesiredPlacement{placement}); e != nil {
		t.Fatal(e)
	}
	frame := <-out
	desired := frame.GetDesiredState().GetPlacementSet()
	if desired == nil || !bytes.Equal(desired.PlacementSetCanonicalBytes, setBytes) ||
		!bytes.Equal(desired.PlacementSetDigest, setDigest) {
		t.Fatalf("desired set was reconstructed: %#v", desired)
	}
}

func bytesToHex(raw []byte) string {
	const digits = "0123456789abcdef"
	out := make([]byte, len(raw)*2)
	for i, value := range raw {
		out[2*i], out[2*i+1] = digits[value>>4], digits[value&15]
	}
	return string(out)
}
