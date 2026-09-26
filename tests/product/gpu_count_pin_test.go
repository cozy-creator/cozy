package producttest

import (
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"strings"
	"testing"
)

// The public submission path must preserve an exact group without changing the
// four-device rental inventory or the old uncounted width behavior.
func TestCountedSubmissionPinsExactGroupOnLargerRental(t *testing.T) {
	for _, count := range []int{0, 1, 2, 4} {
		t.Run(string(rune('0'+count)), func(t *testing.T) {
			pod := &fakePod{serve: true, deviceCount: 4, preparedPlacement: modelBearingPlacement(t)}
			connection, certPath := startFakePod(t, t.TempDir(), pod)
			o := hostOwner(t, "counted-group", rentalWidthWiring(t, pod, connection, certPath, 4))
			_, _, problem := o.c.Submit(orchestrator.Submission{
				IdemKey: "counted-group", Package: "cozy/h3-package", Entrypoint: "tile", Release: "1.1.2",
				Payload: []byte(`{"prompt":"a fox"}`), Outputs: []string{"video"}, Worker: podRental, Rental: true, RentalRequired: true,
				Models: []orchestrator.ModelRef{{GPUs: count, Package: "cozy/h3-package", Slot: "tile.models.model", Model: "source/h3", Release: "1.0.0", Lane: "bf16", Manifest: "sha256:" + strings.Repeat("1", 64), ManifestLength: 164}},
			})
			fatal(t, problem)
			state, placement := awaitPlacementSet(t, o, pod)
			want := count
			if want == 0 {
				want = 4
			}
			if len(state.DevicePins) != 1 || state.DevicePins[0].PlacementId != placement || len(state.DevicePins[0].DeviceOrdinals) != want {
				t.Fatalf("count %d selected pins %v", count, state.DevicePins)
			}
		})
	}
}
