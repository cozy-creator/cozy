package producttest

import (
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
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

// An H3 Turbo-shaped selection reusing a machine already held: base and LoRA both author
// ×2 before ×4. The fleet's own reuse sizing (rental.Standing, as managedRentals asks it)
// pins the widest group both slots author that the rental holds, and the pod is told to
// run the placement on exactly that many cards: all 4 of 4, 2 of 3, 2 of 2.
func TestReusedRentalRunsTheWidestAuthoredGroup(t *testing.T) {
	for _, arm := range []struct{ cards, group int }{{4, 4}, {3, 2}, {2, 2}} {
		t.Run(strconv.Itoa(arm.cards), func(t *testing.T) {
			pod := &fakePod{serve: true, deviceCount: uint32(arm.cards), preparedPlacement: modelBearingPlacement(t)}
			connection, certPath := startFakePod(t, t.TempDir(), pod)
			o := hostOwner(t, "widest-group-"+strconv.Itoa(arm.cards), rentalWidthWiring(t, pod, connection, certPath, arm.cards))
			var models []orchestrator.ModelRef
			for i, slot := range []string{"tile.models.model", "tile.models.lora"} {
				manifest := "sha256:" + strings.Repeat(strconv.Itoa(i+1), 64)
				models = append(models, orchestrator.ModelRef{Package: "cozy/h3-package", Slot: slot, Model: "source/h3",
					Release: "1.0.0", ManifestLength: 164, Ladder: []records.ModelRung{
						{GPU: "4090", GPUs: 2, Lane: "fp8", Manifest: manifest}, {GPU: "4090", GPUs: 4, Lane: "fp8", Manifest: manifest},
						{GPU: "H200", GPUs: 1, Lane: "fp8", Manifest: manifest}}})
			}
			row, problem := o.store.RentalRow(podRental)
			fatal(t, problem)
			candidate := orchestrator.PlacementCandidate{Rental: row.ID, GPUs: row.AcceleratorCount}
			if !rental.Standing(&candidate, models, *row, 0, true, false, false, nil) {
				t.Fatalf("the held rental was excluded: %s", candidate.Verdict)
			}
			_, _, problem = o.c.Submit(orchestrator.Submission{
				IdemKey: "widest-group", Package: "cozy/h3-package", Entrypoint: "tile", Release: "1.1.2",
				Payload: []byte(`{"prompt":"a fox"}`), Outputs: []string{"video"}, Worker: podRental, Rental: true, RentalRequired: true,
				Models: candidate.Models,
			})
			fatal(t, problem)
			state, placement := awaitPlacementSet(t, o, pod)
			if len(state.DevicePins) != 1 || state.DevicePins[0].PlacementId != placement || len(state.DevicePins[0].DeviceOrdinals) != arm.group {
				t.Fatalf("%d cards pinned %v; want one group of %d", arm.cards, state.DevicePins, arm.group)
			}
		})
	}
}
