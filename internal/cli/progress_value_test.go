package cli

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
)

// THE CELL CARRIES ONE NUMBER AND THE STAGE. It used to carry four facts --
// `35% overall (~2s) · denoise 90% stage` -- which asked a reader to hold two
// denominators in mind to learn one thing. A stage-scoped percent answers a
// question nobody asks of a list, and a remaining estimate moves faster than the
// row containing it.
func TestProgressValueIsOneNumberAndTheStage(t *testing.T) {
	frac := func(f float64) *float64 { return &f }
	ms := func(v int64) *int64 { return &v }
	pos := func(v int64) *int64 { return &v }

	for _, row := range []struct {
		name string
		life api.Lifecycle
		want string
	}{
		{
			name: "stage and overall",
			life: api.Lifecycle{Status: "in_progress", ProgressStage: "denoise",
				OverallFraction: frac(0.35), StageFraction: frac(0.90), RemainingMS: ms(2000)},
			want: "denoise 35%",
		},
		{
			name: "counted steps do not reach the cell",
			life: api.Lifecycle{Status: "in_progress", ProgressStage: "denoise",
				OverallFraction: frac(0.61), Position: pos(19), Total: pos(30)},
			want: "denoise 61%",
		},
		{
			name: "overall with no stage",
			life: api.Lifecycle{Status: "in_progress", OverallFraction: frac(0.5)},
			want: "50%",
		},
		{
			name: "stage with no overall",
			life: api.Lifecycle{Status: "in_progress", ProgressStage: "decode"},
			want: "decode",
		},
		{
			name: "nothing measured",
			life: api.Lifecycle{Status: "in_progress"},
			want: "-",
		},
	} {
		t.Run(row.name, func(t *testing.T) {
			if got := progressValue(row.life); got != row.want {
				t.Fatalf("progressValue = %q, want %q", got, row.want)
			}
		})
	}
}
