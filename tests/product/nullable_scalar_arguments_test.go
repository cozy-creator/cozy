package producttest

import (
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// `seed: int | None` is the common optional scalar. Its bare `key=value` spelling must
// reach the payload typed, through the real CLI, exactly as a plain `int` does.
func TestRunSpellsNullableScalarsFromPlainAssignment(t *testing.T) {
	union := func(name string, branches ...any) map[string]any {
		return map[string]any{"name": name, "type": map[string]any{"union": branches}, "wire": "optional"}
	}
	root := interleavedAssetsRoot(t,
		union("noise_seed", "int", "null"),
		union("strength", "float", "null"),
		union("loop", "bool", "null"),
		union("caption", "str", "null"),
		union("mode", map[string]any{"literal": []any{"fast", "slow"}}, "null"),
		union("mask", map[string]any{"asset": "image"}, "null"),
	)
	picture := filepath.Join(root, "picture.png")
	file, err := os.Create(picture)
	must(t, err)
	must(t, png.Encode(file, image.NewRGBA(image.Rect(0, 0, 2, 2))))
	must(t, file.Close())
	submissions := 0
	submit := func(terms ...string) (*records.Request, int, string) {
		submissions++
		args := append([]string{"--json", "run", "proof/assets/prepare", "prompt=p", "--asset", picture}, terms...)
		return submitRun(t, root, fmt.Sprintf("nullable-%d", submissions), append(args, "--rental-only")...)
	}
	for _, test := range []struct {
		terms []string
		want  string
	}{
		{[]string{"noise_seed=7000", "strength=0.75", "loop=true", "caption=7", "mode=slow"},
			`{"caption":"7","loop":true,"mode":"slow","noise_seed":7000,"prompt":"p","strength":0.75}`},
		{[]string{"noise_seed=null", "strength=null", "loop=null", "caption=null", "mode=null", "mask=null"},
			`{"caption":null,"loop":null,"mask":null,"mode":null,"noise_seed":null,"prompt":"p","strength":null}`},
		{[]string{"strength=2", "caption=a lighthouse"}, `{"caption":"a lighthouse","prompt":"p","strength":2}`},
		{[]string{`noise_seed:=7000`, `caption:="null"`}, `{"caption":"null","noise_seed":7000,"prompt":"p"}`},
	} {
		request, code, out := submit(test.terms...)
		if request == nil {
			t.Fatalf("%v refused: %d %s", test.terms, code, out)
		}
		input := submittedPayload(t, request)
		delete(input, "assets")
		got, _ := json.Marshal(input)
		if string(got) != test.want {
			t.Fatalf("%v composed %s, want %s", test.terms, got, test.want)
		}
	}
	for _, test := range []struct{ term, says string }{
		{"noise_seed=7e3", "declared int|null"},
		{"noise_seed=seven", "declared int|null"},
		{"loop=yes", "declared bool|null"},
		{"mode=medium", "is declared"},
		{"mask=/tmp/mask.png", "--asset"},
	} {
		request, code, out := submit(test.term)
		if code == 0 || request != nil || !strings.Contains(out, test.says) {
			t.Fatalf("%s was not refused loudly: %d %s", test.term, code, out)
		}
	}
}
