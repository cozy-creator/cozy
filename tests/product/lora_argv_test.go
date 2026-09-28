package producttest

import (
	"reflect"
	"testing"

	"github.com/alecthomas/kong"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/launch"
)

// Exercise the actual CLI grammar from argv: ParseLoRAs alone cannot observe
// Kong's default comma splitting for []string flags.
func TestLoRAArgvKeepsStrengthInsideEachRepeatedFlag(t *testing.T) {
	values := []string{
		"model:fl2va_dit=paul/minimax-h3-spatial-physics-lora,0.3",
		"model:fl2va_dit=paul/minimax-h3-wushu-action-v7-lora,0",
		"model:fl2va_dit=paul/minimax-h3-spatial-physics-lora@1.0.0/fp16,-0.25",
	}
	for _, equals := range []bool{false, true} {
		var grammar cli.CLI
		parser, err := kong.New(&grammar)
		must(t, err)
		argv := []string{"run", "paul/minimax-h3/fl2va", "--rental=owned-test-rental"}
		for _, value := range values {
			if equals {
				argv = append(argv, "--lora="+value)
			} else {
				argv = append(argv, "--lora", value)
			}
		}
		_, err = parser.Parse(argv)
		must(t, err)
		got := grammar.Run.Execute.LoRAs
		if !reflect.DeepEqual(got, values) {
			t.Fatalf("raw argv changed adapter stack: got %#v want %#v", got, values)
		}
		rows, problem := launch.ParseLoRAs(&launch.Entrypoint{Name: "fl2va", Models: []launch.Slot{{Param: "model", Path: "fl2va.models.model"}}}, got)
		fatal(t, problem)
		if len(rows) != 3 || rows[0].Scale != "0.3" || rows[1].Scale != "0" || rows[2].Scale != "-0.25" {
			t.Fatalf("raw argv changed ordered scales: %+v", rows)
		}
	}
}
