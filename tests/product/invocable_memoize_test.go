package producttest

import (
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
)

func TestInvocableMemoizeIsTheOnlyPublicOptIn(t *testing.T) {
	const document = `{"format":"cozy.package.interface/1","application":"private_ops:app","entrypoints":[],"jobs":[{"name":"compute","models":[],"request":{"fields":[]},"result":{"fields":[]},"publishes":false,"weights_outputs":[],"invocable":{"context":"ctx","module":"private_ops","export":"compute","parameters":[],"defaults":{},"type_names":{},"enum_members":{},"memoize":true,"capabilities":[]}}]}`
	for _, value := range []string{"true", "false", "omitted"} {
		t.Run(value, func(t *testing.T) {
			body := strings.Replace(document, `"memoize":true`, `"memoize":`+value, 1)
			if value == "omitted" {
				body = strings.Replace(document, `"memoize":true,`, "", 1)
			}
			surface, problem := launch.DecodePackageInterface([]byte(body))
			fatal(t, problem)
			job, problem := surface.Function("compute")
			fatal(t, problem)
			if job.Invocable == nil || job.Invocable.Memoize != (value == "true") {
				t.Fatalf("memoization opt-in changed: %+v", job.Invocable)
			}
		})
	}
	// Only memoize opts in. A retired or unknown member is ignored, and a mistyped one refuses.
	for member, memoize := range map[string]bool{`"reusable":true`: false, `"memoize":true,"reusable":true`: true} {
		surface, problem := launch.DecodePackageInterface([]byte(strings.Replace(document, `"memoize":true`, member, 1)))
		fatal(t, problem)
		job, problem := surface.Function("compute")
		fatal(t, problem)
		if job.Invocable.Memoize != memoize {
			t.Fatalf("%s changed the memoize opt-in to %v", member, job.Invocable.Memoize)
		}
	}
	if _, problem := callableOf(t, []byte(strings.Replace(document, `"memoize":true`, `"memoize":"true"`, 1)), "compute"); problem == nil {
		t.Fatal("a string memoize opt-in was admitted")
	}
}
