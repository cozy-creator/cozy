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
	for _, member := range []string{`"reusable":true`, `"memoize":"true"`, `"memoize":true,"reusable":true`} {
		body := strings.Replace(document, `"memoize":true`, member, 1)
		if _, problem := launch.DecodePackageInterface([]byte(body)); problem == nil {
			t.Fatalf("noncanonical public opt-in accepted: %s", member)
		}
	}
}
