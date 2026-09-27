package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

// Authored requirements are the author's to write. An exact pin, an exclusion or an upper
// bound is prepared and published as written; Tensorhub advises on pins that block patch
// upgrades when it commits the release, and the CLI prints that advice as a warning
// (TestPublishCLIPreservesMajorMinorCompatibility).
func TestAuthoredDependencyPinsArePreparedAsWritten(t *testing.T) {
	var cases []struct {
		Requirement string `json:"requirement"`
	}
	raw, err := os.ReadFile("testdata/dependency-policy.json")
	must(t, err)
	must(t, json.Unmarshal(raw, &cases))
	requirements := []string{"pydantic-core==2.46.4", "foo!=2.46.4", "foo<=2.46.4", "foo===2.46.4",
		"foo~=2.46.4.1", "foo[bar] (>=2.46.4,<3); python_version < '3.1'"}
	for _, c := range cases {
		requirements = append(requirements, c.Requirement)
	}
	for _, requirement := range requirements {
		t.Run(requirement, func(t *testing.T) {
			for _, optional := range []bool{false, true} {
				project := fixturePyproject(requirement)
				if optional {
					project = fixturePyproject() + fmt.Sprintf("\n[project.optional-dependencies]\nunused = [%q]\n", requirement)
				}
				pack, problem := packagepublish.PrepareFrom(fixtureTree(t, project, minimalFixtureLock))
				if problem != nil {
					t.Fatalf("optional=%v requirement %q was refused: %s", optional, requirement, problem.Message)
				}
				pack.Close()
			}
		})
	}
}
