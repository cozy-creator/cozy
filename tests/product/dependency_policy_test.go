package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

const dependencyPolicyCode = "project_dependency_version_policy"

func TestAuthoredDependencyVersionPolicy(t *testing.T) {
	var cases []struct {
		Requirement string `json:"requirement"`
		Accepted    bool   `json:"accepted"`
	}
	raw, err := os.ReadFile("testdata/dependency-policy.json")
	must(t, err)
	must(t, json.Unmarshal(raw, &cases))
	cases = append(cases, struct {
		Requirement string `json:"requirement"`
		Accepted    bool   `json:"accepted"`
	}{"foo[bar] (>=2.46.4,<3); python_version < '3.1'", true})
	for _, c := range cases {
		t.Run(c.Requirement, func(t *testing.T) {
			for _, optional := range []bool{false, true} {
				project := fixturePyproject(c.Requirement)
				if optional {
					project = fixturePyproject() + fmt.Sprintf("\n[project.optional-dependencies]\nunused = [%q]\n", c.Requirement)
				}
				tree := fixtureTree(t, project, minimalFixtureLock)
				pack, problem := packagepublish.PrepareFrom(tree)
				if c.Accepted {
					fatal(t, problem)
					pack.Close()
				} else if problem == nil || problem.Name != dependencyPolicyCode || !strings.Contains(problem.Message, c.Requirement) {
					t.Fatalf("optional=%v requirement %q: wanted actionable policy refusal, got %+v", optional, c.Requirement, problem)
				}
			}
		})
	}
}

func TestDependencyPolicyCLIRefusesBeforeBuildOrPublication(t *testing.T) {
	project := fixtureTree(t, fixturePyproject("pydantic-core==2.46.4"), minimalFixtureLock)
	root := t.TempDir()
	// No login or running daemon: a publication attempt must diagnose the author
	// dependency before reaching the account, build backend, or publication API.
	command := exec.Command(cozyBin, "package", "publish", "--json")
	command.Dir, command.Env = project, childEnv(t, root)
	out, err := command.CombinedOutput()
	if err == nil || !strings.Contains(string(out), `"code":"`+dependencyPolicyCode+`"`) ||
		!strings.Contains(string(out), "pydantic-core>=2.46.4,<3") {
		t.Fatalf("ordinary publication did not refuse the author pin before login/build: %v\n%s", err, out)
	}
	code, output := runCozy(t, root, "package", "install", project, "--editable", "--json")
	if code != 1 || !strings.Contains(output, `"code":"`+dependencyPolicyCode+`"`) {
		t.Fatalf("editable capture did not refuse the author pin: %d\n%s", code, output)
	}
	// Script metadata uses the same boundary before uv lock could fetch anything.
	script := filepath.Join(t.TempDir(), "prepare.py")
	must(t, os.WriteFile(script, []byte("# /// script\n# dependencies = [\"pydantic-core==2.46.4\"]\n# ///\nasync def main(ctx):\n    return 1\n"), 0600))
	code, output = runCozy(t, root, "run", script, "--json")
	if code != 1 || !strings.Contains(output, `"code":"`+dependencyPolicyCode+`"`) {
		t.Fatalf("script capture did not refuse before resolution: %d\n%s", code, output)
	}
}

func TestDependencyPolicySuggestsPublicVersionFloors(t *testing.T) {
	for _, version := range []string{"0.18.4+dev.abc", "1!0.18.4+dev.abc"} {
		t.Run(version, func(t *testing.T) {
			project := fixtureTree(t, fixturePyproject("cozy-runtime=="+version), minimalFixtureLock)
			_, problem := packagepublish.PrepareFrom(project)
			if problem == nil || problem.Name != dependencyPolicyCode {
				t.Fatalf("expected exact-pin refusal, got %v", problem)
			}
			public := strings.SplitN(version, "+", 2)[0]
			if !strings.Contains(problem.Remedy, "cozy-runtime>="+public) || strings.Contains(problem.Remedy, "+dev") {
				t.Fatalf("refusal suggests an invalid local version floor: %s", problem.Remedy)
			}
			if strings.Contains(version, "!") && !strings.Contains(problem.Remedy, ",<1!1") {
				t.Fatalf("suggested range crossed epochs: %s", problem.Remedy)
			}
		})
	}
}

func TestDependencyPolicyChecksDynamicWheelRequirements(t *testing.T) {
	// The project declares no dependencies; its backend injects one during build.
	// Use a real PEP 517 backend through uv, without a package index or imports.
	project := fixtureTree(t, `[project]
name = "cozy-fixture-package"
version = "1.0.0"
dynamic = ["dependencies"]
[build-system]
requires = []
build-backend = "backend"
backend-path = ["."]
`, minimalFixtureLock)
	must(t, os.WriteFile(filepath.Join(project, "backend.py"), []byte(`from pathlib import Path
import zipfile

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    filename = "cozy_fixture_package-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(Path(wheel_directory) / filename, "w") as wheel:
        wheel.writestr("cozy_fixture_package-1.0.0.dist-info/METADATA", "Metadata-Version: 2.3\nName: cozy-fixture-package\nVersion: 1.0.0\nRequires-Dist: pydantic-core==2.46.4; extra == 'unused'\n\n")
    return filename
`), 0600))
	pack, problem := packagepublish.PrepareFrom(project)
	fatal(t, problem)
	defer pack.Close()
	problem = pack.BuildForPublish(t.Context(), packagepublish.Namespace{Hub: "http://127.0.0.1:1", Account: "proof"})
	if problem == nil || problem.Name != dependencyPolicyCode || !strings.Contains(problem.Message, "project wheel Requires-Dist") {
		t.Fatalf("backend-injected inactive optional pin bypassed the policy: %+v", problem)
	}
}
