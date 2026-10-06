package producttest

import (
	"fmt"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestPythonWindowSelectsActualCompatibleExecutor(t *testing.T) {
	inventory := hostruntime.PythonInventory{SupportedMinors: []string{"3.12", "3.13", "3.14"}, Interpreters: []hostruntime.PythonInterpreter{
		{Executable: "/python/3.14", Version: "3.14.2", ABI: "cp314"},
		{Executable: "/python/3.12", Version: "3.12.12", ABI: "cp312"},
		{Executable: "/python/3.13", Version: "3.13.11", ABI: "cp313"},
	}}
	for _, tc := range []struct{ requires, explicit, want string }{
		{"", "3.13.11", "3.13.11"}, {">=3.12", "", "3.12.12"}, {">=3.13,<3.14", "", "3.13.11"}, {">=3.14,<3.15", "", "3.14.2"},
		{">=3.12", "3.13", "3.13.11"}, {">=3.12", "3.14.2", "3.14.2"},
		{">=3.15", "", ""}, {"<3.12", "", ""}, {">=3.12", "3.11", ""}, {">=3.12", "3.13.9", ""},
		{">=3.14", "3.13", ""},
	} {
		got, problem := inventory.Select(tc.requires, tc.explicit)
		if tc.want == "" {
			if problem == nil {
				t.Fatalf("outside-window/conflicting selection accepted: %+v", tc)
			}
			continue
		}
		if problem != nil || got.Version != tc.want {
			t.Fatalf("%+v => %+v, %v", tc, got, problem)
		}
	}
	// Advancing the Runtime-owned policy makes the installed old executor unusable.
	inventory.SupportedMinors = []string{"3.13", "3.14", "3.15"}
	if _, problem := inventory.Select("<3.13", ""); problem == nil {
		t.Fatal("retired 3.12 remains selectable")
	}
}

func TestCapturedRegistryWheelsUseActualPythonABI(t *testing.T) {
	hash := strings.Repeat("1", 64)
	lock := fmt.Sprintf(`version=1
[[package]]
name="root"
version="1.0"
source={editable="."}
[[package]]
name="native"
version="1.0"
source={registry="https://pypi.org/simple"}
wheels=[
 {url="https://files.pythonhosted.org/native-1.0-cp312-cp312-manylinux_2_28_x86_64.whl",hash="sha256:%s",size=100},
 {url="https://files.pythonhosted.org/native-1.0-cp313-cp313-manylinux_2_28_x86_64.whl",hash="sha256:%s",size=100},
 {url="https://files.pythonhosted.org/native-1.0-cp314-cp314-manylinux_2_28_x86_64.whl",hash="sha256:%s",size=100}
]`, hash, hash, hash)
	for _, minor := range []string{"12", "13", "14"} {
		rows, _, problem := packagepublish.CapturedRegistryRows([]byte(lock), "root==1.0\nnative==1.0", "root", "1.0", nil, "3."+minor+".2")
		if problem != nil || len(rows) != 1 || !strings.Contains(rows[0].URL, "-cp3"+minor+"-cp3"+minor+"-") {
			t.Fatalf("Python3.%s selected %+v: %v", minor, rows, problem)
		}
	}
}
