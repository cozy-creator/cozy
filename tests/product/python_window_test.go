package producttest

import (
	"fmt"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
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

func TestRentalPythonAdmissionUsesAvailableExecutors(t *testing.T) {
	inventory := &pb.ImageInventory{Python: "3.12.12", Interpreters: []*pb.PythonInterpreter{
		{Version: "3.12.12", Abi: "cp312"}, {Version: "3.13.11", Abi: "cp313"}, {Version: "3.14.2", Abi: "cp314"},
	}}
	if selected, reason := launch.InventoryPython(inventory, ">=3.13,<3.14", ""); reason != "" || selected != "3.13.11" {
		t.Fatalf("control3.12 hid compatible executor: %s %s", selected, reason)
	}
	if _, reason := launch.InventoryPython(inventory, ">=3.15", ""); reason == "" {
		t.Fatal("future executor admitted")
	}
	if _, reason := launch.InventoryPython(inventory, ">=3.12", "3.13.9"); reason == "" {
		t.Fatal("captured patch silently replaced")
	}
	candidates := rental.Purchases([]hub.RentalSKU{{Name: "cpu", AcceleratorModel: "CPU", BaseWorkerProfile: "python3.12-cpu-linux-x86", PythonInterpreters: inventory.Interpreters}}, nil, false, true, rental.Constraints{RequiresPython: ">=3.13,<3.14", PythonVersion: "3.13.11"})
	if len(candidates) != 1 || candidates[0].Verdict != "" {
		t.Fatalf("multi-interpreter base rejected before rental: %+v", candidates)
	}
}

func TestRentalPythonAdmissionRefusesUnmeasuredOrWrongABI(t *testing.T) {
	for _, executors := range [][]*pb.PythonInterpreter{nil, {{Version: "3.13.11", Abi: "cp312"}}, {{Version: "3.13.11", Abi: "cp313t"}}} {
		candidates := rental.Purchases([]hub.RentalSKU{{Name: "cpu", AcceleratorModel: "CPU", BaseWorkerProfile: "python3.13-cpu-linux-x86", PythonInterpreters: executors}}, nil, false, true, rental.Constraints{RequiresPython: ">=3.12", PythonVersion: "3.13.11"})
		if len(candidates) != 1 || candidates[0].Verdict == "" {
			t.Fatalf("unmeasured or mismatched executor admitted: %+v", executors)
		}
	}
}
