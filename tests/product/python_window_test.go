package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
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
	// A captured interpreter fixes the ABI minor; any executor of that minor serves it.
	if selected, reason := launch.InventoryPython(inventory, ">=3.12", "3.13.9"); reason != "" || selected != "3.13.11" {
		t.Fatalf("a compatible patch of the captured minor was refused: %s %s", selected, reason)
	}
	if _, reason := launch.InventoryPython(inventory, ">=3.12", "3.15.1"); reason == "" {
		t.Fatal("an executor of another minor served the captured ABI")
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

// A published remote package is selectable without acquiring any local Python
// interpreter. The actual remote inventory decides whether its captured ABI fits.
func TestPublishedRentalPythonUsesRemoteInventory(t *testing.T) {
	detail := rentalReleaseFacts()
	detail.RequiresPython = ">=3.13,<3.14"
	detail.PythonVersion = "3.13.11"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/packages/proof/remote-python/releases/1" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(detail)
	}))
	defer server.Close()
	constraints, problem := cli.RentalConstraints(&cli.Context{Cfg: config.Config{Home: t.TempDir(), HubURL: server.URL}}, records.Request{Package: "proof/remote-python", Release: "1", Entrypoint: "job"})
	fatal(t, problem)
	if constraints.PythonVersion != "3.13.11" || constraints.RequiresPython != detail.RequiresPython {
		t.Fatalf("release interpreter facts changed: %+v", constraints)
	}
	for _, tc := range []struct {
		version, abi string
		want         bool
	}{{"3.13.11", "cp313", true}, {"3.12.12", "cp312", false}, {"3.13.12", "cp313", true}} {
		choices := rental.Purchases([]hub.RentalSKU{{Name: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, BaseWorkerProfile: "python3.12-cpu-linux-x86", PythonInterpreters: []*pb.PythonInterpreter{{Version: tc.version, Abi: tc.abi}}}}, nil, false, true, constraints)
		if len(choices) != 1 || (choices[0].Verdict == "") != tc.want {
			t.Fatalf("remote %s selected %v; want %v", tc.version, choices, tc.want)
		}
	}
}

func TestRentalPythonProvisioningKeepsCaptureAndInventorySeparate(t *testing.T) {
	for _, tc := range []struct {
		name, version, requires string
		minors                  []string
		want                    bool
	}{
		{"missing exact patch", "3.13.7", ">=3.13,<3.14", []string{"3.12", "3.13", "3.14"}, true},
		{"minor-only capture", "3.13", ">=3.13,<3.14", []string{"3.12", "3.13", "3.14"}, true},
		{"missing capability", "3.13.7", ">=3.13", nil, false},
		{"outside window", "3.15.1", ">=3.13", []string{"3.12", "3.13", "3.14"}, false},
		{"conflicting bound", "3.13.7", ">=3.14", []string{"3.13"}, false},
		{"range without capture", "", ">=3.13", []string{"3.13"}, false},
		{"prerelease is not provisionable", "3.13.7rc1", ">=3.13", []string{"3.13"}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			installed := []*pb.PythonInterpreter{{Version: "3.12.12", Abi: "cp312"}}
			sku := hub.RentalSKU{Name: "cpu", AcceleratorModel: "CPU", PythonInterpreters: installed, PythonProvisionableMinors: tc.minors}
			choices := rental.Purchases([]hub.RentalSKU{sku}, nil, false, true, rental.Constraints{RequiresPython: tc.requires, PythonVersion: tc.version})
			if len(choices) != 1 || (choices[0].Verdict == "") != tc.want {
				t.Fatalf("%+v", choices)
			}
			if len(sku.PythonInterpreters) != 1 || sku.PythonInterpreters[0].Version != "3.12.12" {
				t.Fatal("provisioning capability mutated installed inventory")
			}
		})
	}
	raw := json.RawMessage(`{"format":"tensorhub.image_inventory/1","profile":"python3.12-cpu-linux-x86","python":"3.12.12","interpreters":[{"version":"3.12.12","abi":"cp312"}],"provisionable_minors":["3.12","3.13","3.14"],"distributions":[]}`)
	inventory, err := rental.ImageInventory(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(inventory.Interpreters) != 1 {
		t.Fatal("capability became an installed executor")
	}
	if !launch.ProvisionablePython(rental.ImagePythonCapabilities(raw), ">=3.13", "3.13.7") {
		t.Fatal("rental reuse lost provisioning capability")
	}
}

func TestRentalPythonProvisioningDoesNotHideInvalidInstalledABI(t *testing.T) {
	sku := hub.RentalSKU{Name: "cpu", AcceleratorModel: "CPU", PythonInterpreters: []*pb.PythonInterpreter{{Version: "3.13.7", Abi: "cp313t"}}, PythonProvisionableMinors: []string{"3.13"}}
	choices := rental.Purchases([]hub.RentalSKU{sku}, nil, false, true, rental.Constraints{RequiresPython: ">=3.13", PythonVersion: "3.13.7"})
	if len(choices) != 1 || choices[0].Verdict == "" {
		t.Fatalf("invalid inventory admitted: %+v", choices)
	}
}
