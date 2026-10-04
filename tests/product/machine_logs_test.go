package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// This computer's machine and a rental answer the TensorFS transport log through the same
// machine call: `cozy machine logs --tensorfs` and `cozy rental logs <pod> --tensorfs` print
// what TensorFS wrote beside each store, rotated file first, and a machine whose TensorFS has
// logged nothing prints a note and succeeds. So does a machine whose agent predates the read
// (wire 72): both verbs say so and exit 0.
func TestMachinesPrintTheirTensorFSTransportLog(t *testing.T) {
	h, root, _, _ := parityMachines(t)
	const predates = "agent predates reading its logs"
	code, out := runCozy(t, root, "machine", "logs", "--tensorfs")
	if code == 0 && strings.Contains(out, predates) {
		if code, out := runCozy(t, root, "rental", "logs", "tessa", "--tensorfs"); code != 0 || !strings.Contains(out, predates) {
			t.Fatalf("the rental's log on an older agent [%d]: %s", code, out)
		}
		code, out := runCozy(t, root, "rental", "logs", parityRental, "--tensorfs", "--json")
		var log struct{ Log, Text, Unavailable string }
		if code != 0 || json.Unmarshal([]byte(out), &log) != nil || log.Log != "tensorfs" || log.Text != "" || !strings.Contains(log.Unavailable, predates) {
			t.Fatalf("the rental's log on an older agent as JSON [%d]: %s", code, out)
		}
		return
	}
	if code != 0 || !strings.Contains(out, "has logged no TensorFS transport decisions") {
		t.Fatalf("an unwritten log [%d]: %s", code, out)
	}
	write := func(store, older, current string) {
		dir := filepath.Join(store, "logs")
		must(t, os.MkdirAll(dir, 0o755))
		must(t, os.WriteFile(filepath.Join(dir, "transport.log.1"), []byte(older), 0o644))
		must(t, os.WriteFile(filepath.Join(dir, "transport.log"), []byte(current), 0o644))
	}
	const hedge = "1790879965.781 hedge 2797556697dd7aa8 67108864 ranged lanes=2 rate_bps=5252 left=67043328 measured=true busy=128 level=128\n"
	const grow = "1790879966.001 grow 2797556697dd7aa8 67108864 lanes=64 busy=3 level=128 hedges=2\n"
	const walk = "1790881096.936 walk 4477 101123198626 fetched=4477 cached=0 moved=101123198626 seconds=231.4 level=128 hedges=29 hedges_won=29 outcome=ok\n"
	write(machineStore(root), hedge, grow+walk)
	// A rental's machine keeps its own store; its launcher names none.
	write(filepath.Join(h.provider, "var", "lib", "cozy", "rust-machine", "tensorfs"), "", walk)

	if code, out := runCozy(t, root, "machine", "logs", "--tensorfs"); code != 0 || out != hedge+grow+walk {
		t.Fatalf("this computer's log [%d]: %q", code, out)
	}
	if code, out := runCozy(t, root, "rental", "logs", "tessa", "--tensorfs"); code != 0 || out != walk {
		t.Fatalf("the rental's log [%d]: %q", code, out)
	}
	code, out = runCozy(t, root, "rental", "logs", parityRental, "--tensorfs", "--json")
	var log struct {
		Log, Text, Unavailable string
	}
	if code != 0 || json.Unmarshal([]byte(out), &log) != nil || log.Log != "tensorfs" || log.Text != walk || log.Unavailable != "" {
		t.Fatalf("the rental's log as JSON [%d]: %s", code, out)
	}
	if code, out := runCozy(t, root, "rental", "logs", "tessa", "--tensorfs", "--follow"); code == 0 || !strings.Contains(out, "boot log") {
		t.Fatalf("--tensorfs with the boot log's --follow [%d]: %s", code, out)
	}
	if code, out := runCozy(t, root, "machine", "logs"); code == 0 || !strings.Contains(out, "--tensorfs") {
		t.Fatalf("a machine log named by nothing [%d]: %s", code, out)
	}
}
