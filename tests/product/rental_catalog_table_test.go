package producttest

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
)

var catalogGap = regexp.MustCompile(` {2,}`)

// catalogTable parses the ladder's human table into its header, its rows keyed by the
// NAME cell, and the order it printed them — so an assertion names a column instead of
// matching a substring that a longer row name also contains.
func catalogTable(t *testing.T, out string) ([]string, map[string][]string, []string) {
	t.Helper()
	var header []string
	rows := map[string][]string{}
	var order []string
	for _, line := range strings.Split(out, "\n") {
		if strings.TrimSpace(line) == "" ||
			strings.HasPrefix(line, "Next:") || strings.HasPrefix(line, "Note:") {
			continue
		}
		// Every cell but the GPU name is one word, so cells are split on the column gap.
		cells := catalogGap.Split(strings.TrimSpace(line), -1)
		if header == nil {
			header = cells
			continue
		}
		rows[cells[0]] = cells
		order = append(order, cells[0])
	}
	if header == nil {
		t.Fatalf("the ladder printed no table\n%s", out)
	}
	return header, rows, order
}

// TestRentalLadderReadsAsAGPUList is the ladder as the thing a person picks a machine
// from. The column names a GPU and says so; a card whose provider id carries workstation
// boilerplate reads as its product name while cards that genuinely differ still read
// differently; a CPU product states no compute capability rather than claiming one went
// missing; there is ONE price and it is the whole one; and the rungs climb cheapest
// first.
func TestRentalLadderReadsAsAGPUList(t *testing.T) {
	root := filepath.Join(scratchBase, "rental-ladder-reading")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	port := reservePort(t)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		fmt.Sprintf("tensorhub_url: http://127.0.0.1:%d\n", port)+
			"tensorhub_token: rental-idle-test\n"), 0o600))
	hub := newFakeRentalHub(t, port)
	gpuAt := func(name, model, capability string, vram, price int64, count int) map[string]any {
		return map[string]any{"name": name, "accelerator_model": model,
			"accelerator_count": count, "base_worker_profile": "torch2.13.0-cu130-cp312-linux-x86",
			"compute_capability": capability, "vram_gb": vram, "minimum_ram_per_gpu_gb": 64,
			"price_usd_micros_per_hour": price, "storage_usd_micros_per_hour": 41_700}
	}
	gpu := func(name, model, capability string, vram, price int64) map[string]any {
		return gpuAt(name, model, capability, vram, price, 1)
	}
	cpu := func(name string, price int64) map[string]any {
		return map[string]any{"name": name, "accelerator_model": "CPU",
			"accelerator_count": 1, "base_worker_profile": "cpu2.13.0-cp312-linux-x86",
			"compute_capability": "", "vram_gb": 0, "minimum_ram_per_gpu_gb": 0,
			"price_usd_micros_per_hour": price, "storage_usd_micros_per_hour": 2_780}
	}
	// Served in an order that is neither alphabetical nor priced, and carrying all THREE
	// real RTX PRO 6000 Blackwell cards — three products RunPod advertises under three
	// ids that differ only in their trailing words.
	hub.setSKUs(
		gpu("b200", "NVIDIA B200", "10.0", 180, 6_790_000),
		cpu("cpu-torch", 70_000),
		gpu("rtx-pro-6000-blackwell", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "12.0", 96, 1_690_000),
		gpu("rtx-a4000", "NVIDIA RTX A4000", "8.6", 16, 250_000),
		gpu("rtx-pro-6000", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "12.0", 96, 1_790_000),
		cpu("cpu", 70_000),
		gpu("rtx-pro-6000-maxq", "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", "12.0", 96, 1_590_000),
		gpuAt("h100-sxm5-80gb", "NVIDIA H100 80GB HBM3", "9.0", 80, 5_380_000, 2),
		gpu("h100-sxm5-80gb", "NVIDIA H100 80GB HBM3", "9.0", 80, 2_690_000),
	)

	code, out := runCozy(t, root, "rental", "new")
	if code != 0 {
		t.Fatalf("cozy rental new [exit %d]:\n%s", code, out)
	}
	header, rows, order := catalogTable(t, out)
	want := []string{"NAME", "ACCELERATOR", "GPUS", "COMPUTE", "VRAM", "PRICE"}
	if strings.Join(header, "|") != strings.Join(want, "|") {
		t.Fatalf("the ladder's columns are %v, not %v — these are GPUs, priced once\n%s",
			header, want, out)
	}

	// The card reads as its product name: the workstation boilerplate is not part of
	// what a person is choosing between.
	if got := rows["rtx-pro-6000-blackwell"][1]; got != "NVIDIA RTX Pro 6000 Blackwell" {
		t.Fatalf("the Blackwell card reads %q, not %q\n%s",
			got, "NVIDIA RTX Pro 6000 Blackwell", out)
	}
	// ...and the shortening never merges two real cards into one row. This is the guard:
	// a broader suffix rule renders all three of these identically, and a renter would
	// then be picking by a name that names three different machines.
	seen := map[string]string{}
	for _, name := range []string{"rtx-pro-6000", "rtx-pro-6000-blackwell", "rtx-pro-6000-maxq"} {
		cell := rows[name][1]
		if other, clash := seen[cell]; clash {
			t.Fatalf("%q and %q both read as %q — the shortening merged two distinct cards\n%s",
				other, name, cell, out)
		}
		seen[cell] = name
	}

	// A CPU product has no compute capability to state; "unknown" would say a lookup failed.
	for _, name := range []string{"cpu", "cpu-torch"} {
		if got := rows[name][3]; got != "-" {
			t.Fatalf("the %s row's COMPUTE reads %q, not the house dash\n%s", name, got, out)
		}
	}

	// ONE price, and it is the whole one: the components are not on this table.
	if got := rows["rtx-a4000"][5]; got != "$0.29/hr" {
		t.Fatalf("the A4000's price reads %q, not the combined $0.29/hr\n%s", got, out)
	}
	// A card is listed ONCE, with each GPU count priced per machine beside it.
	if got := rows["h100-sxm5-80gb"]; got[2] != "1, 2" || got[5] != "1x $2.73/hr, 2x $5.42/hr" {
		t.Fatalf("the H100 row reads %q, want GPUS 1, 2 priced per machine\n%s", got, out)
	}
	// $0.07/hr is NOT checked here: rounded to the penny, the cpu SKU's COMBINED
	// price ($0.07278) and its GPU component ($0.07) render identically, so the
	// string cannot distinguish an itemised component from the total this table is
	// supposed to show. The components that remain distinguishable still guard it.
	for _, component := range []string{"$0.25/hr", "$0.21/hr", "<$0.01/hr"} {
		if strings.Contains(out, component) {
			t.Fatalf("the ladder still itemises %s beside the combined price\n%s", component, out)
		}
	}

	// PRICES ARE READ IN PENNIES. Micro-dollar precision is how the provider quotes
	// and how we bill; `$0.291700/hr` asks a reader to parse six decimals to learn
	// "about forty-six cents". This asserts the rendered table, so it holds against
	// whichever formatter produced a cell -- there were two, rendering the same
	// micros independently, and rounding one left the other six decimals wide.
	if over := subPenny.FindString(out); over != "" {
		t.Fatalf("the ladder renders %q with sub-penny precision\n%s", over, out)
	}

	// Cheapest first, on the price the table shows — and equal rungs hold still.
	climb := []string{"cpu", "cpu-torch", "rtx-a4000", "rtx-pro-6000-maxq",
		"rtx-pro-6000-blackwell", "rtx-pro-6000", "h100-sxm5-80gb", "b200"}
	if strings.Join(order, "|") != strings.Join(climb, "|") {
		t.Fatalf("the ladder climbs %v, not cheapest-first %v\n%s", order, climb, out)
	}

	// --full expands the row limit without adding redundant human columns.
	code, full := runCozy(t, root, "rental", "new", "--full")
	if code != 0 {
		t.Fatalf("cozy rental new --full [exit %d]:\n%s", code, full)
	}
	if full != out {
		t.Fatalf("--full changed this complete catalog's columns or prices:\n%s", full)
	}

	// Structured output retains price components and the exact provider model ID.
	code, structured := runCozy(t, root, "rental", "new", "--full", "--json")
	if code != 0 {
		t.Fatalf("cozy rental new --full --json [exit %d]:\n%s", code, structured)
	}
	for _, kept := range []string{
		`"gpu":`, `"accelerator model":`, `"gpus":`, `"gpu price":`, `"storage price":`,
		"NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
		"$1.69/hr", "$0.04/hr", "$1.73/hr",
	} {
		if !strings.Contains(structured, kept) {
			t.Fatalf("structured output lost %q from the ladder\n%s", kept, structured)
		}
	}
	if strings.Contains(full, "STORAGE") || strings.Contains(full, "GPU PRICE") {
		t.Fatalf("expanded catalog itemizes storage instead of one total: %s", full)
	}
	hub.close()
}

// subPenny matches a price carrying more than two decimal places.
var subPenny = regexp.MustCompile(`\$[0-9]+\.[0-9]{3,}`)
