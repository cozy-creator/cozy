package producttest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/tfs"
	"github.com/cozy-creator/cozy/internal/units"
)

// The plain/1 encoding, as `tfs cbor decode` prints it: the one seeded spec a synthetic
// header can name without a registry of its own.
const plainEncoding = `{"doc":"Plain little-endian row-major storage: one ` + "`value`" + ` role carrying the logical bytes verbatim. Entry zero of the registry; not a special absent form.","logical_dtypes":["bf16","bool","f16","f32","f64","f8_e4m3fn","f8_e5m2","i16","i32","i64","i8","u8"],"roles":{"value":{"carrier":{"t":"same_as_logical"},"shape":{"t":"same"}}},"vectors":{"length":516,"sha256":"4766f931bbb70ed431e34b34e4f9fc5dda8174e96f43a8acd79d8799efcd43d2"}}`

// TestModelListShowsEachModelsBytesAndWhatItShares retains two local models through the
// product's own TensorFS path — blobs put, a CozyTensors header and manifest reproduced
// over them, the alias replaced with exact evidence — that share one config blob and one
// tensor segment. `cozy model list` then shows SIZE and SHARED to the byte, JSON carries
// integers plus `unique`, and the store footer appears only once a blob is unreferenced.
// The expected numbers come from `tfs manifest walk`, a second TensorFS surface.
func TestModelListShowsEachModelsBytesAndWhatItShares(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "model-usage")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	must(t, os.Setenv("COZY_HOME", root))
	cfg, e := config.Load()
	fatal(t, e)
	layout, e := home.Open(cfg.Home)
	fatal(t, e)
	tool, e := tfs.Open(cfg, layout)
	if e != nil && e.Name == "tfs_missing" {
		t.Skipf("no tensorfs CLI on this runner: %s", e.Message)
	}
	fatal(t, e)
	store := tfsStore{t: t, bin: tool.Bin, env: cfg.Tool(), root: tool.Root, work: t.TempDir()}

	configBlob := store.put("config.json", []byte(`{"kind":"shared-config"}`))
	shared := store.put("shared.bin", bytes.Repeat([]byte{7}, 300))
	model := func(name string, own byte) map[string]int64 {
		ownBlob := store.put(name+"-own.bin", bytes.Repeat([]byte{own}, 300))
		header := fmt.Sprintf(`{"format":"cozytensors/1","configs":[],"assets":[],"encodings":[%s],`+
			`"components":[["model",[`+
			`["own","f32",[75],0,[["value","f32",[75],{"segments":[["sha256:%s",300]]}]]],`+
			`["shared","f32",[75],0,[["value","f32",[75],{"segments":[["sha256:%s",300]]}]]]]]]}`,
			plainEncoding, ownBlob.sha256, shared.sha256)
		entries := fmt.Sprintf(`[["config.json","file",{"length":%d,"sha256":"%s"}],["model.cozytensors","cozytensors"]]`,
			configBlob.length, configBlob.sha256)
		manifest := store.reproduce(name, header, entries, `[["model","own"],["model","shared"]]`)
		evidence := filepath.Join(store.work, name+"-evidence.json")
		must(t, os.WriteFile(evidence, []byte(`{"classification_digest":"`+store.topology(manifest.header)+`"}`), 0o600))
		if _, e := tool.ReplaceLocal(name, "sha256:"+shared.sha256, "absent", "sha256:"+manifest.sha256,
			manifest.length, evidence, nil); e != nil {
			t.Fatalf("local/%s: %s", name, briefly(e))
		}
		return store.walk(manifest.sha256)
	}
	alpha, beta := model("alpha", 1), model("beta", 2)
	sum := func(refs map[string]int64) (total int64) {
		for _, length := range refs {
			total += length
		}
		return total
	}
	var sharedBytes int64
	for sha256, length := range alpha {
		if _, ok := beta[sha256]; ok {
			sharedBytes += length
		}
	}
	if sharedBytes != configBlob.length+shared.length {
		t.Fatalf("the fixture shares %d bytes, want the config and one segment (%d)", sharedBytes, configBlob.length+shared.length)
	}
	union := sum(alpha) + sum(beta) - sharedBytes

	code, out := runCozy(t, root, "model", "list")
	if code != 0 {
		t.Fatalf("model list [exit %d]\n%s", code, out)
	}
	for name, refs := range map[string]map[string]int64{"alpha": alpha, "beta": beta} {
		want := []string{"local/" + name, "-", "-", units.Bytes(sum(refs)), units.Bytes(sharedBytes)}
		if got := strings.Fields(tableRow(out, "local/"+name)); strings.Join(got, " ") != strings.Join(want, " ") {
			t.Errorf("model list row for local/%s is %v, want %v\n%s", name, got, want, out)
		}
	}
	if !strings.Contains(out, "SIZE") || !strings.Contains(out, "SHARED") || strings.Contains(out, "store:") {
		t.Errorf("model list columns or footer are wrong before anything is unreferenced\n%s", out)
	}

	usage := modelListJSON(t, root)
	for _, row := range usage.Models {
		refs := alpha
		if row.Model == "local/beta" {
			refs = beta
		}
		if row.Size != sum(refs) || row.Shared != sharedBytes || row.Unique != sum(refs)-sharedBytes {
			t.Errorf("%s json size/shared/unique = %d/%d/%d, want %d/%d/%d", row.Model,
				row.Size, row.Shared, row.Unique, sum(refs), sharedBytes, sum(refs)-sharedBytes)
		}
	}
	if len(usage.Models) != 2 || usage.Store.Models != 2 || usage.Store.Size != union ||
		usage.Store.Unique != union-sharedBytes || usage.Store.Unreferenced != 0 {
		t.Errorf("store json = %+v, want size %d unique %d unreferenced 0 in 2 models", usage.Store, union, union-sharedBytes)
	}

	// One verified blob nothing reaches: the footer says what deleting nothing would free.
	orphan := store.put("orphan.bin", bytes.Repeat([]byte{9}, 1000))
	code, out = runCozy(t, root, "model", "list")
	footer := fmt.Sprintf("store: %s in 2 models · %s unreferenced", units.Bytes(union), units.Bytes(orphan.length))
	if code != 0 || !strings.Contains(out, "\n"+footer+"\n") {
		t.Errorf("model list footer missing [exit %d]: want %q\n%s", code, footer, out)
	}
	if usage = modelListJSON(t, root); usage.Store.Unreferenced != orphan.length || usage.Store.Size != union {
		t.Errorf("store json after an orphan = %+v, want unreferenced %d", usage.Store, orphan.length)
	}
}

type modelUsageDocument struct {
	Models []struct {
		Model  string `json:"model"`
		Size   int64  `json:"size"`
		Shared int64  `json:"shared"`
		Unique int64  `json:"unique"`
	} `json:"models"`
	Store struct {
		Size         int64 `json:"size"`
		Unique       int64 `json:"unique"`
		Unreferenced int64 `json:"unreferenced"`
		Models       int   `json:"models"`
	} `json:"store"`
}

func modelListJSON(t *testing.T, root string) modelUsageDocument {
	t.Helper()
	code, out := runCozy(t, root, "model", "list", "--json")
	if code != 0 {
		t.Fatalf("model list --json [exit %d]\n%s", code, out)
	}
	var document modelUsageDocument
	if err := json.Unmarshal([]byte(out), &document); err != nil {
		t.Fatalf("model list --json is not the expected document: %v\n%s", err, out)
	}
	return document
}

func tableRow(out, prefix string) string {
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, prefix+" ") {
			return line
		}
	}
	return ""
}

// tfsStore drives the real tensorfs binary the product resolved, for the fixture steps the
// product has no verb of its own for.
type tfsStore struct {
	t    *testing.T
	bin  string
	env  []string
	root string
	work string
}

type blobRef struct {
	sha256 string
	length int64
}

type reproduced struct {
	blobRef
	header string
}

var (
	admittedLine   = regexp.MustCompile(`admitted sha256:([0-9a-f]{64}) length=(\d+)`)
	headerLine     = regexp.MustCompile(`header\s+sha256:([0-9a-f]{64})`)
	manifestLine   = regexp.MustCompile(`manifest\s+sha256:([0-9a-f]{64}) length=(\d+)`)
	topologyDigest = regexp.MustCompile(`topology_digest (sha256:[0-9a-f]{64})`)
)

func (s tfsStore) run(args ...string) string {
	s.t.Helper()
	cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", s.bin}, args...)...)
	cmd.Env = s.env
	out, err := cmd.CombinedOutput()
	if err != nil {
		s.t.Fatalf("tfs %s: %v\n%s", strings.Join(args, " "), err, out)
	}
	return string(out)
}

func (s tfsStore) put(name string, content []byte) blobRef {
	s.t.Helper()
	path := filepath.Join(s.work, name)
	must(s.t, os.WriteFile(path, content, 0o600))
	m := admittedLine.FindStringSubmatch(s.run("put", s.root, path))
	if m == nil {
		s.t.Fatalf("tfs put %s printed no admission", name)
	}
	length, _ := strconv.ParseInt(m[2], 10, 64)
	return blobRef{sha256: m[1], length: length}
}

func (s tfsStore) reproduce(name, header, entries, order string) reproduced {
	s.t.Helper()
	paths := map[string]string{}
	for suffix, content := range map[string]string{"header": header, "entries": entries, "order": order} {
		paths[suffix] = filepath.Join(s.work, name+"-"+suffix+".json")
		must(s.t, os.WriteFile(paths[suffix], []byte(content), 0o600))
	}
	out := s.run("checkpoint", "reproduce", s.root, paths["header"], paths["entries"], "--order", paths["order"])
	h, m := headerLine.FindStringSubmatch(out), manifestLine.FindStringSubmatch(out)
	if h == nil || m == nil {
		s.t.Fatalf("tfs checkpoint reproduce printed no header and manifest\n%s", out)
	}
	length, _ := strconv.ParseInt(m[2], 10, 64)
	return reproduced{blobRef: blobRef{sha256: m[1], length: length}, header: h[1]}
}

func (s tfsStore) topology(header string) string {
	s.t.Helper()
	m := topologyDigest.FindStringSubmatch(s.run("checkpoint", "info", s.root, header))
	if m == nil {
		s.t.Fatalf("tfs checkpoint info printed no topology digest")
	}
	return m[1]
}

// walk is the manifest's distinct blob closure by TensorFS's other surface.
func (s tfsStore) walk(manifest string) map[string]int64 {
	s.t.Helper()
	refs := filepath.Join(s.work, manifest[:16]+"-refs.jsonl")
	s.run("manifest", "walk", s.root, manifest, "--refs", refs)
	raw, err := os.ReadFile(refs)
	must(s.t, err)
	closure := map[string]int64{}
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var row struct {
			SHA256 string `json:"sha256"`
			Length int64  `json:"length"`
		}
		must(s.t, json.Unmarshal([]byte(line), &row))
		closure[row.SHA256] = row.Length
	}
	return closure
}
