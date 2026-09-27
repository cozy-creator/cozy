package producttest

import (
	"encoding/json"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
)

// retainedAssetRef is an existing content reference; a bare JSON string is a local filename.
const retainedAssetRef = "sha256:0000000000000000000000000000000000000000000000000000000000000000"

const declaredAssetsInterface = `{"application":"assets:app","format":"cozy.package.interface/1","jobs":[],"entrypoints":[{"name":"run","assets":{"parameter":"assets","kinds":[{"kind":"image","max_bytes":1024,"max_decoded_bytes":4096,"media_types":["image/png"]}]},"request":{"fields":[{"name":"prompt","type":"str"},{"name":"assets","constraints":{"min_length":1,"max_length":3},"type":{"list":{"fields":[{"name":"asset","type":{"asset":"file"}},{"name":"label","type":"str","wire":"optional"},{"name":"fidelity","type":{"literal":["auto","high","low","medium"]},"wire":"optional"}]}}},{"name":"poster","type":{"asset":"image"},"asset_bound":{"max_bytes":1024,"media_types":["image/png"]},"wire":"optional"}]},"result":{"fields":[]}}]}`

func assetsCallable(t *testing.T) *launch.Entrypoint {
	t.Helper()
	raw := []byte(declaredAssetsInterface)
	iface, problem := launch.DecodePackageInterface(raw)
	fatal(t, problem)
	ep, problem := iface.Function("run")
	fatal(t, problem)
	return ep
}

func TestDeclaredAssetsPreserveOccurrencesLabelsAndNamedFields(t *testing.T) {
	ep := assetsCallable(t)
	path := filepath.Join(t.TempDir(), "a=b.png")
	f, err := os.Create(path)
	must(t, err)
	must(t, png.Encode(f, image.NewRGBA(image.Rect(0, 0, 2, 2))))
	must(t, f.Close())
	home, err := os.UserHomeDir()
	must(t, err)
	relative, err := filepath.Rel(home, path)
	must(t, err)
	payload, assets, problem := launch.ParseAssets(ep, []byte(`{"prompt":"<Picture1> and <Picture2>"}`), []string{" 艾丽丝 =~/" + relative, path, "POSTER=" + path}, nil, nil)
	fatal(t, problem)
	fatal(t, launch.ValidatePayload("proof/assets", ep, payload))
	var doc struct {
		Prompt string `json:"prompt"`
		Assets []struct{ Asset, Label string }
		Poster string
	}
	must(t, json.Unmarshal(payload, &doc))
	if len(doc.Assets) != 2 || doc.Assets[0].Label != " 艾丽丝 " || doc.Assets[1].Label != "" || doc.Assets[0].Asset != doc.Assets[1].Asset || doc.Prompt != "<Picture1> and <Picture2>" {
		t.Fatalf("occurrences/labels/prompt changed: %s", payload)
	}
	if len(assets) != 3 || assets[0].FieldPath != "assets.0.asset" || assets[1].FieldPath != "assets.1.asset" || assets[2].FieldPath != "poster" || assets[1].Order != 1 || assets[0].Digest != assets[1].Digest || assets[0].MediaType != "image/png" || assets[0].MaxBytes != 1024 {
		t.Fatalf("ordinary input bindings changed: %+v", assets)
	}
	// A callee can repeat/subset its parent's authorized contents with new
	// occurrence labels. A cached digest outside the parent grant is no authority.
	child := json.RawMessage(`{"prompt":"child","assets":[{"asset":"` + assets[0].Digest + `","label":"second"},{"asset":"` + assets[0].Digest + `","label":"first"}]}`)
	forwarded, problem := launch.InheritChildAssets(ep, child, assets)
	fatal(t, problem)
	if len(forwarded) != 2 || forwarded[0].LocalPath != assets[0].LocalPath || forwarded[1].Order != 1 || forwarded[1].FieldPath != "assets.1.asset" {
		t.Fatalf("child occurrences changed: %+v", forwarded)
	}
	if _, problem = launch.InheritChildAssets(ep, child, nil); problem == nil || problem.ErrName() != "child.asset_ungranted" {
		t.Fatalf("child inherited an ungranted asset: %v", problem)
	}
	for _, specs := range [][]string{{"same=" + path, "same=" + path}, {path, path, path, path}} {
		if _, _, problem := launch.ParseAssets(ep, []byte(`{"prompt":"test"}`), specs, nil, nil); problem == nil {
			t.Fatalf("invalid occurrence list accepted: %v", specs)
		}
	}
	if _, _, problem := launch.ParseAssets(ep, []byte(`{"prompt":"test","assets":[{"asset":"existing","label":"same"},{"asset":"existing","label":"same"}]}`), []string{"/absent-file"}, nil, nil); problem == nil || !strings.Contains(problem.Message, "more than once") {
		t.Fatalf("duplicate payload labels did not refuse before IO: %v", problem)
	}
	// Bounds and duplicate labels are checked before trying to read missing files.
	if _, _, problem := launch.ParseAssets(ep, []byte(`{"prompt":"test"}`), []string{"same=/absent-a", "same=/absent-b"}, nil, nil); problem == nil || !strings.Contains(problem.Message, "more than once") {
		t.Fatalf("duplicate label did not refuse before IO: %v", problem)
	}
	plain := filepath.Join(t.TempDir(), "not-an-image.png")
	must(t, os.WriteFile(plain, []byte("plain text"), 0600))
	if _, _, problem := launch.ParseAssets(ep, []byte(`{"prompt":"test"}`), []string{plain}, nil, nil); problem == nil {
		t.Fatal("file extension overruled observed MIME")
	}
	bytes, err := os.ReadFile(path)
	must(t, err)
	large := filepath.Join(t.TempDir(), "large.png")
	must(t, os.WriteFile(large, append(bytes, make([]byte, 2048)...), 0600))
	if _, _, problem := launch.ParseAssets(ep, []byte(`{"prompt":"test"}`), []string{large}, nil, nil); problem == nil {
		t.Fatal("encoded per-item limit was ignored")
	}
	undeclared := *ep
	undeclared.Assets = nil
	if _, _, problem := launch.ParseAssets(&undeclared, []byte(`{"prompt":"test"}`), []string{path}, nil, nil); problem == nil {
		t.Fatal("bare asset guessed a payload destination without a declared Assets slot")
	}
}

func TestDeclaredAssetsDefaultEmptyCollection(t *testing.T) {
	ep := assetsCallable(t)
	payload, bindings, problem := launch.ParseAssets(ep, []byte(`{"prompt":"text only"}`), nil, nil, nil)
	fatal(t, problem)
	if len(bindings) != 0 || !strings.Contains(string(payload), `"assets":[]`) {
		t.Fatalf("missing declared Assets did not become an empty collection: %s %+v", payload, bindings)
	}
	if problem := launch.ValidatePayload("proof/assets", ep, payload); problem == nil {
		t.Fatal("empty required image collection ignored the authored minimum count")
	}
	// A zero-reference callable uses the same payload builder and schema owner.
	ep.Request.Fields[1].Constraints.MinLength = nil
	fatal(t, launch.ValidatePayload("proof/assets", ep, payload))
	explicit := []byte(`{"prompt":"text only","assets":[{"asset":"` + retainedAssetRef + `","label":"last"}]}`)
	payload, bindings, problem = launch.ParseAssets(ep, explicit, nil, nil, nil)
	fatal(t, problem)
	if len(bindings) != 0 || !strings.Contains(string(payload), `"asset":"`+retainedAssetRef+`","label":"last"`) {
		t.Fatalf("explicit collection was replaced: %s %+v", payload, bindings)
	}
	if _, _, problem = launch.ParseAssets(ep, []byte(`{"prompt":"text only","assets":null}`), nil, nil, nil); problem == nil {
		t.Fatal("explicit null collection became an empty list")
	}
}

func TestDeclaredAssetsDecodedViewDescriptor(t *testing.T) {
	for _, value := range []string{`"decoded"`, `"raw"`, `""`, `null`} {
		t.Run(value, func(t *testing.T) {
			raw := strings.Replace(declaredAssetsInterface, `"parameter":"assets"`, `"parameter":"assets","view":`+value, 1)
			iface, problem := launch.DecodePackageInterface([]byte(raw))
			fatal(t, problem)
			ep, problem := iface.Function("run")
			fatal(t, problem)
			if ep.Assets == nil || (value == `"decoded"`) != (ep.Assets.View == "decoded") {
				t.Fatalf("Assets view %s was not carried as authored: %+v", value, ep.Assets)
			}
		})
	}
	fileKind := strings.Replace(declaredAssetsInterface, `"kind":"image"`, `"kind":"file"`, 1)
	if _, problem := launch.DecodePackageInterface([]byte(fileKind)); problem != nil {
		t.Fatalf("ordinary raw file Assets refused: %v", problem)
	}
	// Decoding is Runtime-owned: a view/kind pairing this host does not act on still loads.
	decodedFile := strings.Replace(fileKind, `"parameter":"assets"`, `"parameter":"assets","view":"decoded"`, 1)
	if _, problem := launch.DecodePackageInterface([]byte(decodedFile)); problem != nil {
		t.Fatalf("a decoded file Assets slot refused the interface: %v", problem)
	}
	if _, problem := launch.DecodePackageInterface([]byte(strings.Replace(declaredAssetsInterface,
		`"parameter":"assets"`, `"parameter":"assets","view":true`, 1))); problem == nil {
		t.Fatal("a mistyped Assets view was admitted")
	}
}

func TestDeclaredAssetsCountsAndFidelity(t *testing.T) {
	ep := assetsCallable(t)
	limit := int64(1)
	ep.Assets.Kinds[0].MaxCount = &limit
	photo := filepath.Join(t.TempDir(), "image.png")
	file, err := os.Create(photo)
	must(t, err)
	must(t, png.Encode(file, image.NewRGBA(image.Rect(0, 0, 2, 2))))
	must(t, file.Close())
	payload, bindings, problem := launch.ParseAssets(ep, []byte(`{"prompt":"same"}`), []string{"alice=" + photo, "poster=" + photo}, []string{"alice=high"}, nil)
	fatal(t, problem)
	fatal(t, launch.ValidatePayload("proof/assets", ep, payload))
	if len(bindings) != 2 || !strings.Contains(string(payload), `"fidelity":"high"`) {
		t.Fatalf("named field consumed a collection count or fidelity was lost: %s", payload)
	}
	child := []byte(`{"prompt":"child","assets":[{"asset":"` + bindings[0].Digest + `"},{"asset":"` + bindings[0].Digest + `"}]}`)
	if _, problem := launch.InheritChildAssets(ep, child, bindings); problem == nil || problem.ErrName() != "input_asset_count" {
		t.Fatalf("inherited child bypassed the same kind count: %v", problem)
	}
	for _, count := range []int64{1, 0} {
		limit = count
		if _, _, problem := launch.ParseAssets(ep, []byte(`{"prompt":"same"}`), []string{photo, photo}, nil, nil); problem == nil || problem.ErrName() != "input_asset_count" {
			t.Fatalf("observed image count exceeded cap %d: %v", count, problem)
		}
	}
	// Unknown hints and duplicate aliases refuse before opening either file.
	for _, mappings := range [][]string{{"alice=low", "0=high"}, {"missing=high"}, {"0=extreme"}, {"01=low"}, {"-1=low"}} {
		if _, _, problem := launch.ParseAssets(ep, []byte(`{"prompt":"same"}`), []string{"alice=/absent-one", "/absent-two"}, mappings, nil); problem == nil || !strings.Contains(problem.Message, "fidelity") {
			t.Fatalf("invalid fidelity mapping reached file IO: %v %v", mappings, problem)
		}
	}
	// Exact labels win, including numeric labels; the final '=' belongs to the hint.
	payload, _, problem = launch.ParseAssets(ep, []byte(`{"prompt":"same","assets":[{"asset":"`+retainedAssetRef+`","label":"名=字"},{"asset":"`+retainedAssetRef+`","label":"0"}]}`), nil, []string{"名=字=medium", "0=low"}, nil)
	fatal(t, problem)
	var document struct {
		Assets []struct{ Label, Fidelity string }
	}
	must(t, json.Unmarshal(payload, &document))
	if document.Assets[0].Fidelity != "medium" || document.Assets[1].Fidelity != "low" {
		t.Fatalf("fidelity selector changed label meaning: %s", payload)
	}
	// A readable cap is applied; an absent or negative one is no cap; a mistyped one refuses.
	for value, want := range map[string]int64{"0": 0, "2": 2, "-1": -1, "null": -1} {
		raw := strings.Replace(declaredAssetsInterface, `"kind":"image"`, `"kind":"image","max_count":`+value, 1)
		iface, problem := launch.DecodePackageInterface([]byte(raw))
		fatal(t, problem)
		ep, problem := iface.Function("run")
		fatal(t, problem)
		if cap := ep.Assets.Kinds[0].MaxCount; (cap == nil) != (want < 0) || cap != nil && *cap != want {
			t.Fatalf("max_count %s read as %v", value, cap)
		}
	}
	for _, value := range []string{"true", "1.5"} {
		raw := strings.Replace(declaredAssetsInterface, `"kind":"image"`, `"kind":"image","max_count":`+value, 1)
		if _, problem := launch.DecodePackageInterface([]byte(raw)); problem == nil {
			t.Fatalf("mistyped max_count %s was admitted", value)
		}
	}
}
