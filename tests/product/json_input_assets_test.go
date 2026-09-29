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

func TestJSONAssetFilesUseOrdinaryBindings(t *testing.T) {
	root := t.TempDir()
	imagePath := filepath.Join(root, "hero=portrait.png")
	f, err := os.Create(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(f, image.NewRGBA(image.Rect(0, 0, 2, 2))); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	ep := &launch.Entrypoint{Name: "long_form", Request: launch.Struct{Fields: []launch.Field{
		{Name: "overall", Type: json.RawMessage(`"str"`)},
		{Name: "references", Type: json.RawMessage(`{"list":{"fields":[{"name":"name","type":"str"},{"name":"description","type":"str"},{"name":"image","wire":"optional","type":{"union":["null",{"asset":"image"}]},"asset_bound":{"max_bytes":4096,"media_types":["image/png"]}}]}}`)},
	}}}
	request := `{"overall":"./ordinary prompt.png","references":[{"name":"hero","description":"./not-a-file.png","image":"hero=portrait.png"},{"name":"generated","description":"A guide"}]}`
	requestPath := filepath.Join(root, "request.json")
	if err := os.WriteFile(requestPath, []byte(request), 0600); err != nil {
		t.Fatal(err)
	}
	payload, _, problem := launch.ParsePayload(ep, nil, requestPath)
	if problem != nil {
		t.Fatal(problem)
	}
	if !strings.Contains(string(payload), imagePath) || !strings.Contains(string(payload), "./not-a-file.png") {
		t.Fatalf("JSON-relative asset resolution changed ordinary text: %s", payload)
	}
	resolved, assets, problem := launch.ParseAssets(ep, payload, nil, nil, nil)
	if problem != nil {
		t.Fatal(problem)
	}
	if len(assets) != 1 || assets[0].FieldPath != "references.0.image" || assets[0].LocalPath != imagePath || assets[0].MediaType != "image/png" || assets[0].MaxBytes != 4096 {
		t.Fatalf("wrong asset grant: %+v", assets)
	}
	if problem := payloadProblem("proof/h3", ep, resolved); problem != nil {
		t.Fatal(problem)
	}
	if strings.Contains(string(resolved), imagePath) || !strings.Contains(string(resolved), assets[0].Digest) {
		t.Fatalf("worker payload leaked filename or omitted asset identity: %s", resolved)
	}
	_, explicit, problem := launch.ParseAssets(ep, []byte(`{"overall":"x","references":[{"name":"hero","description":"A person"}]}`), []string{"references.0.image=" + imagePath}, nil, nil)
	if problem != nil || len(explicit) != 1 || explicit[0].Digest != assets[0].Digest {
		t.Fatalf("--asset differs: %+v %v", explicit, problem)
	}
	_, _, problem = launch.ParseAssets(ep, payload, []string{"references.0.image=" + imagePath}, nil, nil)
	if problem == nil || !strings.Contains(problem.Message, "more than once") {
		t.Fatalf("duplicate input silently accepted: %v", problem)
	}
	for name, source := range map[string]string{"missing": "absent.png", "url": "https://example.com/image.png", "wrong-media": "note.txt"} {
		t.Run(name, func(t *testing.T) {
			if name == "wrong-media" {
				if err := os.WriteFile(filepath.Join(root, source), []byte("not an image"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			bad := strings.Replace(request, "hero=portrait.png", source, 1)
			if err := os.WriteFile(requestPath, []byte(bad), 0600); err != nil {
				t.Fatal(err)
			}
			p, _, problem := launch.ParsePayload(ep, nil, requestPath)
			if problem == nil {
				_, _, problem = launch.ParseAssets(ep, p, nil, nil, nil)
			}
			if problem == nil {
				t.Fatal("invalid media accepted")
			}
		})
	}
}

func TestJSONAssetWalkRespectsUnionTagsAndReferences(t *testing.T) {
	ep := &launch.Entrypoint{Request: launch.Struct{Fields: []launch.Field{
		{Name: "item", Type: json.RawMessage(`{"tag_field":"kind","union":[{"tag":"image","fields":[{"name":"value","type":{"asset":"image"}}]},{"tag":"text","fields":[{"name":"value","type":"str"}]}]}`)},
		{Name: "ambiguous", Type: json.RawMessage(`{"union":["str",{"asset":"image"}]}`)},
		{Name: "known", Type: json.RawMessage(`{"asset":"image"}`)},
		{Name: "choice", Type: json.RawMessage(`{"union":[{"asset":"image"},{"literal":["automatic"]}]}`)},
	}}}
	payload := []byte(`{"item":{"kind":"text","value":"/not/a/file"},"ambiguous":"/also/not/a/file","known":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","choice":"automatic"}`)
	_, assets, problem := launch.ParseAssets(ep, payload, nil, nil, nil)
	if problem != nil || len(assets) != 0 {
		t.Fatalf("text/reference was interpreted as a filename: %v %+v", problem, assets)
	}
}
