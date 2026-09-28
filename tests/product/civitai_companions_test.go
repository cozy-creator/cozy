package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/tfs"
)

// civitaiVersionFiles is one Civitai version's declared SafeTensors files: name to exact bytes.
func civitaiVersionFiles(t *testing.T, version int) map[string]int64 {
	t.Helper()
	probe := &http.Client{Timeout: 20 * time.Second}
	response, err := probe.Get(fmt.Sprintf("https://civitai.com/api/v1/model-versions/%d", version))
	if err != nil {
		t.Skipf("provider unreachable from this runner: %v", err)
	}
	defer response.Body.Close()
	var document struct {
		Files []struct {
			Name    string  `json:"name"`
			Primary bool    `json:"primary"`
			SizeKB  float64 `json:"sizeKB"`
		} `json:"files"`
	}
	must(t, json.NewDecoder(response.Body).Decode(&document))
	files := map[string]int64{}
	for _, file := range document.Files {
		name := file.Name
		if file.Primary {
			name = "primary"
		}
		files[name] = int64(file.SizeKB * 1024)
	}
	return files
}

// releasedTfs is TensorFS 0.3.77's tfs, the first that composes a Civitai version from its
// companion files, as the host tool.
func releasedTfs(t *testing.T) string {
	t.Helper()
	out, err := exec.Command("uv", "run", "--python", "3.12", "--no-project", "--with", "tensorfs==0.3.77", "--",
		"python", "-c", "import shutil; print(shutil.which('tfs'))").Output()
	if err != nil {
		t.Skipf("TensorFS 0.3.77 is not installable here: %v", err)
	}
	return strings.TrimSpace(string(out))
}

// A Civitai version's companion SafeTensors files are planned as TensorFS's select_members
// selects them (cl-632): Anima (2945208) is its DiT primary with the text encoder and VAE
// beside it, one reviewed model, sized whole and planned with no as-is note. RealVisXL
// (789646) is an SDXL primary with an fp32 twin, which stays the primary alone.
func TestCivitaiCompanionsArePlannedAsTheRentalSelectsThem(t *testing.T) {
	anima, realvis := civitaiVersionFiles(t, 2945208), civitaiVersionFiles(t, 789646)
	host := []string{"COZY_TFS=" + releasedTfs(t)}
	for _, test := range []struct {
		version int
		want    int64
	}{
		{2945208, anima["primary"] + anima["anima_baseV10_txt.safetensors"] + anima["qwen_image_vae_2004692.safetensors"]},
		{789646, realvis["primary"]},
	} {
		t.Run(fmt.Sprint(test.version), func(t *testing.T) {
			root, paid := rentalCapture(t)
			out, code := rentedUpload(t, root, host, fmt.Sprintf("civitai://%d", test.version), "proof/companions")
			if strings.Contains(string(out), tfs.AsIsNote) {
				t.Fatalf("the host planned the version as-is:\n%s", out)
			}
			if request := paid(t, out, code); request.PlannedSourceBytes != test.want {
				t.Fatalf("the ingest declared %d planned source bytes, want %d", request.PlannedSourceBytes, test.want)
			}
		})
	}
}

