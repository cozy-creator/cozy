package producttest

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/tfs"
)

var liveModelImportE2E = flag.String("live-model-import-e2e", "", "real local model-import fixture JSON")
var liveModelImportHF = flag.String("live-model-import-hf", "", "real public HF dry-run fixture JSON")

func TestModelImportGrammarAndFailFastRefusals(t *testing.T) {
	root := t.TempDir()
	env := []string{"COZY_HOME=" + root, "PATH=/usr/local/bin:/usr/bin:/bin"}
	if result := runCozyEnv(env, "model", "import", "--help"); result.code != 0 ||
		!strings.Contains(result.output, "<source>") || !strings.Contains(result.output, "--name") ||
		!strings.Contains(result.output, "--dry-run") || !strings.Contains(result.output, "--token-stdin") ||
		strings.Contains(result.output, "--hf") || strings.Contains(result.output, "--civitai") ||
		strings.Contains(result.output, "--url") || strings.Contains(result.output, "--rental") ||
		strings.Contains(result.output, "--release") || strings.Contains(result.output, "--lane") {
		t.Fatalf("model import help = exit %d\n%s", result.code, result.output)
	}
	if result := runCozyEnv(env, "model", "import", "hf://org/model@"+strings.Repeat("a", 40),
		"--name", "BadName"); result.code != 2 || !strings.Contains(result.output, "portable name") {
		t.Fatalf("invalid local name = exit %d\n%s", result.code, result.output)
	}
	file := filepath.Join(root, "local.safetensors")
	must(t, os.WriteFile(file, []byte("not-a-real-carrier"), 0o600))
	if result := runCozyEnv(env, "model", "import", file, "--name", "local-test", "--token-stdin"); result.code != 2 || !strings.Contains(result.output, "applies only") {
		t.Fatalf("local token stdin = exit %d\n%s", result.code, result.output)
	}
	if result := runCozyEnv(env, "model", "import",
		"https://civitai.com/api/download/models/1?token=leak", "--name", "leak"); result.code != 2 || !strings.Contains(result.output, "credentials must not appear") {
		t.Fatalf("URL credential = exit %d\n%s", result.code, result.output)
	}
	badConfig := t.TempDir()
	must(t, os.WriteFile(filepath.Join(badConfig, "config.yaml"), []byte(
		"tensorfs_registry: /tmp/operator-only.json\n"), 0o600))
	if result := runCozyEnv([]string{"COZY_HOME=" + badConfig, "PATH=/usr/local/bin:/usr/bin:/bin"}, "rental"); result.code != 2 || !strings.Contains(result.output, `unknown key "tensorfs_registry"`) {
		t.Fatalf("operator registry entered YAML = exit %d\n%s", result.code, result.output)
	}
}

func TestLocalModelStagingRefusesMutationAndSymlink(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "moving.safetensors")
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	must(t, err)
	must(t, file.Truncate(64<<20))
	must(t, file.Close())
	source, problem := modelsource.Parse(path, root)
	fatal(t, problem)
	started := make(chan struct{})
	done := make(chan struct{})
	stopped := make(chan struct{})
	var once sync.Once
	go func() {
		defer close(stopped)
		f, err := os.OpenFile(path, os.O_WRONLY, 0)
		if err != nil {
			close(started)
			return
		}
		defer f.Close()
		value := byte(1)
		for {
			_, _ = f.WriteAt([]byte{value}, 64<<20-1)
			value++
			once.Do(func() { close(started) })
			select {
			case <-done:
				return
			default:
				time.Sleep(time.Millisecond)
			}
		}
	}()
	<-started
	_, _, problem = modelsource.StageLocal(context.Background(), source, filepath.Join(root, "stage"))
	close(done)
	<-stopped
	if problem == nil || problem.Name != "model_source_changed" {
		t.Fatalf("mutating local source = %v", problem)
	}

	target := filepath.Join(root, "target.safetensors")
	must(t, os.WriteFile(target, []byte("carrier"), 0o600))
	link := filepath.Join(root, "link.safetensors")
	must(t, os.Symlink(target, link))
	if _, problem := modelsource.Parse(link, root); problem == nil || problem.Name != "model_source_file_refused" {
		t.Fatalf("symlink local source = %v", problem)
	}
}

func TestLocalModelStagingHonorsCancellation(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "source.safetensors")
	must(t, os.WriteFile(path, []byte("carrier"), 0o600))
	source, problem := modelsource.Parse(path, root)
	fatal(t, problem)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _, problem = modelsource.StageLocal(ctx, source, filepath.Join(root, "stage"))
	if problem == nil || problem.Code != exit.Canceled {
		t.Fatalf("canceled local staging = %v", problem)
	}
}

func TestLocalModelDryRunReadsOnlyTheHeader(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "large.safetensors")
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	must(t, err)
	header := []byte(`{"value":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}} `)
	prefix := make([]byte, 8)
	binary.LittleEndian.PutUint64(prefix, uint64(len(header)))
	_, err = file.Write(append(prefix, header...))
	must(t, err)
	const sourceSize = int64(8 << 30)
	must(t, file.Truncate(sourceSize))
	_, err = file.WriteAt([]byte{0x7f}, sourceSize-1)
	must(t, err)
	must(t, file.Close())

	source, problem := modelsource.Parse(path, root)
	fatal(t, problem)
	plan, staged, problem := modelsource.StageLocalHeader(context.Background(), source, filepath.Join(root, "stage"))
	fatal(t, problem)
	if plan.InspectedBytes != int64(8+len(header)) || plan.Bytes != sourceSize {
		t.Fatalf("dry-run inspection = %d bytes of %d", plan.InspectedBytes, plan.Bytes)
	}
	stagedFile, err := os.Open(staged.Path)
	must(t, err)
	defer stagedFile.Close()
	tail := []byte{0xff}
	_, err = stagedFile.ReadAt(tail, sourceSize-1)
	must(t, err)
	if tail[0] != 0 {
		t.Fatal("dry-run copied the model body into its sparse header view")
	}
}

func TestLocalModelDryRunHonorsCancellationAndMutation(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "moving-header.safetensors")
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	must(t, err)
	prefix := make([]byte, 8)
	binary.LittleEndian.PutUint64(prefix, uint64(maxTestHeader))
	_, err = file.Write(prefix)
	must(t, err)
	must(t, file.Truncate(8+maxTestHeader+4))
	must(t, file.Close())
	source, problem := modelsource.Parse(path, root)
	fatal(t, problem)

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, problem = modelsource.StageLocalHeader(canceled, source, filepath.Join(root, "canceled")); problem == nil || problem.Code != exit.Canceled {
		t.Fatalf("canceled dry-run staging = %v", problem)
	}

	started := make(chan struct{})
	done := make(chan struct{})
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		moving, openErr := os.OpenFile(path, os.O_WRONLY, 0)
		if openErr != nil {
			close(started)
			return
		}
		defer moving.Close()
		close(started)
		value := byte(1)
		for {
			_, _ = moving.WriteAt([]byte{value}, 8+maxTestHeader-1)
			value++
			select {
			case <-done:
				return
			default:
			}
		}
	}()
	<-started
	_, _, problem = modelsource.StageLocalHeader(context.Background(), source, filepath.Join(root, "moving"))
	close(done)
	<-stopped
	if problem == nil || problem.Name != "model_source_changed" {
		t.Fatalf("mutating dry-run source = %v", problem)
	}
}

const maxTestHeader = int64(64 << 20)

func TestModelImportBindsRangesAndSourceSelection(t *testing.T) {
	if !modelsource.ExactRange(http.StatusPartialContent, 8, "bytes 0-7/4096", 0, 7, 4096) {
		t.Fatal("exact provider range was refused")
	}
	for name, contentRange := range map[string]string{
		"wrong start": "bytes 8-15/4096",
		"wrong total": "bytes 0-7/8192",
	} {
		t.Run(name, func(t *testing.T) {
			if modelsource.ExactRange(http.StatusPartialContent, 8, contentRange, 0, 7, 4096) {
				t.Fatal("mismatched provider range was accepted")
			}
		})
	}

	decode := func(body string) tfs.SourcePlan {
		t.Helper()
		var plan tfs.SourcePlan
		must(t, json.Unmarshal([]byte(body), &plan))
		return plan
	}
	base := `{"profile":"reviewed/1","registry_sha256":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","target":"cozytensors/1","sources":[{"component":"transformer","source_member":"weights/model.safetensors","projected":false}]}`
	if !decode(base).SameSelection(decode(base)) {
		t.Fatal("identical source selections differ")
	}
	for name, changed := range map[string]string{
		"profile": `{"profile":"other/1","registry_sha256":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","target":"cozytensors/1","sources":[{"component":"transformer","source_member":"weights/model.safetensors","projected":false}]}`,
		"target":  `{"profile":"reviewed/1","registry_sha256":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","target":"other/1","sources":[{"component":"transformer","source_member":"weights/model.safetensors","projected":false}]}`,
		"member":  `{"profile":"reviewed/1","registry_sha256":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","target":"cozytensors/1","sources":[{"component":"transformer","source_member":"weights/other.safetensors","projected":false}]}`,
	} {
		t.Run(name, func(t *testing.T) {
			if decode(base).SameSelection(decode(changed)) {
				t.Fatalf("changed %s was accepted", name)
			}
		})
	}
}

type liveImport struct {
	TFS      string `json:"tfs"`
	Registry string `json:"registry,omitempty"`
	Source   string `json:"source"`
	Name     string `json:"name"`
}

func readLiveImport(t *testing.T, path string) liveImport {
	t.Helper()
	var fixture liveImport
	raw, err := os.ReadFile(path)
	must(t, err)
	must(t, json.Unmarshal(raw, &fixture))
	base := filepath.Dir(path)
	for _, field := range []*string{&fixture.TFS, &fixture.Registry, &fixture.Source} {
		if *field != "" && !strings.Contains(*field, "://") && !filepath.IsAbs(*field) {
			*field = filepath.Join(base, *field)
		}
	}
	if fixture.TFS == "" || fixture.Source == "" || fixture.Name == "" {
		t.Fatal("live model-import fixture is incomplete")
	}
	return fixture
}

func importEnv(root string, fixture liveImport) []string {
	env := []string{"COZY_HOME=" + root, "COZY_TFS=" + fixture.TFS,
		"PATH=/usr/local/bin:/usr/bin:/bin", "TENSORHUB_URL=http://127.0.0.1:1"}
	if fixture.Registry != "" {
		env = append(env, "COZY_TFS_REGISTRY="+fixture.Registry)
	}
	return env
}

func TestLiveLocalModelImport(t *testing.T) {
	if *liveModelImportE2E == "" {
		t.Skip("-live-model-import-e2e is not set")
	}
	fixture := readLiveImport(t, *liveModelImportE2E)
	root := t.TempDir()
	raw, err := os.ReadFile(fixture.Source)
	must(t, err)
	copyPath := filepath.Join(root, "source.safetensors")
	must(t, os.WriteFile(copyPath, raw, 0o600))
	fixture.Source = copyPath
	env := importEnv(root, fixture)
	if result := runCozyEnv(env, "model", "import", fixture.Source, "--name", fixture.Name, "--dry-run"); result.code != 0 || !strings.Contains(result.output, "planned") {
		t.Fatalf("local import dry-run = exit %d\n%s", result.code, result.output)
	}
	first := runCozyEnv(env, "model", "import", fixture.Source, "--name", fixture.Name, "--json")
	if first.code != 0 || !strings.Contains(first.output, "local/"+fixture.Name) ||
		!strings.Contains(first.output, "imported") {
		t.Fatalf("local import = exit %d\n%s", first.code, first.output)
	}
	var firstResult map[string]any
	must(t, json.Unmarshal([]byte(first.output), &firstResult))
	raw[len(raw)-1] ^= 0xff
	must(t, os.WriteFile(copyPath, raw, 0o600))
	second := runCozyEnv(env, "model", "import", fixture.Source, "--name", fixture.Name, "--json")
	if second.code != 0 {
		t.Fatalf("local reimport = exit %d\n%s", second.code, second.output)
	}
	var secondResult map[string]any
	must(t, json.Unmarshal([]byte(second.output), &secondResult))
	if firstResult["manifest_id"] == secondResult["manifest_id"] || secondResult["changed"] != true {
		t.Fatalf("local reimport did not atomically advance: first=%#v second=%#v", firstResult, secondResult)
	}
	if _, err := os.Stat(fixture.Source); err != nil {
		t.Fatalf("local source was deleted: %v", err)
	}
	if result := runCozyEnv(env, "model", "list"); result.code != 0 ||
		!strings.Contains(result.output, "local/"+fixture.Name) {
		t.Fatalf("model list after import = exit %d\n%s", result.code, result.output)
	}
}

func TestLivePublicHFModelImportPlan(t *testing.T) {
	if *liveModelImportHF == "" {
		t.Skip("-live-model-import-hf is not set")
	}
	fixture := readLiveImport(t, *liveModelImportHF)
	root := t.TempDir()
	result := runCozyEnv(importEnv(root, fixture), "model", "import", fixture.Source,
		"--name", fixture.Name, "--dry-run")
	if result.code != 0 || !strings.Contains(result.output, "hf://") ||
		!strings.Contains(result.output, "planned") {
		t.Fatalf("public HF import plan = exit %d\n%s", result.code, result.output)
	}
}
