package producttest

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

type intakeWheel struct {
	name, filename, digest string
	data                   []byte
}

func intakeWheels(t *testing.T, count int) []intakeWheel {
	t.Helper()
	var wheels []intakeWheel
	for n := range count {
		name := fmt.Sprintf("intake-library-%d", n)
		path := prebuiltProjectWheel(t, t.TempDir(), name, "py3-none-any", "weightless:app")
		data, err := os.ReadFile(path)
		must(t, err)
		wheels = append(wheels, intakeWheel{name, filepath.Base(path), fmt.Sprintf("%x", sha256.Sum256(data)), data})
	}
	return wheels
}

// Separate source trees and staging directories model a Runtime-only local
// package revision: the selected dependency wheel identities remain unchanged.
func intakeProject(t *testing.T, origin, runtimeVersion string, wheels []intakeWheel) (string, string, map[string]map[string]string) {
	t.Helper()
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, "pyproject.toml"), []byte("[project]\nname='capture-root'\nversion='1.0.0'\n"), 0600))
	lock := "version=1\n[[package]]\nname='capture-root'\nversion='1.0.0'\nsource={editable='.'}\n"
	closure := "capture-root==1.0.0"
	selected := map[string]string{}
	for _, w := range wheels {
		object := origin + "/v1/index/paul/files/" + w.digest + "/" + w.filename
		lock += fmt.Sprintf("[[package]]\nname=%q\nversion='1.0.0'\nsource={registry=%q}\nwheels=[{url=%q,hash=%q,size=%d}]\n", w.name, origin+"/v1/index/paul/simple/", object, "sha256:"+w.digest, len(w.data))
		closure += "\n" + w.name + "==1.0.0"
		selected[w.name] = "1.0.0"
	}
	if runtimeVersion != "" {
		// Image-owned public wheels retain locked requirements; intake does not
		// fetch their bytes merely to inspect callable dependency entry points.
		lock += fmt.Sprintf("[[package]]\nname='cozy-runtime'\nversion=%q\nsource={registry='https://pypi.org/simple'}\nwheels=[{url=%q,hash=%q,size=100}]\n", runtimeVersion,
			"https://files.pythonhosted.org/packages/aa/bb/"+strings.Repeat("1", 60)+"/cozy_runtime-"+runtimeVersion+"-py3-none-any.whl", "sha256:"+strings.Repeat("2", 64))
		closure += "\ncozy-runtime==" + runtimeVersion
		selected["cozy-runtime"] = runtimeVersion
	}
	must(t, os.WriteFile(filepath.Join(root, "uv.lock"), []byte(lock), 0600))
	return root, closure, map[string]map[string]string{"callable": selected}
}

func TestWheelIntakeCacheReusesUnchangedClosureAcrossRuntimeRevision(t *testing.T) {
	wheels := intakeWheels(t, 46)
	var requests atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		for _, item := range wheels {
			if strings.HasSuffix(r.URL.Path, "/"+item.filename) {
				w.Write(item.data)
				return
			}
		}
		http.NotFound(w, r)
	}))
	defer server.Close()
	cache := home.Paths(t.TempDir()).DependencyCache()
	var first map[string]packagepublish.CapturedDependency
	for i, version := range []string{"0.23.1", "0.23.2"} {
		root, closure, selected := intakeProject(t, server.URL, version, wheels)
		captured, problem := packagepublish.CaptureWheelDependencies(t.Context(), root, "capture-root", closure, t.TempDir(), cache, selected)
		fatal(t, problem)
		if requests.Load() != 46 {
			t.Fatalf("capture %d fetched unchanged wheel bytes: %d HTTP requests", i, requests.Load())
		}
		if captured["cozy-runtime"].Version != version {
			t.Fatal("Runtime-only change was hidden by cache reuse")
		}
		for _, item := range wheels {
			got := captured[item.name]
			data, err := os.ReadFile(got.Path)
			must(t, err)
			if !bytes.Equal(data, item.data) || got.Digest != "sha256:"+item.digest || !got.Application || got.Package != "paul/"+item.name {
				t.Fatalf("capture lost locked bytes or callable identity: %+v", got)
			}
			if i == 1 && got.Path == first[item.name].Path {
				t.Fatal("capture reused another invocation's scratch path")
			}
		}
		if i == 0 {
			first = captured
			// Staging is independent from the immutable cache.
			must(t, os.WriteFile(first[wheels[0].name].Path, []byte("scratch changed"), 0600))
		}
	}
}

func TestWheelIntakeCacheRepairsCorruptionAndRejectsChangedBytes(t *testing.T) {
	wheels := intakeWheels(t, 1)
	item := wheels[0]
	var requests atomic.Int64
	var bad atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		data := append([]byte(nil), item.data...)
		if bad.Load() {
			data[0] ^= 1 // Preserve length: hash verification must catch this.
		}
		w.Write(data)
	}))
	defer server.Close()
	cache := home.Paths(t.TempDir()).DependencyCache()
	capture := func() (map[string]packagepublish.CapturedDependency, *exit.Error) {
		root, closure, selected := intakeProject(t, server.URL, "", wheels)
		return packagepublish.CaptureWheelDependencies(t.Context(), root, "capture-root", closure, t.TempDir(), cache, selected)
	}
	_, problem := capture()
	fatal(t, problem)
	cached := filepath.Join(cache, item.digest, item.filename)
	must(t, os.Chmod(cached, 0600))
	corrupt := append([]byte(nil), item.data...)
	corrupt[0] ^= 1
	must(t, os.WriteFile(cached, corrupt, 0600))
	bad.Store(true)
	if _, problem := capture(); problem == nil || problem.Name != "private_dependency_download_changed" {
		t.Fatalf("corrupt cache or changed replacement was served: %v", problem)
	}
	bad.Store(false)
	captured, problem := capture()
	fatal(t, problem)
	data, err := os.ReadFile(captured[item.name].Path)
	must(t, err)
	if !bytes.Equal(data, item.data) || requests.Load() != 3 {
		t.Fatal("corrupt wheel was not repaired from verified bytes")
	}
	_, problem = capture()
	fatal(t, problem)
	if requests.Load() != 3 {
		t.Fatal("repaired wheel was fetched again")
	}
}

func TestWheelIntakeCacheConcurrentCapturesIgnoreIncompleteDownloads(t *testing.T) {
	wheels := intakeWheels(t, 1)
	item := wheels[0]
	var requests atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if requests.Add(1) == 1 {
			w.Header().Set("Content-Length", fmt.Sprint(len(item.data)))
			w.Write(item.data[:len(item.data)/2])
			return
		}
		time.Sleep(30 * time.Millisecond)
		w.Write(item.data)
	}))
	defer server.Close()
	cache := home.Paths(t.TempDir()).DependencyCache()
	root, closure, selected := intakeProject(t, server.URL, "", wheels)
	if _, problem := packagepublish.CaptureWheelDependencies(t.Context(), root, "capture-root", closure, t.TempDir(), cache, selected); problem == nil {
		t.Fatal("truncated download was admitted")
	}
	if _, err := os.Stat(filepath.Join(cache, item.digest, item.filename)); !os.IsNotExist(err) {
		t.Fatal("failed download acquired a committed cache name")
	}
	// A process that dies mid-download can leave scratch, never a cache hit.
	partial := filepath.Join(cache, item.digest, ".partial-abandoned")
	must(t, os.Mkdir(partial, 0700))
	must(t, os.WriteFile(filepath.Join(partial, item.filename), item.data[:20], 0600))
	type result struct {
		captured map[string]packagepublish.CapturedDependency
		problem  *exit.Error
	}
	results := make(chan result, 8)
	for range 8 {
		stage := t.TempDir()
		go func() {
			captured, problem := packagepublish.CaptureWheelDependencies(t.Context(), root, "capture-root", closure, stage, cache, selected)
			results <- result{captured, problem}
		}()
	}
	for range 8 {
		r := <-results
		fatal(t, r.problem)
		data, err := os.ReadFile(r.captured[item.name].Path)
		must(t, err)
		if !bytes.Equal(data, item.data) {
			t.Fatal("parallel reader received partial bytes")
		}
	}
	if requests.Load() != 2 {
		t.Fatalf("parallel captures downloaded the same wheel repeatedly: %d", requests.Load())
	}
}
