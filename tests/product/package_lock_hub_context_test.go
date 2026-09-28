package producttest

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
)

// `--tensorhub <name>` is the named context's URL and its login, as for every verb: a
// project with an account dependency locks against the account index of the Hub the name
// selects, as the account signed in there by that same name.
func TestPackageLockUsesTheNamedHubContextsLogin(t *testing.T) {
	if _, err := exec.LookPath("cozy-runtime"); err != nil { //cozy:allow the host's own tool, as in host_runtime_test.go
		t.Skip("package lock selects its Python through the host cozy-runtime tool")
	}
	const dependency = "ctx-dep"
	wheel := contextDependencyWheel(t, dependency)
	digest := fmt.Sprintf("%x", sha256.Sum256(wheel))
	filename := "ctx_dep-1.0.0-py3-none-any.whl"
	hub := newAccountHubWith(t, func(mux *http.ServeMux) {
		mux.HandleFunc("GET /v1/index/{org}/simple/{distribution}/", func(w http.ResponseWriter, r *http.Request) {
			if r.PathValue("org") != "proof" || r.PathValue("distribution") != dependency {
				http.NotFound(w, r)
				return
			}
			fmt.Fprintf(w, `<a href="/v1/index/proof/files/%s/%s#sha256=%s">%s</a><br>`, digest, filename, digest, filename)
		})
		mux.HandleFunc("GET /v1/index/proof/files/{digest}/{filename}", func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write(wheel)
		})
	})
	root := t.TempDir()
	// The current hub answers nothing; only the named context reaches the stand-in.
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: http://127.0.0.1:1\nhubs:\n  standin: "+hub.URL+"\n"), 0o600))
	login := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "auth", "login", "proof@example.test", "--tensorhub", "standin")
	login.Env = childEnv(t, root)
	login.Stdin = strings.NewReader("123456\nproof\n") //cozy:stdin-value test login code and account name
	var out bytes.Buffer
	login.Stdout, login.Stderr = &out, &out
	if err := login.Run(); err != nil {
		t.Fatalf("login by context name: %v\n%s", err, out.String())
	}

	project := t.TempDir()
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[project]
name = "ctx-lock-proof"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["`+dependency+`"]

[tool.uv.sources]
`+dependency+` = { index = "tensorhub" }
`), 0o644))
	lock := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "package", "lock", "--tensorhub", "standin", "--json")
	lock.Dir, lock.Env = project, childEnv(t, root)
	raw, err := lock.CombinedOutput()
	if err != nil {
		t.Fatalf("lock by context name: %v\n%s", err, raw)
	}
	var locked struct {
		AccountIndex string `json:"account_index"`
	}
	index := hub.URL + "/v1/index/proof/simple/"
	if json.Unmarshal(raw, &locked) != nil || locked.AccountIndex != index {
		t.Fatalf("the lock did not use the named hub's account index %s: %s", index, raw)
	}
	written, err := os.ReadFile(filepath.Join(project, "uv.lock"))
	must(t, err)
	if !strings.Contains(string(written), index) {
		t.Fatalf("uv.lock does not resolve %s from the named hub:\n%s", dependency, written)
	}
}

func contextDependencyWheel(t *testing.T, name string) []byte {
	t.Helper()
	var out bytes.Buffer
	archive := zip.NewWriter(&out)
	info := strings.ReplaceAll(name, "-", "_") + "-1.0.0.dist-info/"
	record := ""
	for _, member := range [][2]string{
		{"ctx_dep.py", "VALUE = 1\n"},
		{info + "METADATA", "Metadata-Version: 2.3\nName: " + name + "\nVersion: 1.0.0\n"},
		{info + "WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"},
	} {
		writer, err := archive.Create(member[0])
		must(t, err)
		_, err = writer.Write([]byte(member[1]))
		must(t, err)
		sum := sha256.Sum256([]byte(member[1]))
		record += fmt.Sprintf("%s,sha256=%s,%d\n", member[0], base64.RawURLEncoding.EncodeToString(sum[:]), len(member[1]))
	}
	writer, err := archive.Create(info + "RECORD")
	must(t, err)
	_, err = writer.Write([]byte(record + info + "RECORD,,\n"))
	must(t, err)
	must(t, archive.Close())
	return out.Bytes()
}
