package install

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

type CaptureInput struct {
	Source, Key, Python, Extra string
}

// CaptureCaller groups replacement ownership. It is never a content identity.
func CaptureCaller(path string) (string, *exit.Error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", exit.Internalf("cannot identify local capture caller: %s", err)
	}
	if resolved, err := filepath.EvalSymlinks(abs); err == nil {
		abs = resolved
	}
	return captureDigest([]byte(filepath.Clean(abs))), nil
}

// CaptureIdentity rewalks source and hashes actual qualified tool bytes. An
// unqualified/editable host tool still follows ordinary installation, but cannot
// establish a reusable capture. Build backends belong to the completed install;
// source, lock, explicit reinstall, and capture-tool changes select a new one.
func CaptureIdentity(ctx context.Context, pack *packagepublish.Package) (*CaptureInput, *exit.Error) {
	current, problem := packagepublish.PrepareLocalFrom(pack.Tree)
	if problem != nil {
		return nil, problem
	}
	defer current.Close()
	var warnings []string
	extra := pickCUDAExtra(current.Tree, &warnings)
	var extras []string
	if extra != "" {
		extras = strings.Fields(extra)
	}
	source, _, _, problem := current.SourceIdentity(extras...)
	if problem != nil {
		return nil, problem
	}
	input := &CaptureInput{Source: source}
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		return input, nil
	}
	env := config.Frozen().Tool()
	bin, problem := hostruntime.Path(env)
	if problem != nil {
		return nil, problem
	}
	runtimeIdentity, ok := CaptureSDKIdentity(bin)
	if !ok {
		return input, nil
	}
	python := exec.CommandContext(ctx, "uv", "python", "find", "--system", "--no-python-downloads")
	python.Dir, python.Env = current.Tree, env
	raw, err := python.Output()
	if err != nil {
		return input, nil
	}
	pythonPath := strings.TrimSpace(string(raw))
	version := strings.TrimSpace(runOut(pythonPath, "-I", "-S", "-V"))
	if !strings.HasPrefix(version, "Python 3.12.") {
		return input, nil
	}
	input.Python, input.Extra = strings.TrimPrefix(version, "Python "), extra
	uv, err := exec.LookPath("uv")
	if err != nil {
		return input, nil
	}
	creator, err := os.Executable()
	if err != nil {
		return input, nil
	}
	type document struct {
		Source, Extra, Platform, Python, Runtime string
		Tools                                    []captureFile
		Environment                              []string
	}
	doc := document{Source: source, Extra: extra, Platform: runtime.GOOS + "/" + runtime.GOARCH,
		Python: version, Runtime: runtimeIdentity}
	for _, tool := range []struct{ name, path string }{{"python", pythonPath}, {"uv", uv}, {"creator", creator}} {
		file, ok := captureFileIdentity(tool.name, tool.path)
		if !ok {
			return input, nil
		}
		doc.Tools = append(doc.Tools, file)
	}
	for _, value := range env {
		name, _, _ := strings.Cut(value, "=")
		if name == "UV_CACHE_DIR" || name == "UV_PROJECT_ENVIRONMENT" || name == "UV_PYTHON_DOWNLOADS" {
			continue
		}
		if strings.HasPrefix(name, "UV_") || strings.HasPrefix(name, "PIP_") ||
			name == "CC" || name == "CXX" || name == "CFLAGS" || name == "CPPFLAGS" || name == "LDFLAGS" {
			doc.Environment = append(doc.Environment, value)
		}
	}
	sort.Strings(doc.Environment)
	raw, _ = json.Marshal(doc)
	input.Key = captureDigest(raw)
	return input, nil
}

type captureFile struct {
	Name, Digest string
	Length       int64
}

func captureDigest(raw []byte) string {
	sum := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func captureFileIdentity(name, path string) (captureFile, bool) {
	file, err := os.Open(path)
	if err != nil {
		return captureFile{}, false
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() > 256<<20 {
		return captureFile{}, false
	}
	hash := sha256.New()
	n, err := io.Copy(hash, io.LimitReader(file, (256<<20)+1))
	if err != nil || n != info.Size() {
		return captureFile{}, false
	}
	return captureFile{Name: name, Digest: "sha256:" + hex.EncodeToString(hash.Sum(nil)), Length: n}, true
}

// CaptureSDKIdentity hashes an admitted Runtime command's complete SDK environment.
// Hash actual Runtime and dependency files and verify every declared hash;
// version strings or RECORD text alone cannot hide a same-version edit.
// Editable or oversized SDKs conservatively follow ordinary installation.
func CaptureSDKIdentity(command string) (string, bool) {
	resolved, err := filepath.EvalSymlinks(command)
	if err != nil {
		return "", false
	}
	prefix := filepath.Dir(filepath.Dir(resolved))
	metadata := venvMetadata(prefix)
	version := metadata["version_info"]
	if version == "" {
		version = metadata["version"]
	}
	if version != "3.12" && !strings.HasPrefix(version, "3.12.") {
		return "", false
	}
	site := filepath.Join(prefix, "lib", "python3.12", "site-packages")
	runtimes, err := filepath.Glob(filepath.Join(site, "cozy_runtime-*.dist-info"))
	if err != nil || len(runtimes) != 1 {
		return "", false
	}
	directories, err := filepath.Glob(filepath.Join(site, "*.dist-info"))
	if err != nil || len(directories) == 0 || len(directories) > 4096 {
		return "", false
	}
	var files []captureFile
	var total int64
	seen := map[string]bool{}
	for _, directory := range directories {
		rows, ok := captureDistribution(prefix, site, directory)
		if !ok {
			return "", false
		}
		for _, row := range rows {
			if seen[row.Name] {
				return "", false
			}
			seen[row.Name] = true
			total += row.Length
			if total > 512<<20 || len(files) >= 20000 {
				return "", false
			}
			files = append(files, row)
		}
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Name < files[j].Name })
	raw, _ := json.Marshal(files)
	return captureDigest(raw), true
}

func captureDistribution(prefix, site, directory string) ([]captureFile, bool) {
	directPath := filepath.Join(directory, "direct_url.json")
	if info, err := os.Lstat(directPath); err == nil {
		if !info.Mode().IsRegular() || info.Size() > 1<<20 {
			return nil, false
		}
		raw, err := os.ReadFile(directPath)
		var direct struct {
			Directory struct{ Editable bool } `json:"dir_info"`
		}
		if err != nil || json.Unmarshal(raw, &direct) != nil || direct.Directory.Editable {
			return nil, false
		}
	} else if !os.IsNotExist(err) {
		return nil, false
	}
	recordPath := filepath.Join(directory, "RECORD")
	info, err := os.Lstat(recordPath)
	if err != nil || !info.Mode().IsRegular() || info.Size() > 4<<20 {
		return nil, false
	}
	record, err := os.Open(recordPath)
	if err != nil {
		return nil, false
	}
	defer record.Close()
	reader := csv.NewReader(io.LimitReader(record, 4<<20))
	reader.FieldsPerRecord = 3
	var files []captureFile
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil || len(files) >= 20000 {
			return nil, false
		}
		path := filepath.Clean(filepath.Join(site, filepath.FromSlash(row[0])))
		rel, err := filepath.Rel(prefix, path)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return nil, false
		}
		if filepath.Dir(path) == directory {
			switch filepath.Base(path) {
			case "direct_url.json", "INSTALLER", "REQUESTED", "RECORD":
				continue // only this distribution's installer metadata is location-dependent
			}
		}
		if row[1] == "" && strings.HasSuffix(path, ".pyc") && filepath.Base(filepath.Dir(path)) == "__pycache__" {
			module, _, tagged := strings.Cut(filepath.Base(path), ".cpython-")
			if tagged {
				source := filepath.Join(filepath.Dir(filepath.Dir(path)), module+".py")
				if info, err := os.Lstat(source); err == nil && info.Mode().IsRegular() {
					continue // generated bytecode is derived from the hashed source beside it
				}
			}
		}
		file, ok := captureFileIdentity(row[0], path)
		if !ok {
			return nil, false
		}
		if row[1] != "" {
			value, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(row[1], "sha256="))
			if err != nil || !strings.HasPrefix(row[1], "sha256=") || hex.EncodeToString(value) != strings.TrimPrefix(file.Digest, "sha256:") {
				return nil, false
			}
		}
		files = append(files, file)
	}
	return files, len(files) > 0
}
