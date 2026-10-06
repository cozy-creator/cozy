package machines

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// publishedWheel fetches an exact release for this computer and verifies the index digest.
func publishedWheel(ctx context.Context, distribution, version, dir string) (string, error) {
	if version == "" {
		return "", fmt.Errorf("an exact %s version is required", distribution)
	}
	release := distribution + "/" + version
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://pypi.org/pypi/"+release+"/json", nil)
	if err != nil {
		return "", err
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return "", fmt.Errorf("the package index did not answer: %w", err)
	}
	defer response.Body.Close()
	var project struct {
		URLs []struct {
			Filename string            `json:"filename"`
			URL      string            `json:"url"`
			Digests  map[string]string `json:"digests"`
		} `json:"urls"`
	}
	if response.StatusCode != http.StatusOK || json.NewDecoder(io.LimitReader(response.Body, 32<<20)).Decode(&project) != nil {
		return "", fmt.Errorf("the package index has no release %s", release)
	}
	arch := map[string]string{"amd64": "x86_64", "arm64": "aarch64"}[runtime.GOARCH]
	for _, file := range project.URLs {
		if !strings.HasSuffix(file.Filename, ".whl") || !strings.Contains(file.Filename, "linux") || arch == "" || !strings.Contains(file.Filename, arch) {
			continue
		}
		path := filepath.Join(dir, file.Filename)
		if err := download(ctx, file.URL, path, file.Digests["sha256"]); err != nil {
			return "", err
		}
		return path, nil
	}
	return "", fmt.Errorf("%s has no wheel for linux/%s", release, runtime.GOARCH)
}

func download(ctx context.Context, url, path, digest string) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("%s answered %s", url, response.Status)
	}
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	hash := sha256.New()
	_, err = io.Copy(io.MultiWriter(file, hash), response.Body)
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err == nil && hex.EncodeToString(hash.Sum(nil)) != digest {
		err = fmt.Errorf("%s differs from the index's digest", filepath.Base(path))
	}
	return err
}
