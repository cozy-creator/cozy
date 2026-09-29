package machines

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	pep440 "github.com/aquasecurity/go-pep440-version"
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// AgentModule is the standalone machine server, built from the Runtime repository.
const AgentModule = "github.com/cozy-creator/cozy-runtime/machine-agent"

const agentReleases = "https://api.github.com/repos/cozy-creator/cozy/releases"

type agentManifest struct {
	Name         string `json:"name"`
	Version      string `json:"version"`
	WireMinor    uint32 `json:"wire_minor"`
	MinimumMinor uint32 `json:"minimum_wire_minor"`
	Artifacts    []struct {
		OS, Arch, URL, SHA256 string
	} `json:"artifacts"`
}

type agentRelease struct {
	Tag        string `json:"tag_name"`
	Draft      bool   `json:"draft"`
	Prerelease bool   `json:"prerelease"`
	Assets     []struct {
		Name string `json:"name"`
		URL  string `json:"browser_download_url"`
	} `json:"assets"`
}

// PublishedAgent selects the newest compatible public machine release. Agent releases use
// their own immutable tags; GitHub's latest release belongs to the CLI, not this server.
// An explicit --host bypasses discovery for development, never the shipping default.
func (h *Host) PublishedAgent(ctx context.Context) (string, *exit.Error) {
	path, err := h.publishedAgent(ctx, http.DefaultClient, agentReleases)
	if err != nil {
		return "", exit.Named(exit.Unavailable, "machine.agent_unavailable", "cannot install cozy-machine: %s", err).
			WithRemedy("retry `cozy machine install`, or supply a built machine agent with --host")
	}
	return path, nil
}

func (h *Host) publishedAgent(ctx context.Context, client *http.Client, releasesURL string) (string, error) {
	var releases []agentRelease
	for page := 1; ; page++ {
		var batch []agentRelease
		if err := agentJSON(ctx, client, fmt.Sprintf("%s?per_page=100&page=%d", releasesURL, page), &batch); err != nil {
			return "", err
		}
		for _, release := range batch {
			if !release.Draft && !release.Prerelease && strings.HasPrefix(release.Tag, "machine-v") {
				if _, err := pep440.Parse(strings.TrimPrefix(release.Tag, "machine-v")); err == nil {
					releases = append(releases, release)
				}
			}
		}
		if len(batch) < 100 {
			break
		}
	}
	slices.SortFunc(releases, func(a, b agentRelease) int {
		left, _ := pep440.Parse(strings.TrimPrefix(a.Tag, "machine-v"))
		right, _ := pep440.Parse(strings.TrimPrefix(b.Tag, "machine-v"))
		return right.Compare(left)
	})
	for _, release := range releases {
		manifestURL := ""
		for _, asset := range release.Assets {
			if asset.Name == "machine-agent.json" {
				manifestURL = asset.URL
				break
			}
		}
		if manifestURL == "" {
			continue
		}
		var manifest agentManifest
		if err := agentJSON(ctx, client, manifestURL, &manifest); err != nil {
			return "", err
		}
		if manifest.Name != "cozy-machine" || "machine-v"+manifest.Version != release.Tag || manifest.MinimumMinor > manifest.WireMinor {
			return "", fmt.Errorf("%s has an invalid machine manifest", release.Tag)
		}
		if manifest.WireMinor < pb.MinCompatibleWireMinor || manifest.MinimumMinor > pb.WireMinor {
			continue
		}
		for _, artifact := range manifest.Artifacts {
			if artifact.OS != runtime.GOOS || artifact.Arch != runtime.GOARCH {
				continue
			}
			want, err := hex.DecodeString(artifact.SHA256)
			if err != nil || len(want) != sha256.Size || strings.ToLower(artifact.SHA256) != artifact.SHA256 {
				return "", fmt.Errorf("%s has an invalid artifact digest", release.Tag)
			}
			directory := h.path(filepath.Join("agents", manifest.Version))
			path := filepath.Join(directory, "cozy-machine")
			if digest, err := fileDigest(path); err == nil && digest == artifact.SHA256 {
				return path, nil
			}
			if err := os.MkdirAll(directory, 0o700); err != nil {
				return "", err
			}
			if err := downloadAgent(ctx, client, artifact.URL, path, artifact.SHA256); err != nil {
				return "", err
			}
			if HostModule(path) != AgentModule {
				return "", fmt.Errorf("%s is not a cozy-machine executable", release.Tag)
			}
			return path, nil
		}
	}
	return "", fmt.Errorf("no compatible cozy-machine release for %s/%s", runtime.GOOS, runtime.GOARCH)
}

func agentRequest(ctx context.Context, client *http.Client, address string) (*http.Response, error) {
	parsed, err := url.Parse(address)
	if err != nil || parsed.Scheme != "https" || parsed.Host == "" || parsed.User != nil {
		return nil, fmt.Errorf("machine release URL must use HTTPS")
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, address, nil)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Accept", "application/json")
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		response.Body.Close()
		return nil, fmt.Errorf("machine release server answered HTTP %d", response.StatusCode)
	}
	return response, nil
}

func agentJSON(ctx context.Context, client *http.Client, address string, into any) error {
	response, err := agentRequest(ctx, client, address)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(response.Body, (8<<20)+1))
	if err != nil {
		return err
	}
	if len(raw) > 8<<20 {
		return fmt.Errorf("machine release metadata exceeds 8 MiB")
	}
	return json.Unmarshal(raw, into)
}

func downloadAgent(ctx context.Context, client *http.Client, address, destination, digest string) error {
	response, err := agentRequest(ctx, client, address)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	file, err := os.CreateTemp(filepath.Dir(destination), ".cozy-machine-*")
	if err != nil {
		return err
	}
	defer os.Remove(file.Name())
	defer file.Close()
	hash := sha256.New()
	n, err := io.Copy(io.MultiWriter(file, hash), io.LimitReader(response.Body, (256<<20)+1))
	if err != nil {
		return err
	}
	if n > 256<<20 || hex.EncodeToString(hash.Sum(nil)) != digest {
		return fmt.Errorf("cozy-machine artifact does not match its published digest or size bound")
	}
	if err := file.Chmod(0o755); err != nil {
		return err
	}
	if err := file.Sync(); err != nil {
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return os.Rename(file.Name(), destination)
}
