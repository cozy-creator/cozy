package hostruntime

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	pep440 "github.com/aquasecurity/go-pep440-version"
	"github.com/pelletier/go-toml/v2"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Describe is the one reading of a package surface this host performs — at publish pre-flight
// and at install: `cozy-runtime describe` parses the source under dir and imports nothing, and
// answers the canonical PackageInterface bytes. Its reader is at least the cozy-runtime the
// package's uv.lock pins (the SDK it is written against), or the newest released when it pins
// none, brought forward first when this host's is older (atLeast). A refusal names both.
func Describe(ctx context.Context, env []string, dir string) ([]byte, *exit.Error) {
	sdk, locked := lockedSDK(dir)
	if !locked {
		sdk = newestRelease(ctx)
	}
	minimum := release
	if sdk != nil && minimum.LessThan(*sdk) {
		minimum = *sdk
	}
	reader, problem := atLeast(env, minimum)
	if problem != nil {
		return nil, problem
	}
	cmd := exec.CommandContext(ctx, reader.path, "--json", "--dir", dir, "describe")
	cmd.Env = env
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if cmd.ProcessState == nil {
		return nil, exit.Internalf("cannot run %s: %s", reader.path, err)
	}
	if code := cmd.ProcessState.ExitCode(); code != 0 {
		return nil, refused(RuntimeExit(code, "describe", "runtime_query_failed", stdout.String(), stderr.String()),
			reader.release, sdk, locked)
	}
	return bytes.TrimSuffix([]byte(stdout.String()), []byte("\n")), nil
}

// refused names the releases on both sides of a refusal. A reader older than the package's SDK
// (one that could not be brought forward) may not know what the SDK added, and says so.
func refused(problem *exit.Error, reader pep440.Version, sdk *pep440.Version, locked bool) *exit.Error {
	pins := "the package pins no cozy-runtime"
	if locked {
		pins = "the package is locked to cozy-runtime " + sdk.String()
	}
	if sdk == nil || !reader.LessThan(*sdk) {
		problem.Message += fmt.Sprintf(" (read by cozy-runtime %s; %s)", reader.String(), pins)
		return problem
	}
	if !locked {
		pins += ", and the newest is " + sdk.String()
	}
	return exit.Named(problem.Code, "host_runtime_older_than_package",
		"cozy-runtime %s refused this package and is older than its SDK (%s), whose constructs it may not know: %s",
		reader.String(), pins, problem.Message).
		WithRemedy("install cozy-runtime %s or newer: %s — then retry", sdk.String(), installCommand(sdk.String()))
}

// lockedSDK is the newest cozy-runtime release dir's uv.lock pins, as a public version.
func lockedSDK(dir string) (*pep440.Version, bool) {
	raw, err := os.ReadFile(filepath.Join(dir, "uv.lock"))
	var lock struct {
		Package []struct {
			Name    string `toml:"name"`
			Version string `toml:"version"`
		} `toml:"package"`
	}
	if err != nil || toml.Unmarshal(raw, &lock) != nil {
		return nil, false
	}
	var newest *pep440.Version
	for _, entry := range lock.Package {
		if entry.Name != Distribution {
			continue
		}
		found, err := pep440.Parse(entry.Version)
		if err == nil {
			found, err = pep440.Parse(found.Public())
		}
		if err == nil && (newest == nil || newest.LessThan(found)) {
			newest = &found
		}
	}
	return newest, newest != nil
}

// newestRelease is the newest cozy-runtime on the package index, or nil when it cannot be asked.
func newestRelease(ctx context.Context) *pep440.Version {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://pypi.org/pypi/"+Distribution+"/json", nil)
	if err != nil {
		return nil
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return nil
	}
	defer response.Body.Close()
	var project struct {
		Info struct {
			Version string `json:"version"`
		} `json:"info"`
	}
	if response.StatusCode != http.StatusOK || json.NewDecoder(io.LimitReader(response.Body, 64<<20)).Decode(&project) != nil {
		return nil
	}
	found, err := pep440.Parse(project.Info.Version)
	if err != nil {
		return nil
	}
	return &found
}
