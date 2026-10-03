package machines

import (
	"archive/zip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
)

// HubAccessCapability is the Go agent's delegated Hub access; the v1 API carries the token in
// the run spec instead.
const HubAccessCapability = "hub-access/1"

// MachineAPI is the client API this controller drives a machine with (G/API.md).
const MachineAPI = "cozy.machine.v1"

// servesAPI says whether an executable is a cozy-machine serving MachineAPI, by its own
// `version --json`. Its implementation and release numbers are descriptive.
func servesAPI(ctx context.Context, path string) bool {
	out, err := exec.CommandContext(ctx, path, "version", "--json").Output()
	var version struct {
		Name string   `json:"name"`
		API  []string `json:"api"`
	}
	return err == nil && json.Unmarshal(out, &version) == nil && version.Name == "cozy-machine" && slices.Contains(version.API, MachineAPI)
}

// bundledAgent writes the cozy-machine a Runtime wheel bundles (its data scripts) into dir.
func bundledAgent(wheel, dir string) (string, error) {
	archive, err := zip.OpenReader(wheel)
	if err != nil {
		return "", err
	}
	defer archive.Close()
	for _, member := range archive.File {
		if !strings.HasSuffix(member.Name, ".data/scripts/cozy-machine") {
			continue
		}
		reader, err := member.Open()
		if err != nil {
			return "", err
		}
		defer reader.Close()
		path := filepath.Join(dir, "cozy-machine")
		file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o755)
		if err != nil {
			return "", err
		}
		_, err = io.Copy(file, reader)
		if closeErr := file.Close(); err == nil {
			err = closeErr
		}
		return path, err
	}
	return "", fmt.Errorf("%s bundles no cozy-machine", filepath.Base(wheel))
}
