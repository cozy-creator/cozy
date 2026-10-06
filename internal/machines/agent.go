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

	"github.com/cozy-creator/cozy/internal/exit"
)

// MachineAPI is the client API this controller drives a machine with (G/API.md).
const MachineAPI = "cozy.machine.v1"

// servesAPI says whether an executable is a tensord serving MachineAPI, by its own
// `version --json`. Its implementation and release numbers are descriptive.
func servesAPI(ctx context.Context, path string) bool {
	out, err := exec.CommandContext(ctx, path, "version", "--json").Output()
	var version struct {
		Name string   `json:"name"`
		API  []string `json:"api"`
	}
	return err == nil && json.Unmarshal(out, &version) == nil && version.Name == "tensord" && slices.Contains(version.API, MachineAPI)
}

// servesAPI asks the installed machine's executable once per file: a daemon asks on every
// launch, and an install replaces the file under it.
func (h *Host) servesAPI(ctx context.Context) bool {
	info, err := os.Stat(h.binary())
	if err != nil {
		return false
	}
	if h.asked == nil || !os.SameFile(h.asked, info) || !h.asked.ModTime().Equal(info.ModTime()) {
		serves := servesAPI(ctx, h.binary())
		if ctx.Err() != nil {
			return true // the caller gave up before it answered: nothing is known, nothing is kept
		}
		h.asked, h.serves = info, serves
	}
	return h.serves
}

// refuseRenamedNative keeps an old-named native root out of the unreadable legacy
// replacement path. This is a refusal census, never an executable alias or control path.
func (h *Host) refuseRenamedNative(ctx context.Context) *exit.Error {
	if h.servesAPI(ctx) {
		return nil
	}
	previous := filepath.Join(h.Root(), "usr/local/bin/cozy-machine")
	if _, err := os.Stat(previous); err != nil {
		return nil
	}
	out, err := exec.CommandContext(ctx, previous, "version", "--json").Output()
	if ctx.Err() != nil {
		return exit.Named(exit.Unavailable, "machine.identity_unknown", "the existing machine identity could not be read; its root is preserved")
	}
	var identity struct {
		API []string `json:"api"`
	}
	if err != nil || json.Unmarshal(out, &identity) != nil {
		return exit.Named(exit.Unavailable, "machine.identity_unknown", "the existing machine identity could not be read; its root is preserved")
	}
	if !slices.Contains(identity.API, MachineAPI) {
		return nil
	}
	return exit.Named(exit.Conflict, "machine.rename_required", "this root holds an older-named native machine; replacing it as a legacy worker would discard its journal").
		WithRemedy("keep this root and its machine intact; install tensord in a separate machine home for the cutover")
}

// bundledAgent writes the tensord a Runtime wheel bundles (its data scripts) into dir.
func bundledAgent(wheel, dir string) (string, error) {
	archive, err := zip.OpenReader(wheel)
	if err != nil {
		return "", err
	}
	defer archive.Close()
	for _, member := range archive.File {
		if !strings.HasSuffix(member.Name, ".data/scripts/tensord") {
			continue
		}
		reader, err := member.Open()
		if err != nil {
			return "", err
		}
		defer reader.Close()
		path := filepath.Join(dir, "tensord")
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
	return "", fmt.Errorf("%s bundles no tensord", filepath.Base(wheel))
}
