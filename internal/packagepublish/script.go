package packagepublish

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/pelletier/go-toml/v2"
)

type scriptMetadata struct {
	Dependencies   []string       `toml:"dependencies"`
	RequiresPython string         `toml:"requires-python"`
	Tool           map[string]any `toml:"tool"`
}

// PrepareScript adapts a bounded single file into an ordinary private Python
// project. Only uv resolves dependencies; Runtime later discovers the explicit App.
func PrepareScript(ctx context.Context, path string) (*Package, *exit.Error) {
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() {
		return nil, exit.Named(exit.Validation, "script_source_invalid", "script must be a regular file")
	}
	if info.Size() > MaxSourceFileBytes {
		return nil, exit.Named(exit.Validation, "script_source_too_large", "script exceeds the package source-file bound")
	}
	in, err := os.Open(path)
	if err != nil {
		return nil, exit.Internalf("cannot open script: %s", err)
	}
	raw, err := io.ReadAll(io.LimitReader(in, MaxSourceFileBytes+1))
	closeErr := in.Close()
	if err != nil || closeErr != nil || int64(len(raw)) > MaxSourceFileBytes {
		return nil, exit.Named(exit.Validation, "script_source_invalid", "cannot read a bounded complete script")
	}
	metadata, problem := readScriptMetadata(raw)
	if problem != nil {
		return nil, problem
	}
	if metadata.RequiresPython == "" {
		metadata.RequiresPython = ">=3.12"
	}
	// The worker's author surface is an ordinary dependency, captured in uv.lock.
	hasRuntime := false
	for _, requirement := range metadata.Dependencies {
		name := strings.FieldsFunc(requirement, func(r rune) bool {
			return strings.ContainsRune("[<>=!~; @", r)
		})
		if len(name) > 0 && normalizedProjectName(name[0]) == "cozy-runtime" {
			hasRuntime = true
		}
	}
	if !hasRuntime {
		metadata.Dependencies = append(metadata.Dependencies, "cozy-runtime")
	}
	if metadata.Tool != nil {
		// A script's dependency metadata has the same explicit source rules as a
		// project. Relative local sources must keep their original meaning when staged.
		if uv, ok := metadata.Tool["uv"].(map[string]any); ok {
			if sources, ok := uv["sources"].(map[string]any); ok {
				for _, value := range sources {
					if source, ok := value.(map[string]any); ok {
						if relative, ok := source["path"].(string); ok && !filepath.IsAbs(relative) {
							absolute, err := filepath.Abs(filepath.Join(filepath.Dir(path), relative))
							if err != nil {
								return nil, exit.Usagef("cannot resolve local script dependency: %s", err)
							}
							source["path"] = absolute
						}
					}
				}
			}
		}
	}
	digest := sha256.Sum256(raw)
	name := "cozy-script-" + hex.EncodeToString(digest[:12])
	tool := metadata.Tool
	if tool == nil {
		tool = map[string]any{}
	}
	tool["hatch"] = map[string]any{"build": map[string]any{"targets": map[string]any{
		"wheel": map[string]any{"only-include": []string{"cozy_script.py"}},
	}}}
	document := map[string]any{
		"project": map[string]any{
			"name": name, "version": "0.0.0", "requires-python": metadata.RequiresPython,
			"dependencies": metadata.Dependencies,
			"entry-points": map[string]any{applicationGroup: map[string]string{"default": "cozy_script:app"}},
		},
		"build-system": map[string]any{"requires": []string{"hatchling>=1.25"}, "build-backend": "hatchling.build"},
		"tool":         tool,
	}
	project, err := toml.Marshal(document)
	if err != nil {
		return nil, exit.Internalf("cannot encode script project metadata: %s", err)
	}
	root, err := os.MkdirTemp("", "cozy-script-")
	if err != nil {
		return nil, exit.Internalf("cannot prepare script project: %s", err)
	}
	keep := false
	defer func() {
		if !keep {
			_ = os.RemoveAll(root)
		}
	}()
	for filename, contents := range map[string][]byte{
		"cozy_script.py": raw,
		"pyproject.toml": project,
		"package.toml":   []byte("[application]\nobject = \"cozy_script:app\"\n"),
		// The current Runtime contract requires standard CPython 3.12. Keep
		// uv's choice with the snapshot; incompatible script metadata refuses.
		".python-version": []byte("3.12\n"),
	} {
		if err := os.WriteFile(filepath.Join(root, filename), contents, 0o600); err != nil {
			return nil, exit.Internalf("cannot stage script project: %s", err)
		}
	}
	cmd := exec.CommandContext(ctx, "uv", "lock", "--no-progress")
	cmd.Dir, cmd.Env = root, config.Frozen().Tool()
	var output strings.Builder
	cmd.Stdout, cmd.Stderr = &output, &output
	if err := cmd.Run(); err != nil {
		return nil, exit.Named(exit.Structural, "script_dependencies_refused", "uv could not lock the script dependencies").
			WithRemedy("%s", strings.TrimSpace(output.String()))
	}
	pack, problem := PrepareLocalFrom(root)
	if problem != nil {
		return nil, problem
	}
	pack.temporarySource = root
	keep = true
	return pack, nil
}

func readScriptMetadata(raw []byte) (scriptMetadata, *exit.Error) {
	var metadata scriptMetadata
	var block []string
	open, found := false, false
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSuffix(line, "\r")
		if line == "# /// script" {
			if open || found {
				return metadata, exit.Named(exit.Validation, "script_metadata_duplicate", "script has more than one metadata block")
			}
			open = true
			continue
		}
		if !open {
			continue
		}
		if line == "# ///" {
			found, open = true, false
			continue
		}
		if line == "#" {
			block = append(block, "")
		} else if strings.HasPrefix(line, "# ") {
			block = append(block, line[2:])
		} else {
			return metadata, exit.Named(exit.Validation, "script_metadata_invalid", "script metadata must contain comment lines only")
		}
	}
	if !found {
		return metadata, nil
	}
	content := strings.Join(block, "\n")
	if len(content) > maxProjectMetadataBytes {
		return metadata, exit.Named(exit.Validation, "script_metadata_invalid", "script dependency metadata is too large")
	}
	if err := toml.NewDecoder(bytes.NewReader([]byte(content))).DisallowUnknownFields().Decode(&metadata); err != nil {
		return metadata, exit.Named(exit.Validation, "script_metadata_invalid", "invalid PEP 723 metadata: %s", err)
	}
	return metadata, nil
}
