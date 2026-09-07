package packagepublish

import (
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/pelletier/go-toml/v2"
)

// WithChildInterfaces prepares a private copy whose invocable dependencies are
// exact lightweight interface wheels. The author's project and lock never change.
// Unrelated local libraries retain their original declarations and are captured
// by the ordinary snapshotter after this interface-only lock adjustment.
func WithChildInterfaces(ctx context.Context, parent *Package, replacements map[string]string) (*Package, *exit.Error) {
	if len(replacements) == 0 {
		return parent, nil
	}
	dependencies, problem := LocalDependencyPaths(parent.Tree)
	if problem != nil {
		return nil, problem
	}
	before, _, _, problem := parent.SourceIdentity()
	if problem != nil {
		return nil, problem
	}
	root, err := os.MkdirTemp("", "cozy-child-interfaces-")
	if err != nil {
		return nil, exit.Internalf("cannot stage private interfaces: %s", err)
	}
	fail := func(problem *exit.Error) (*Package, *exit.Error) { _ = os.RemoveAll(root); return nil, problem }
	for name, source := range parent.Files {
		to := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(to), 0o700); err != nil {
			return fail(exit.Internalf("cannot stage parent source: %s", err))
		}
		input, err := os.Open(source)
		if err != nil {
			return fail(exit.Internalf("cannot read parent source: %s", err))
		}
		output, err := os.OpenFile(to, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err != nil {
			input.Close()
			return fail(exit.Internalf("cannot capture parent source: %s", err))
		}
		n, copyErr := io.Copy(output, io.LimitReader(input, MaxSourceFileBytes+1))
		input.Close()
		closeErr := output.Close()
		if copyErr != nil || closeErr != nil || n > MaxSourceFileBytes {
			return fail(exit.New(exit.Conflict, "parent source changed while capturing child interfaces"))
		}
	}
	after, _, _, problem := parent.SourceIdentity()
	if problem != nil || after != before {
		return fail(exit.New(exit.Conflict, "parent source or dependency changed during interface capture"))
	}
	path := filepath.Join(root, "pyproject.toml")
	raw, err := os.ReadFile(path)
	if err != nil {
		return fail(exit.Internalf("cannot read parent metadata: %s", err))
	}
	var document map[string]any
	if err := toml.Unmarshal(raw, &document); err != nil {
		return fail(exit.New(exit.Validation, "parent metadata is not TOML"))
	}
	nested := func(parent map[string]any, key string) map[string]any {
		value, ok := parent[key].(map[string]any)
		if !ok {
			value = map[string]any{}
			parent[key] = value
		}
		return value
	}
	uv := nested(nested(document, "tool"), "uv")
	sources := nested(uv, "sources")
	for name, path := range dependencies {
		sources[name] = map[string]any{"path": path}
	}
	for name, path := range replacements {
		sources[name] = map[string]any{"path": path}
	}
	delete(uv, "workspace") // all selected local members now have exact explicit paths
	raw, err = toml.Marshal(document)
	if err != nil {
		return fail(exit.Internalf("cannot encode private interface metadata: %s", err))
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		return fail(exit.Internalf("cannot retain private interface metadata: %s", err))
	}
	command := exec.CommandContext(ctx, "uv", "lock", "--no-progress")
	command.Dir = root
	command.Env = config.Frozen().Tool()
	var log strings.Builder
	command.Stdout, command.Stderr = &log, &log
	if err := command.Run(); err != nil {
		detail := strings.TrimSpace(log.String())
		if len(detail) > 2000 {
			detail = detail[len(detail)-2000:]
		}
		return fail(exit.Named(exit.Structural, "child.interface_lock_refused", "cannot lock the captured interface dependencies: %s", detail))
	}
	prepared, problem := PrepareLocalFrom(root)
	if problem != nil {
		return fail(problem)
	}
	prepared.temporarySource = root
	return prepared, nil
}
