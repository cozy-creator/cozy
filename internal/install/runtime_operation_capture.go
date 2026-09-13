package install

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/runtimeoperation"
)

const runtimeCaptureDirectory = "runtime-operations"
const runtimeCaptureFile = "capture.json"

// CaptureRuntimeOperations belongs to the caller install's ordinary lifetime.
// Its immutable base snapshot is independent of that caller's dependency closure.
func CaptureRuntimeOperations(ctx context.Context, layout home.Layout, parent records.PackageInstall) *exit.Error {
	root := filepath.Join(parent.Dir, runtimeCaptureDirectory)
	if _, err := os.Lstat(filepath.Join(root, runtimeCaptureFile)); err == nil {
		_, problem := readRuntimeCapture(root)
		return problem
	} else if !os.IsNotExist(err) {
		return exit.New(exit.Conflict, "captured Runtime metadata is unavailable")
	}
	env := config.Frozen().Tool("COZY_HOME="+runtimeScratchHome(),
		"COZY_DEPENDENCY_CACHE="+layout.DependencyCache())
	tool, problem := launch.BuiltinOperationsTool(layout.Root, runtimeScratchHome(), env)
	if problem != nil {
		return problem
	}
	version, problem := tool.RuntimeVersion(ctx)
	if problem != nil {
		return problem
	}
	surface, problem := tool.BuiltinOperations(ctx)
	if problem != nil {
		return problem
	}
	name, raw, problem := runtimeoperation.Carrier(version, surface.Digest)
	if problem != nil {
		return problem
	}
	if err := os.MkdirAll(root, 0700); err != nil {
		return exit.Internalf("cannot stage Runtime capture: %s", err)
	}
	wheel := filepath.Join(root, name)
	if err := os.WriteFile(wheel, raw, 0400); err != nil {
		return exit.Internalf("cannot stage Runtime carrier: %s", err)
	}
	captured, problem := tool.CaptureBuiltin(ctx, wheel, filepath.Join(root, "base"))
	if problem != nil {
		return problem
	}
	actual, problem := launch.DecodePackageInterface(captured.PackageInterface)
	if problem != nil {
		return problem
	}
	if captured.RuntimeVersion != version || actual.Digest != surface.Digest || !bytes.Equal(actual.Raw, surface.Raw) {
		return exit.New(exit.Conflict, "Runtime changed its builtin during capture")
	}
	if problem := validateRuntimeCapture(root, captured); problem != nil {
		return problem
	}
	data, err := json.Marshal(captured)
	if err != nil {
		return exit.Internalf("cannot encode Runtime capture: %s", err)
	}
	if err := os.WriteFile(filepath.Join(root, runtimeCaptureFile), data, 0400); err != nil {
		return exit.Internalf("cannot retain Runtime capture: %s", err)
	}
	return nil
}

func readRuntimeCapture(root string) (launch.BuiltinPreparation, *exit.Error) {
	var capture launch.BuiltinPreparation
	path := filepath.Join(root, runtimeCaptureFile)
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Size() > canonical.DocMax {
		return capture, exit.New(exit.Conflict, "Runtime operation has no bounded captured base")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return capture, exit.New(exit.Conflict, "captured Runtime metadata is unavailable")
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&capture) != nil || decoder.Decode(&struct{}{}) != io.EOF {
		return capture, exit.New(exit.Conflict, "captured Runtime metadata changed its closed schema")
	}
	return capture, validateRuntimeCapture(root, capture)
}

func validateRuntimeCapture(root string, capture launch.BuiltinPreparation) *exit.Error {
	for _, digest := range []string{capture.EnvironmentDigest, capture.ContentDigest, capture.ReceiptDigest, capture.ImplementationDigest} {
		if _, err := canonical.Raw(digest); err != nil {
			return exit.New(exit.Conflict, "captured Runtime omitted exact identities")
		}
	}
	surface, problem := launch.DecodePackageInterface(capture.PackageInterface)
	if problem != nil {
		return problem
	}
	if surface.Application != runtimeoperation.Application || len(surface.Jobs) != 1 || surface.Jobs[0].Name != "quantize" {
		return exit.New(exit.Conflict, "captured Runtime changed its fixed builtin")
	}
	if !strings.Contains("\n"+capture.Closure+"\n", "\ncozy-runtime=="+capture.RuntimeVersion+"\n") {
		return exit.New(exit.Conflict, "captured Runtime omitted its actual distribution")
	}
	return runtimeoperation.ValidateOwnedPython(root, capture.EnvironmentPython)
}

// HasRuntimeOperationsCapture preserves the script's orchestration role without
// preparing a numerical environment or creating a speculative child binding.
func HasRuntimeOperationsCapture(parent records.PackageInstall) (bool, *exit.Error) {
	root := filepath.Join(parent.Dir, runtimeCaptureDirectory)
	if _, err := os.Lstat(filepath.Join(root, runtimeCaptureFile)); os.IsNotExist(err) {
		return false, nil
	}
	_, problem := readRuntimeCapture(root)
	return problem == nil, problem
}

// ResolveRuntimeOperations creates the ordinary binding only on an actual call.
// The writer lock serializes concurrent calls and publishes one immutable choice.
func ResolveRuntimeOperations(ctx context.Context, layout home.Layout, store *records.Store, parent records.PackageInstall, iface string) (*records.ChildBinding, *exit.Error) {
	writer, problem := Lock(layout)
	if problem != nil {
		return nil, problem
	}
	defer writer.Unlock()
	held, problem := store.ChildBinding(parent.ID, iface, runtimeoperation.Module, "quantize")
	if problem != nil || held != nil {
		return held, problem
	}
	root := filepath.Join(parent.Dir, runtimeCaptureDirectory)
	capture, problem := readRuntimeCapture(root)
	if problem != nil {
		return nil, problem
	}
	surface, problem := launch.DecodePackageInterface(capture.PackageInterface)
	if problem != nil {
		return nil, problem
	}
	if surface.Digest != iface {
		return nil, exit.Named(exit.Conflict, "child.builtin_changed", "call differs from its captured Runtime interface")
	}
	env := config.Frozen().Tool("COZY_HOME="+runtimeScratchHome(),
		"COZY_DEPENDENCY_CACHE="+layout.DependencyCache())
	tool := launch.CapturedBuiltinTool(root, capture.EnvironmentPython, env)
	binding, _, problem := prepareRuntimeOperations(ctx, layout, store, tool, capture)
	if problem != nil {
		return nil, problem
	}
	binding.ParentInstallID = parent.ID
	if problem := store.RecordChildBindings([]records.ChildBinding{*binding}); problem != nil {
		return nil, problem
	}
	return binding, nil
}
