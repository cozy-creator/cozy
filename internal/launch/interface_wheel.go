package launch

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	pep440 "github.com/aquasecurity/go-pep440-version"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

const interfaceGeneratorABI = "cozy.interface-generator/5"
const interfaceGeneratorRuntimeFloor = "0.11.0"

func GenerateInterfaceWheel(ctx context.Context, install records.PackageInstall, home string, env []string, implementation, source, sourceDigest, output string) (InterfaceWheel, *exit.Error) {
	runtime := RuntimeCLI{Bin: Binary(install), Dir: install.ProjectDir, Home: home, Env: env}
	return runtime.InterfaceWheel(ctx, PackageInterfacePath(install.Dir), strings.TrimPrefix(install.Package, "local/"), install.Version, implementation, source, sourceDigest, output)
}

type InterfaceWheel struct {
	Filename     string `json:"filename"`
	Path         string `json:"path"`
	Digest       string `json:"digest"`
	Length       int64  `json:"length"`
	GeneratorABI string `json:"generator_abi"`
}

func (r RuntimeCLI) InterfaceWheel(ctx context.Context, interfacePath, distribution, version, implementation, source, sourceDigest, output string) (InterfaceWheel, *exit.Error) {
	var wheel InterfaceWheel
	var identity struct {
		Distribution string `json:"distribution"`
	}
	if problem := r.callContext(ctx, &identity, "version"); problem != nil {
		return wheel, problem
	}
	release, err := pep440.Parse(identity.Distribution)
	if err != nil || release.LessThan(pep440.MustParse(interfaceGeneratorRuntimeFloor)) {
		return wheel, exit.Named(exit.Structural, "interface_runtime_below_floor",
			"editable dependency %s uses cozy-runtime %q; %s requires Runtime %s or newer",
			distribution, identity.Distribution, interfaceGeneratorABI, interfaceGeneratorRuntimeFloor).
			WithRemedy("upgrade this dependency project's cozy-runtime requirement and uv.lock to >=%s (uv lock --upgrade-package cozy-runtime), then retry cozy run", interfaceGeneratorRuntimeFloor)
	}
	raw, _ := json.Marshal(map[string]string{"interface_path": interfacePath, "distribution": distribution, "version": version, "implementation_digest": implementation, "output_directory": output, "implementation_wheel": source, "implementation_wheel_digest": sourceDigest})
	raw, err = canonical.NormalizeJCS(raw)
	if err != nil {
		return wheel, exit.Internalf("cannot encode interface generation request: %s", err)
	}
	if problem := r.callInputContext(ctx, raw, &wheel, "interface-wheel"); problem != nil {
		return wheel, problem
	}
	if filepath.Base(wheel.Filename) != wheel.Filename || filepath.Clean(wheel.Path) != filepath.Join(output, wheel.Filename) || wheel.Length <= 0 || wheel.Length > 256<<20 || wheel.GeneratorABI != interfaceGeneratorABI {
		return wheel, exit.New(exit.Validation, "interface generator returned an invalid bounded artifact")
	}
	info, err := os.Lstat(wheel.Path)
	if err != nil || !info.Mode().IsRegular() || info.Size() != wheel.Length {
		return wheel, exit.New(exit.Validation, "interface wheel changed after generation")
	}
	data, err := os.ReadFile(wheel.Path)
	if err != nil {
		return wheel, exit.Internalf("cannot read generated interface wheel: %s", err)
	}
	digest, _ := canonical.Spell(canonical.Digest(data))
	if digest != wheel.Digest {
		return wheel, exit.New(exit.Conflict, "interface wheel digest changed after generation")
	}
	return wheel, nil
}
