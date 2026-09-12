package runtimeoperation

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

const EnvironmentFile = "runtime-operation.json"

// Environment is local bookkeeping for the trusted Runtime's prepared generation.
// Its three identities are checked against the immutable install before its path
// can be used. It contains no calling package identity or remote authority.
type Environment struct {
	Python            string `json:"python"`
	EnvironmentDigest string `json:"environment_digest"`
	InterfaceDigest   string `json:"interface_digest"`
	SourceDigest      string `json:"source_digest"`
}

func ReadEnvironment(root, environmentDigest, interfaceDigest, sourceDigest string) (string, *exit.Error) {
	path := filepath.Join(root, EnvironmentFile)
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Size() > 4096 {
		return "", exit.New(exit.Conflict, "Runtime operation environment is absent or not bounded metadata")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", exit.New(exit.Conflict, "Runtime operation environment changed")
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	var environment Environment
	if decoder.Decode(&environment) != nil || decoder.Decode(&struct{}{}) != io.EOF ||
		environment.EnvironmentDigest != environmentDigest || environment.InterfaceDigest != interfaceDigest || environment.SourceDigest != sourceDigest {
		return "", exit.New(exit.Conflict, "Runtime operation environment differs from its retained install")
	}
	for _, digest := range []string{environmentDigest, interfaceDigest, sourceDigest} {
		if _, err := canonical.Raw(digest); err != nil {
			return "", exit.New(exit.Conflict, "Runtime operation environment has an invalid identity")
		}
	}
	if problem := ValidateOwnedPython(root, environment.Python); problem != nil {
		return "", problem
	}
	return environment.Python, nil
}

// ValidateOwnedPython checks the real containing directory without rejecting the
// ordinary venv link to its external interpreter executable.
func ValidateOwnedPython(root, python string) *exit.Error {
	if !filepath.IsAbs(python) || filepath.Base(python) != "python" {
		return exit.New(exit.Conflict, "Runtime operation interpreter is not an exact local path")
	}
	// A venv's python may link to its external base executable. Its containing
	// directory, however, must be the real owned generation under this install.
	directory, err := filepath.EvalSymlinks(filepath.Dir(python))
	if err != nil {
		return exit.New(exit.Conflict, "Runtime operation generation is unavailable")
	}
	owned, err := filepath.EvalSymlinks(root)
	if err != nil {
		return exit.New(exit.Conflict, "Runtime operation install is unavailable")
	}
	relative, err := filepath.Rel(owned, directory)
	if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return exit.New(exit.Conflict, "Runtime operation interpreter escaped its owned install")
	}
	executable, err := os.Stat(python)
	if err != nil || !executable.Mode().IsRegular() || executable.Mode()&0o111 == 0 {
		return exit.New(exit.Conflict, "Runtime operation interpreter is unavailable")
	}
	return nil
}
