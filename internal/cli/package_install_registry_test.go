package cli

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
)

func exactHubDocument(t *testing.T, body []byte) hub.ExactDocument {
	t.Helper()
	digest, err := canonical.Spell(canonical.Digest(body))
	if err != nil {
		t.Fatal(err)
	}
	return hub.ExactDocument{CanonicalBytes: body, Digest: digest, Length: int64(len(body))}
}

func TestExactPackageInstallDocument(t *testing.T) {
	body := []byte("[bindings]\n")
	document, problem := exactPackageInstallDocument("package.toml", exactHubDocument(t, body))
	if problem != nil {
		t.Fatal(problem)
	}
	if !bytes.Equal(document.Bytes, body) || document.Length != int64(len(body)) {
		t.Fatalf("validated document = %+v", document)
	}
	body[0] = 'x'
	if document.Bytes[0] != '[' {
		t.Fatal("validated document retained the mutable response buffer")
	}

	bad := exactHubDocument(t, []byte("{}"))
	bad.Length++
	if _, problem := exactPackageInstallDocument("package descriptor", bad); problem == nil ||
		problem.Name != "hub.package_install_document_invalid" {
		t.Fatalf("changed length was not refused: %v", problem)
	}
	if _, problem := exactPackageInstallDocument("package.toml", hub.ExactDocument{}); problem == nil {
		t.Fatal("empty package.toml identity was accepted")
	}
}

func TestPublishedDefaultBindingsUsesOnlyExactMetadata(t *testing.T) {
	root := t.TempDir()
	runtime := filepath.Join(root, "cozy-runtime")
	script := `#!/bin/sh
set -eu
[ "$1" = "--json" ]
[ "$2" = "--dir" ]
[ "$4" = "--descriptor" ]
[ "$6" = "bindings" ]
IFS= read -r package_line < "$3/package.toml"
IFS= read -r descriptor_line < "$5"
[ "$package_line" = "[bindings]" ]
[ "$descriptor_line" = "{}" ]
printf '%s\n' '{"bindings":[{"model_binding_path":"generate.model","model_parameter_name":"model","model_class":"sdxl","ref":"paul/wai@17.0.0","lane":"bf16","source":"package.toml:path"}]}'
`
	if err := os.WriteFile(runtime, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	configDocument := exactHubDocument(t, []byte("[bindings]\n"))
	descriptorDocument := exactHubDocument(t, []byte("{}\n"))
	toInstall := func(document hub.ExactDocument) install.ExactDocument {
		return install.ExactDocument{Bytes: document.CanonicalBytes,
			Digest: document.Digest, Length: document.Length}
	}
	bindings, problem := publishedDefaultBindings(context.Background(), config.Config{Home: root},
		root, runtime, toInstall(configDocument), toInstall(descriptorDocument))
	if problem != nil {
		t.Fatal(problem)
	}
	if len(bindings) != 1 || bindings[0].ModelBindingPath != "generate.model" ||
		bindings[0].Ref != "paul/wai@17.0.0" || bindings[0].Lane != "bf16" {
		t.Fatalf("bindings = %+v", bindings)
	}
}

func TestPublishedDefaultBindingsRejectsDuplicateSlots(t *testing.T) {
	root := t.TempDir()
	runtime := filepath.Join(root, "cozy-runtime")
	script := `#!/bin/sh
printf '%s\n' '{"bindings":[{"model_binding_path":"generate.model","model_parameter_name":"model","model_class":"sdxl","ref":"paul/a@1.0.0","lane":"bf16","source":"package.toml:path"},{"model_binding_path":"generate.model","model_parameter_name":"model","model_class":"sdxl","ref":"paul/b@1.0.0","lane":"bf16","source":"package.toml:path"}]}'
`
	if err := os.WriteFile(runtime, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	configDocument := exactHubDocument(t, []byte("[bindings]\n"))
	descriptorDocument := exactHubDocument(t, []byte("{}\n"))
	bindings, problem := publishedDefaultBindings(context.Background(), config.Config{Home: root},
		root, runtime,
		install.ExactDocument{Bytes: configDocument.CanonicalBytes, Digest: configDocument.Digest,
			Length: configDocument.Length},
		install.ExactDocument{Bytes: descriptorDocument.CanonicalBytes, Digest: descriptorDocument.Digest,
			Length: descriptorDocument.Length})
	if problem == nil || problem.Name != "runtime_bindings_invalid" || bindings != nil {
		t.Fatalf("duplicate bindings were not refused: bindings=%+v problem=%v", bindings, problem)
	}
}
