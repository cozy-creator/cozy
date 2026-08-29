package wheel

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/exit"
)

// The metadata files the packer reads. Both are read as DATA — parsed, never executed,
// never imported, and never handed to a build backend.
const (
	pyprojectName = "pyproject.toml"
	endpointName  = "endpoint.toml"
)

// project is what `[project]` declares, as data. Absent fields stay empty and are then
// absent from METADATA — the packer never invents a value it was not given.
type project struct {
	present      bool
	name         string
	version      string
	description  string
	requiresPy   string
	dependencies []string
	dynamic      []string
}

// declaration is everything the two metadata files say that the packer acts on.
type declaration struct {
	proj project
	// backend is `[build-system] build-backend`. It is read for ONE purpose: to REFUSE.
	// The packer never dispatches to it, never installs `requires`, and never fires a
	// PEP 517 hook (tensorhub-build.md §0).
	backend    string
	hasBuildSy bool
	// appModule is the module half of endpoint.toml's `[application] object = "mod:attr"`.
	// The wheel must be able to answer for it, or the endpoint is unservable by
	// construction and the packer says so at pack time instead of at import time.
	appModule string
}

var (
	// PEP 508 distribution name.
	reName = regexp.MustCompile(`^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$`)
	// A PEP 440 public version plus the local segment. Deliberately narrower than the
	// full grammar: an endpoint release version this does not admit is a refusal the
	// author can fix, not a wheel with a version pip and we disagree about.
	reVersion = regexp.MustCompile(`^[0-9]+(\.[0-9]+)*((a|b|rc)[0-9]+)?(\.post[0-9]+)?(\.dev[0-9]+)?(\+[a-zA-Z0-9]+([.][a-zA-Z0-9]+)*)?$`)
	// The leading name of a PEP 508 requirement. The rest of the requirement rides
	// through verbatim; what is checked here is that it is one printable ASCII line.
	reReqHead = regexp.MustCompile(`^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?`)
	reEscape  = regexp.MustCompile(`[^\w\d.]+`)
)

func malformed(format string, args ...any) *exit.Error {
	return exit.Named(exit.Validation, "metadata_malformed", format, args...).
		WithRemedy("the packer reads metadata as DATA; it cannot ask a build backend what the project meant").
		WithNext("cozy help endpoint publish")
}

// read parses the two metadata files a tree may carry. A tree with neither is legal:
// its identity then comes entirely from the caller (the release record, or --name and
// --version locally).
func readDeclaration(root string) (declaration, *exit.Error) {
	var d declaration

	if body, err := os.ReadFile(filepath.Join(root, pyprojectName)); err == nil {
		tables, e := parseTOML(pyprojectName, body, map[string]bool{
			"project": true, "build-system": true,
		})
		if e != nil {
			return d, e
		}
		if t, ok := tables["build-system"]; ok {
			d.hasBuildSy = true
			d.backend = t.str("build-backend")
		}
		if t, ok := tables["project"]; ok {
			d.proj.present = true
			d.proj.name = t.str("name")
			d.proj.version = t.str("version")
			d.proj.description = t.str("description")
			d.proj.requiresPy = t.str("requires-python")
			d.proj.dependencies = t.list("dependencies")
			d.proj.dynamic = t.list("dynamic")
		}
	} else if !os.IsNotExist(err) {
		return d, malformed("%s is unreadable: %v", pyprojectName, err)
	}

	if body, err := os.ReadFile(filepath.Join(root, endpointName)); err == nil {
		tables, e := parseTOML(endpointName, body, map[string]bool{"application": true})
		if e != nil {
			return d, e
		}
		if t, ok := tables["application"]; ok {
			obj := t.str("object")
			if obj == "" {
				return d, malformed("%s declares `[application]` with no `object`", endpointName)
			}
			mod, _, ok := strings.Cut(obj, ":")
			if !ok || mod == "" {
				return d, malformed("%s `[application] object = %q` is not `module:attribute`", endpointName, obj)
			}
			d.appModule = mod
		}
	} else if !os.IsNotExist(err) {
		return d, malformed("%s is unreadable: %v", endpointName, err)
	}
	return d, nil
}

// identity settles the distribution name and version. `[project]` is authoritative when
// present; otherwise the caller supplies them, which is the normal endpoint case — an
// endpoint's identity is its RELEASE (`org/endpoint` + version), not a line in its tree.
func identity(d declaration, req Request) (string, string, *exit.Error) {
	name, version := req.Name, req.Version

	if d.proj.present {
		for _, dyn := range d.proj.dynamic {
			return "", "", exit.Named(exit.Validation, "metadata_dynamic",
				"`[project] dynamic` lists %q — a dynamic field is by definition computed by a build backend", dyn).
				WithRemedy("declare %s statically; the packer runs no backend to compute it", dyn)
		}
		if d.proj.name == "" {
			return "", "", malformed("`[project]` declares no `name`")
		}
		if name != "" && normalize(name) != normalize(d.proj.name) {
			return "", "", exit.Named(exit.Validation, "metadata_conflict",
				"--name %q disagrees with `[project] name = %q`", name, d.proj.name).
				WithRemedy("drop --name, or fix the declaration; two names for one wheel is two identities")
		}
		name = d.proj.name
		if d.proj.version != "" {
			if version != "" && version != d.proj.version {
				return "", "", exit.Named(exit.Validation, "metadata_conflict",
					"--version %q disagrees with `[project] version = %q`", version, d.proj.version).
					WithRemedy("drop --version, or fix the declaration")
			}
			version = d.proj.version
		}
	}

	if name == "" {
		return "", "", malformed("no distribution name: the tree declares no `[project] name` and none was supplied").
			WithRemedy("pass --name (the env lane passes the endpoint's release name)")
	}
	if version == "" {
		return "", "", malformed("no version: the tree declares no `[project] version` and none was supplied").
			WithRemedy("pass --version (the env lane passes the release version)")
	}
	if !reName.MatchString(name) {
		return "", "", malformed("%q is not a distribution name (PEP 508)", name)
	}
	if !reVersion.MatchString(version) {
		return "", "", malformed("%q is not a version this packer admits (PEP 440 public + local)", version)
	}
	return name, version, nil
}

// normalize is PEP 503 name normalization — the comparison form.
func normalize(name string) string {
	return strings.ToLower(regexp.MustCompile(`[-_.]+`).ReplaceAllString(name, "-"))
}

// escape is PEP 427's filename escaping — the form that appears in the wheel filename
// and in the `.dist-info` directory name.
func escape(s string) string { return reEscape.ReplaceAllString(s, "_") }

// metadataBody synthesizes core metadata under a FIXED field order. Nothing here is
// derived from the host, the clock, the environment, or the packer's working directory.
func metadataBody(name, version string, d declaration) ([]byte, *exit.Error) {
	var b strings.Builder
	line := func(k, v string) { fmt.Fprintf(&b, "%s: %s\n", k, v) }

	line("Metadata-Version", "2.1")
	line("Name", name)
	line("Version", version)
	if s := d.proj.description; s != "" {
		if strings.ContainsAny(s, "\r\n") {
			return nil, malformed("`[project] description` must be one line")
		}
		line("Summary", s)
	}
	if s := d.proj.requiresPy; s != "" {
		line("Requires-Python", s)
	}
	// Requirements ride in DECLARED order: reordering them would be a second opinion
	// about a list the author wrote, and the order is already fixed by the file.
	for _, dep := range d.proj.dependencies {
		if e := checkRequirement(dep); e != nil {
			return nil, e
		}
		line("Requires-Dist", dep)
	}
	return []byte(b.String()), nil
}

func checkRequirement(dep string) *exit.Error {
	if dep == "" || !reReqHead.MatchString(dep) {
		return malformed("%q is not a PEP 508 requirement", dep)
	}
	for i := 0; i < len(dep); i++ {
		if dep[i] < 0x20 || dep[i] > 0x7e {
			return malformed("requirement %q is not one line of printable ASCII", dep)
		}
	}
	return nil
}

// wheelBody is the WHEEL file: the tag, the generator, and the purelib claim. `Tag` is
// FIXED at py3-none-any — a wheel that would need any other tag is refused upstream by
// the compiled-extension arm, never quietly retagged here.
func wheelBody() []byte {
	return []byte("Wheel-Version: 1.0\n" +
		"Generator: " + PackerVersion + "\n" +
		"Root-Is-Purelib: true\n" +
		"Tag: " + Tag + "\n")
}
