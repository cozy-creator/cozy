// Package executionowner validates the immutable admission carried from a client
// to the existing Creator coordinator on its private worker.
package executionowner

import (
	"bytes"
	"encoding/json"
	"io"
	"regexp"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

const CapsuleFormat = "cozy.creator.private-execution/1"

// Host owns this private fixed-command argument shape; it carries no client paths.
const AuthorityFormat = "cozy.creator.execution-authority/1"

// A capsule contains identities and callable inventory, never executable paths,
// credentials, a database export, or an invocation schedule. Its package wheels
// travel through the existing signed LocalPackageUpload lane.
type Capsule struct {
	Format   string    `json:"format"`
	Root     Root      `json:"root"`
	Packages []Package `json:"packages"`
	Bindings []Binding `json:"bindings"`
}

type Root struct {
	RequestID      string          `json:"request_id"`
	Revision       string          `json:"revision"`
	Entrypoint     string          `json:"entrypoint"`
	Input          json.RawMessage `json:"input"`
	IdempotencyKey string          `json:"idempotency_key"`
}

var rootID = regexp.MustCompile(`^job-[a-f0-9]{24}$`)

// Revision is the existing canonical LocalPackageRevision document. The inline
// interface is its exact declared reference; validation imports no Python code.
type Package struct {
	Revision  json.RawMessage `json:"revision"`
	Interface json.RawMessage `json:"interface"`
	Capture   Capture         `json:"capture"`
}

// Capture preserves the source install's observed metadata. These are captured
// client facts, not a claim that the pod has already materialized this environment.
type Capture struct {
	Python   string `json:"python"`
	Platform string `json:"platform"`
	Closure  string `json:"closure"`
	Extra    string `json:"extra"`
}

type Binding struct {
	ParentRevision  string `json:"parent_revision"`
	ChildRevision   string `json:"child_revision"`
	InterfaceDigest string `json:"interface_digest"`
	Module          string `json:"module"`
	Export          string `json:"export"`
	Entrypoint      string `json:"entrypoint"`
}

// Validated preserves the signed bytes and their parsed documents. A successful
// decode is metadata validation only; the importer must verify uploaded objects
// and record the root before acknowledging execution admission.
type Validated struct {
	Capsule  Capsule
	Digest   string
	Raw      []byte
	Packages map[string]canonical.Doc
	Surfaces map[string]*launch.PackageInterface
}

func Encode(c Capsule) ([]byte, *exit.Error) {
	raw, err := json.Marshal(c)
	if err != nil {
		return nil, invalid("capsule cannot be encoded")
	}
	raw, err = canonical.NormalizeJCS(raw)
	if err != nil {
		return nil, invalid("capsule is not canonical JSON")
	}
	if _, problem := Decode(raw); problem != nil {
		return nil, problem
	}
	return raw, nil
}

func Decode(raw []byte) (*Validated, *exit.Error) {
	if len(raw) == 0 || len(raw) > pb.MaxInlineControlBytes {
		return nil, invalid("capsule exceeds its control document bound")
	}
	normalized, err := canonical.NormalizeJCS(raw)
	if err != nil || !bytes.Equal(raw, normalized) {
		return nil, invalid("capsule must be exact canonical JSON")
	}
	var c Capsule
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&c) != nil || decoder.Decode(&struct{}{}) != io.EOF {
		return nil, invalid("capsule differs from its closed schema")
	}
	if c.Format != CapsuleFormat || len(c.Packages) == 0 || len(c.Packages) > 1024 || len(c.Bindings) > 1024 ||
		!rootID.MatchString(c.Root.RequestID) || c.Root.IdempotencyKey == "" || len(c.Root.IdempotencyKey) > 256 || !identifier(c.Root.Entrypoint, false) {
		return nil, invalid("capsule admission is incomplete or exceeds its inventory bound")
	}
	var input map[string]json.RawMessage
	if json.Unmarshal(c.Root.Input, &input) != nil || input == nil {
		return nil, invalid("root input must be a typed object")
	}
	v := &Validated{Capsule: c, Raw: append([]byte(nil), raw...), Packages: map[string]canonical.Doc{}, Surfaces: map[string]*launch.PackageInterface{}}
	v.Digest, _ = canonical.Spell(canonical.Digest(raw))
	for _, pkg := range c.Packages {
		doc, err := canonical.Read(pkg.Revision, &pb.LocalPackageRevision{})
		if err != nil || !strings.HasPrefix(doc.Str("package"), "local/") || doc.Str("release") == "" {
			return nil, invalid("capsule package is not an exact unpublished revision")
		}
		id, _ := canonical.Spell(canonical.Digest(pkg.Revision))
		if v.Packages[id] != nil {
			return nil, invalid("capsule repeats a package revision")
		}
		if pkg.Capture.Platform != "linux/amd64" || !strings.HasPrefix(pkg.Capture.Python, "3.12.") ||
			!strings.Contains("\n"+pkg.Capture.Closure+"\n", "\n"+strings.TrimPrefix(doc.Str("package"), "local/")+"=="+doc.Str("release")+"\n") {
			return nil, invalid("capsule omitted its captured source environment facts")
		}
		surface, problem := launch.DecodePackageInterface(pkg.Interface)
		ref := doc.Sub("package_interface")
		if problem != nil || !bytes.Equal(surface.Raw, pkg.Interface) || surface.Digest != ref.Str("digest") || int64(len(pkg.Interface)) != ref.Int("length") {
			return nil, invalid("capsule interface differs from its revision reference")
		}
		files := doc.List("files")
		if len(files) == 0 || len(files) > pb.MaxLocalPackageFiles {
			return nil, invalid("capsule revision has no bounded wheel inventory")
		}
		seenFiles, seenNames := map[string]bool{}, map[string]bool{}
		for _, file := range files {
			name, digest := file.Str("filename"), file.Str("digest")
			if name == "" || name != strings.TrimSpace(name) || strings.ContainsAny(name, "/\\") || !strings.HasSuffix(name, ".whl") ||
				file.Int("length") <= 0 || seenFiles[digest] || seenNames[name] {
				return nil, invalid("capsule revision contains an invalid or duplicate wheel")
			}
			seenFiles[digest], seenNames[name] = true, true
		}
		v.Packages[id], v.Surfaces[id] = doc, surface
	}
	root := v.Surfaces[c.Root.Revision]
	if root == nil {
		return nil, invalid("root revision is outside the captured inventory")
	}
	entry, problem := root.Function(c.Root.Entrypoint)
	if problem != nil || entry.Kind != "job" {
		return nil, invalid("root does not name a captured job")
	}
	edges, seen := map[string][]string{}, map[string]bool{}
	for _, binding := range c.Bindings {
		child := v.Surfaces[binding.ChildRevision]
		if v.Packages[binding.ParentRevision] == nil || child == nil || child.Digest != binding.InterfaceDigest ||
			!identifier(binding.Module, true) || !identifier(binding.Export, false) {
			return nil, invalid("callable binding escapes the captured inventory")
		}
		entry, problem := child.Function(binding.Entrypoint)
		if problem != nil || entry.Invocable == nil || entry.Invocable.Module != binding.Module || entry.Invocable.Export != binding.Export {
			return nil, invalid("callable binding differs from its captured export")
		}
		key := binding.ParentRevision + "\x00" + binding.InterfaceDigest + "\x00" + binding.Module + "\x00" + binding.Export
		if seen[key] {
			return nil, invalid("capsule repeats a callable binding")
		}
		seen[key] = true
		edges[binding.ParentRevision] = append(edges[binding.ParentRevision], binding.ChildRevision)
	}
	queue, reachable := []string{c.Root.Revision}, map[string]bool{}
	for len(queue) > 0 {
		id := queue[0]
		queue = queue[1:]
		if reachable[id] {
			continue
		}
		reachable[id] = true
		queue = append(queue, edges[id]...)
	}
	if len(reachable) != len(v.Packages) {
		return nil, invalid("capsule includes packages outside its root's captured call inventory")
	}
	return v, nil
}

func identifier(value string, dotted bool) bool {
	if len(value) == 0 || len(value) > 255 {
		return false
	}
	for _, part := range strings.Split(value, ".") {
		if len(part) == 0 || !dotted && part != value {
			return false
		}
		for index, char := range part {
			if char != '_' && (char < 'a' || char > 'z') && (char < 'A' || char > 'Z') && (index == 0 || char < '0' || char > '9') {
				return false
			}
		}
	}
	return true
}

func invalid(detail string) *exit.Error {
	return exit.Named(exit.Validation, "execution.capsule_invalid", "%s", detail)
}
