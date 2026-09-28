// Package modelsource owns the foreign-source spelling accepted by model download/upload.
// It identifies a source; TensorFS remains the only code that interprets model bytes.
package modelsource

import (
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

type Kind string

const (
	HuggingFace Kind = "huggingface"
	Civitai     Kind = "civitai"
	LocalFile   Kind = "local-file"
)

// Source is a parsed, credential-free source identity. A Hugging Face source
// with an empty Revision names a moving Reference (a branch or tag) and must be
// resolved to a full commit before it can be persisted or transferred.
type Source struct {
	Kind      Kind
	Canonical string
	Org       string
	Repo      string
	Reference string
	Revision  string
	Member    string // Optional exact Hugging Face carrier, relative to the pinned repository.
	VersionID uint64
	Path      string
	Bytes     int64
}

func Parse(raw, cwd string) (Source, *exit.Error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return Source{}, exit.Usagef("model transfer requires a source")
	}
	switch {
	case strings.HasPrefix(raw, "file:"):
		// Transfers persist this identity and parse it again on the executing
		// machine. It is exactly parseLocal's output, not a file:// URL.
		source, problem := parseLocal(strings.TrimPrefix(raw, "file:"), cwd)
		if problem != nil {
			return Source{}, problem
		}
		if source.Canonical != raw {
			return Source{}, badSource(raw, "local file identities require a canonical absolute path")
		}
		return source, nil
	case strings.HasPrefix(raw, "hf://"):
		return parseHFURI(raw)
	case strings.HasPrefix(raw, "civitai://"):
		return parseCivitaiURI(raw)
	case strings.HasPrefix(raw, "https://"):
		return parseHTTPS(raw)
	case strings.Contains(raw, "://"):
		return Source{}, exit.Named(exit.Usage, "model_source_scheme_unsupported",
			"model source %q does not use hf, civitai, or https", raw).
			WithRemedy("use a pinned hf:// or civitai:// source, an allowlisted provider URL, or an explicit local file")
	default:
		return parseLocal(raw, cwd)
	}
}

func parseHFURI(raw string) (Source, *exit.Error) {
	u, err := url.Parse(raw)
	if err != nil || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return Source{}, badSource(raw, "hf source contains credentials, query, fragment, or invalid escaping")
	}
	head, escapedMember, hasMember := strings.Cut(strings.TrimPrefix(u.EscapedPath(), "/"), "/")
	repo, reference, _ := strings.Cut(head, "@")
	member, memberErr := url.PathUnescape(escapedMember)
	decodedRepo, decodeErr := url.PathUnescape(repo)
	if reference == "" {
		reference = "main"
	}
	if decodeErr != nil || !portablePart(u.Host) || !portablePart(decodedRepo) || !portablePart(reference) ||
		memberErr != nil || (hasMember && !supportedHFMember(member)) {
		return Source{}, badSource(raw, "hf sources are hf://org/repo[@commit-or-branch][/file.safetensors]")
	}
	return hfSource(u.Host, decodedRepo, reference, member), nil
}

// hfSource pins a full commit; any other reference is a moving ref the resolver pins.
func hfSource(org, repo, reference, member string) Source {
	source := Source{Kind: HuggingFace, Org: org, Repo: repo, Reference: reference, Member: member}
	if fullCommit(reference) {
		source.Reference = strings.ToLower(reference)
		source.Revision = source.Reference
	}
	source.Canonical = source.hfCanonical()
	return source
}

func parseCivitaiURI(raw string) (Source, *exit.Error) {
	u, err := url.Parse(raw)
	if err != nil || u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.Path != "" {
		return Source{}, badSource(raw, "civitai sources are civitai://<model-version-id>")
	}
	id, err := strconv.ParseUint(u.Host, 10, 64)
	if err != nil || id == 0 {
		return Source{}, badSource(raw, "civitai sources require one positive model-version id")
	}
	return Source{Kind: Civitai, Canonical: "civitai://" + strconv.FormatUint(id, 10), VersionID: id}, nil
}

func parseHTTPS(raw string) (Source, *exit.Error) {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme != "https" || u.User != nil || u.Fragment != "" {
		return Source{}, badSource(raw, "provider URLs must be credential-free HTTPS URLs")
	}
	for key := range u.Query() {
		lower := strings.ToLower(key)
		if strings.Contains(lower, "token") || strings.Contains(lower, "key") || strings.Contains(lower, "auth") {
			return Source{}, badSource(raw, "provider credentials must not appear in a URL")
		}
	}
	switch strings.ToLower(u.Hostname()) {
	case "huggingface.co", "www.huggingface.co":
		return parseHFPasted(raw, u)
	case "civitai.com", "www.civitai.com":
		return parseCivitaiPasted(raw, u)
	default:
		return Source{}, exit.Named(exit.Validation, "model_source_host_refused",
			"model source host %q is not allowlisted", u.Hostname()).
			WithRemedy("use huggingface.co, civitai.com, or an explicit local file")
	}
}

func parseHFPasted(raw string, u *url.URL) (Source, *exit.Error) {
	parts := splitPath(u.EscapedPath())
	if len(parts) > 4 && strings.HasSuffix(u.EscapedPath(), "/") {
		return Source{}, badSource(raw, "a Hugging Face file URL cannot end with a slash")
	}
	if len(parts) < 2 {
		return Source{}, badSource(raw, "Hugging Face URLs must name org/repo")
	}
	org, err1 := url.PathUnescape(parts[0])
	repo, err2 := url.PathUnescape(parts[1])
	if err1 != nil || err2 != nil || !portablePart(org) || !portablePart(repo) {
		return Source{}, badSource(raw, "Hugging Face URL contains an invalid org or repo")
	}
	reference, member := "main", ""
	if len(parts) > 2 {
		if len(parts) < 4 || (parts[2] != "tree" && parts[2] != "blob" && parts[2] != "resolve") {
			return Source{}, badSource(raw, "use a Hugging Face repository, tree, or tensor file URL")
		}
		reference, _ = url.PathUnescape(parts[3])
		if !portablePart(reference) {
			return Source{}, badSource(raw, "Hugging Face URL names an invalid revision")
		}
		if parts[2] == "tree" {
			if len(parts) != 4 {
				return Source{}, badSource(raw, "select a tensor file, not a repository subdirectory")
			}
		} else {
			var err error
			member, err = url.PathUnescape(strings.Join(parts[4:], "/"))
			if err != nil || !supportedHFMember(member) {
				return Source{}, badSource(raw, "Hugging Face file URLs must select a safe .safetensors file or .safetensors.index.json")
			}
		}
	}
	return hfSource(org, repo, reference, member), nil
}

func supportedHFMember(member string) bool {
	lower := strings.ToLower(member)
	return safeMember(member) && (strings.HasSuffix(lower, ".safetensors") ||
		strings.HasSuffix(lower, ".safetensors.index.json"))
}

func (s Source) hfCanonical() string {
	canonical := "hf://" + s.Org + "/" + s.Repo
	if s.Reference != "main" {
		canonical += "@" + s.Reference
	}
	if s.Member != "" {
		canonical += "/" + escapeMember(s.Member)
	}
	return canonical
}

func parseCivitaiPasted(raw string, u *url.URL) (Source, *exit.Error) {
	parts := splitPath(u.EscapedPath())
	var text string
	if len(parts) == 4 && parts[0] == "api" && parts[1] == "download" && parts[2] == "models" {
		text = parts[3]
	} else if len(parts) >= 2 && parts[0] == "models" {
		values := u.Query()["modelVersionId"]
		if len(values) != 1 {
			return Source{}, badSource(raw, "a Civitai model page must carry exactly one modelVersionId")
		}
		text = values[0]
	} else {
		return Source{}, badSource(raw, "Civitai URLs must name one modelVersionId")
	}
	id, err := strconv.ParseUint(text, 10, 64)
	if err != nil || id == 0 {
		return Source{}, badSource(raw, "Civitai modelVersionId must be a positive integer")
	}
	return Source{Kind: Civitai, Canonical: "civitai://" + strconv.FormatUint(id, 10), VersionID: id}, nil
}

func parseLocal(raw, cwd string) (Source, *exit.Error) {
	if !filepath.IsAbs(raw) && !strings.HasPrefix(raw, ".") {
		return Source{}, badSource(raw, "local model files must be explicit paths such as ./model.safetensors")
	}
	path := raw
	if !filepath.IsAbs(path) {
		path = filepath.Join(cwd, path)
	}
	path, err := filepath.Abs(path)
	if err != nil {
		return Source{}, badSource(raw, "local path cannot be resolved")
	}
	info, err := os.Lstat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return Source{}, exit.New(exit.NotFound, "local model source %s does not exist", path)
		}
		return Source{}, exit.Internalf("cannot inspect local model source %s: %s", path, err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() || info.Size() <= 0 ||
		!strings.HasSuffix(strings.ToLower(info.Name()), ".safetensors") {
		return Source{}, exit.Named(exit.Validation, "model_source_file_refused",
			"local model source %s is not one nonempty regular non-symlink .safetensors file", path)
	}
	return Source{Kind: LocalFile, Canonical: "file:" + filepath.ToSlash(path), Path: path, Bytes: info.Size()}, nil
}

// LocalName returns the canonical spelling of the one portable segment used below
// the reserved local/ namespace. Names are never case-sensitive.
func LocalName(name string) (string, *exit.Error) {
	name = strings.ToLower(strings.TrimSpace(name))
	return name, localName(name)
}

// LocalAlias reads `local/<name>` in any case and returns its canonical name.
func LocalAlias(raw string) (string, bool, *exit.Error) {
	raw = strings.TrimSpace(raw)
	if len(raw) < len("local/") || !strings.EqualFold(raw[:len("local/")], "local/") {
		return "", false, nil
	}
	name, problem := LocalName(raw[len("local/"):])
	return name, true, problem
}

func localName(name string) *exit.Error {
	if len(name) < 1 || len(name) > 128 || !lowerAlphaNum(name[0]) {
		return exit.Usagef("local model name %q is not a portable name", name).
			WithRemedy("use 1-128 letters, digits, dots, dashes, or underscores, starting with a letter or digit")
	}
	for i := 1; i < len(name); i++ {
		if !lowerAlphaNum(name[i]) && !strings.ContainsRune("._-", rune(name[i])) {
			return exit.Usagef("local model name %q is not a portable name", name).
				WithRemedy("use 1-128 letters, digits, dots, dashes, or underscores")
		}
	}
	return nil
}

func splitPath(path string) []string {
	trimmed := strings.Trim(path, "/")
	if trimmed == "" {
		return nil
	}
	return strings.Split(trimmed, "/")
}

func portablePart(part string) bool {
	if part == "" || part == "." || part == ".." || len(part) > 128 {
		return false
	}
	for _, r := range part {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || strings.ContainsRune("._-", r) {
			continue
		}
		return false
	}
	return true
}

func fullCommit(text string) bool {
	if len(text) != 40 {
		return false
	}
	for _, r := range text {
		if !strings.ContainsRune("0123456789abcdefABCDEF", r) {
			return false
		}
	}
	return true
}

func lowerAlphaNum(b byte) bool { return b >= 'a' && b <= 'z' || b >= '0' && b <= '9' }

func badSource(raw, why string) *exit.Error {
	return exit.Named(exit.Usage, "model_source_invalid", "model source %q is invalid: %s", raw, why)
}

func (s Source) String() string { return s.Canonical }
