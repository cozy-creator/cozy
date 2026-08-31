// Package modelsource owns the foreign-source spelling accepted by model import.
// It identifies a source; TensorFS remains the only code that interprets model bytes.
package modelsource

import (
	"fmt"
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
// with an empty Revision came from a moving pasted URL and must be resolved to
// a full commit before it can be persisted or transferred.
type Source struct {
	Kind      Kind
	Canonical string
	Org       string
	Repo      string
	Revision  string
	VersionID uint64
	Path      string
	Bytes     int64
}

func Parse(raw, cwd string) (Source, *exit.Error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return Source{}, exit.Usagef("model import requires a source")
	}
	switch {
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
	org := u.Host
	repo, revision, ok := strings.Cut(strings.TrimPrefix(u.EscapedPath(), "/"), "@")
	decodedRepo, decodeErr := url.PathUnescape(repo)
	if !ok || decodeErr != nil || !portablePart(org) || !portablePart(decodedRepo) || !fullCommit(revision) {
		return Source{}, badSource(raw, "hf sources are hf://org/repo@<40-character-commit>")
	}
	revision = strings.ToLower(revision)
	return Source{Kind: HuggingFace, Canonical: "hf://" + org + "/" + decodedRepo + "@" + revision,
		Org: org, Repo: decodedRepo, Revision: revision}, nil
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
	if len(parts) < 2 {
		return Source{}, badSource(raw, "Hugging Face URLs must name org/repo")
	}
	org, err1 := url.PathUnescape(parts[0])
	repo, err2 := url.PathUnescape(parts[1])
	if err1 != nil || err2 != nil || !portablePart(org) || !portablePart(repo) {
		return Source{}, badSource(raw, "Hugging Face URL contains an invalid org or repo")
	}
	revision := ""
	if len(parts) > 2 {
		if len(parts) != 4 || parts[2] != "tree" {
			return Source{}, badSource(raw, "use a repository or repository tree URL, not an arbitrary Hugging Face path")
		}
		revision, _ = url.PathUnescape(parts[3])
		if !fullCommit(revision) {
			// A branch/tag is moving. The provider resolver deliberately resolves it once.
			revision = ""
		}
	}
	canonical := "hf://" + org + "/" + repo
	if revision != "" {
		revision = strings.ToLower(revision)
		canonical += "@" + revision
	}
	return Source{Kind: HuggingFace, Canonical: canonical, Org: org, Repo: repo, Revision: revision}, nil
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
	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return Source{}, exit.New(exit.NotFound, "local model source %s does not exist", path)
		}
		return Source{}, exit.Internalf("cannot inspect local model source %s: %s", path, err)
	}
	if !info.Mode().IsRegular() || info.Size() <= 0 || !strings.HasSuffix(strings.ToLower(info.Name()), ".safetensors") {
		return Source{}, exit.Named(exit.Validation, "model_source_file_refused",
			"local model source %s is not one nonempty regular .safetensors file", path)
	}
	return Source{Kind: LocalFile, Canonical: "file:" + filepath.ToSlash(path), Path: path, Bytes: info.Size()}, nil
}

// LocalName validates the one portable segment used below the reserved local/ namespace.
func LocalName(name string) *exit.Error {
	if len(name) < 1 || len(name) > 128 || !lowerAlphaNum(name[0]) {
		return exit.Usagef("local model name %q is not a portable name", name).
			WithRemedy("use 1-128 lowercase letters, digits, dots, dashes, or underscores, starting with a letter or digit")
	}
	for i := 1; i < len(name); i++ {
		if !lowerAlphaNum(name[i]) && !strings.ContainsRune("._-", rune(name[i])) {
			return exit.Usagef("local model name %q is not a portable name", name).
				WithRemedy("use 1-128 lowercase letters, digits, dots, dashes, or underscores")
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

func (s Source) Description() string {
	switch s.Kind {
	case HuggingFace:
		return fmt.Sprintf("Hugging Face %s/%s", s.Org, s.Repo)
	case Civitai:
		return fmt.Sprintf("Civitai model version %d", s.VersionID)
	default:
		return s.Path
	}
}
