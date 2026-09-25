package modelsource

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/url"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

type MetadataFile struct {
	Length int64  `json:"length"`
	SHA256 string `json:"sha256"`
}

// VerifyMetadata checks the exact reviewed small files before accepting paid work.
// Weight bodies are never fetched here; they remain native source operations.
func (r *Resolver) VerifyMetadata(ctx context.Context, source Source, files map[string]MetadataFile) *exit.Error {
	if source.Kind != HuggingFace || !fullCommit(source.Revision) || len(files) == 0 || len(files) > 256 {
		return exit.New(exit.Validation, "model metadata requires a pinned HF source and bounded file selection")
	}
	var total int64
	names := make([]string, 0, len(files))
	for name, expected := range files {
		if name == "" || strings.HasPrefix(name, "/") || strings.Contains(name, "\\") || expected.Length <= 0 {
			return exit.New(exit.Validation, "model metadata selection contains an invalid member")
		}
		for _, part := range strings.Split(name, "/") {
			if part == ".." || part == "." || part == "" {
				return exit.New(exit.Validation, "model metadata member is not contained")
			}
		}
		if expected.Length > (64<<20)-total {
			return exit.New(exit.Validation, "model metadata exceeds 64 MiB")
		}
		total += expected.Length
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		expected := files[name]
		location := "https://huggingface.co/" + url.PathEscape(source.Org) + "/" + url.PathEscape(source.Repo) + "/resolve/" + source.Revision + "/" + escapeMember(name)
		raw, problem := r.small(ctx, location, expected.Length)
		if problem != nil {
			return problem
		}
		digest := sha256.Sum256(raw)
		if int64(len(raw)) != expected.Length || hex.EncodeToString(digest[:]) != expected.SHA256 {
			return exit.Named(exit.Conflict, "model_source.metadata_changed", "model metadata %s differs from its reviewed bytes", name)
		}
	}
	return nil
}
