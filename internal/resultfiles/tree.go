package resultfiles

import (
	"encoding/json"
	"path"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

const TreeMediaType = "application/vnd.cozy.tree-manifest"

type TreeMember struct {
	Path, Digest string
	Length       int64
}

func ParseTreeManifest(raw []byte, contentBytes int64) ([]TreeMember, *exit.Error) {
	fail := func() ([]TreeMember, *exit.Error) {
		return nil, exit.New(exit.Conflict, "tree manifest has an invalid file closure")
	}
	// Runtime writes the manifest and its digest is already verified; members another
	// version adds are ignored.
	if len(raw) == 0 || len(raw) > 1<<20 || contentBytes < 0 {
		return fail()
	}
	var document map[string]json.RawMessage
	if json.Unmarshal(raw, &document) != nil {
		return fail()
	}
	var entries []map[string]json.RawMessage
	if json.Unmarshal(document["entries"], &entries) != nil || len(entries) == 0 {
		return fail()
	}
	var members []TreeMember
	names := map[string]bool{}
	sizes := map[string]int64{}
	var total int64
	for _, entry := range entries {
		var kind, name string
		var blob map[string]json.RawMessage
		var sha string
		var length int64
		if json.Unmarshal(entry["kind"], &kind) != nil || kind != "file" || json.Unmarshal(entry["path"], &name) != nil || json.Unmarshal(entry["blob"], &blob) != nil || json.Unmarshal(blob["sha256"], &sha) != nil || json.Unmarshal(blob["length"], &length) != nil || length < 0 || length > contentBytes-total {
			return fail()
		}
		if name == "." || path.Clean(name) != name || !filepath.IsLocal(filepath.FromSlash(name)) || strings.ContainsAny(name, "\\\x00") || names[name] {
			return fail()
		}
		digest := "sha256:" + sha
		if _, err := canonical.Raw(digest); err != nil {
			return fail()
		}
		if prior, found := sizes[digest]; found && prior != length {
			return fail()
		}
		sizes[digest], names[name] = length, true
		total += length
		members = append(members, TreeMember{Path: name, Digest: digest, Length: length})
	}
	if total != contentBytes {
		return fail()
	}
	for name := range names {
		for parent := path.Dir(name); parent != "."; parent = path.Dir(parent) {
			if names[parent] {
				return fail()
			}
		}
	}
	return members, nil
}
