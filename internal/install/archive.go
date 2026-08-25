package install

import (
	"archive/tar"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/render"
)

// Bounds every staged release archive is read under. A hostile archive is refused
// while it is being read, never after it has landed.
const (
	MaxCompressed = 256 << 20 // archive bytes on the wire
	MaxExpanded   = 1 << 30   // bytes written to disk
	MaxFiles      = 20000
	MaxPathLen    = 1024
	MaxComponent  = 255
	DeclName      = "release.json"
)

// Declaration is the release record that travels with the archive: the file set the
// publisher declared, with a digest each. Pre-hub it is written by scripts/pack.py;
// cl-011/cl-012 replace it with the hub's own release record over the same fields.
type Declaration struct {
	Endpoint string          `json:"endpoint"`
	Version  string          `json:"version"`
	Files    []DeclaredEntry `json:"files"`
}

type DeclaredEntry struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

// Staged is the result of the stage step: bytes on disk, plus the evidence the
// verify step needs. No code from the archive has run at this point.
type Staged struct {
	Root       string
	Digest     string // sha256: over the archive bytes
	Decl       Declaration
	Files      int
	Bytes      int64
	Compressed int64
}

func refuse(name, format string, args ...any) *exit.Error {
	return exit.Named(exit.Validation, name, format, args...)
}

// countingReader enforces the compressed cap and digests the archive in one pass.
type countingReader struct {
	r   io.Reader
	n   int64
	sum interface {
		io.Writer
		Sum([]byte) []byte
	}
	err *exit.Error
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	if n > 0 {
		c.n += int64(n)
		c.sum.Write(p[:n])
		if c.n > MaxCompressed {
			c.err = refuse("archive_compressed_cap",
				"the release archive exceeds the %s compressed cap (read %s so far)",
				bytesText(MaxCompressed), bytesText(c.n)).
				WithRemedy("a release archive is bounded on the wire; publish a smaller release")
			return n, io.ErrUnexpectedEOF
		}
	}
	return n, err
}

// StageArchive extracts one .tar.gz release into dest under the bounds above. The
// declaration must be the archive's FIRST entry, so an undeclared entry is refused
// before it is written, not after.
func StageArchive(archivePath, dest string) (*Staged, *exit.Error) {
	f, err := os.Open(archivePath)
	if err != nil {
		return nil, exit.New(exit.NotFound, "cannot read the release archive %s: %s", archivePath, err).
			WithRemedy("--from names a .tar.gz release archive on this host")
	}
	defer f.Close()

	cr := &countingReader{r: f, sum: sha256.New()}
	zr, err := gzip.NewReader(cr)
	if err != nil {
		if cr.err != nil {
			return nil, cr.err
		}
		return nil, refuse("archive_unreadable", "%s is not a gzip release archive: %s", archivePath, err)
	}
	defer zr.Close()

	st := &Staged{Root: dest}
	tr := tar.NewReader(zr)
	seen := map[string]bool{}     // normalized path -> present
	folded := map[string]string{} // lowercased path -> the path that claimed it
	declared := map[string]DeclaredEntry{}
	haveDecl := false

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			if cr.err != nil {
				return nil, cr.err
			}
			return nil, refuse("archive_unreadable", "the release archive is truncated or malformed: %s", err)
		}

		clean, e := safeName(hdr.Name)
		if e != nil {
			return nil, e
		}
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(filepath.Join(dest, filepath.FromSlash(clean)), 0o755); err != nil {
				return nil, exit.Internalf("cannot create %s: %s", clean, err)
			}
			continue
		case tar.TypeReg:
		case tar.TypeSymlink, tar.TypeLink:
			return nil, refuse("archive_link",
				"the release archive carries a link entry %q -> %q", clean, hdr.Linkname).
				WithRemedy("a release archive carries regular files and directories only; links can escape the tree")
		default:
			return nil, refuse("archive_special_entry",
				"the release archive carries a non-regular entry %q (tar type %q)", clean, string(hdr.Typeflag)).
				WithRemedy("devices, fifos and sockets are never part of a release")
		}

		if seen[clean] {
			return nil, refuse("archive_duplicate_entry",
				"the release archive declares %q twice", clean).
				WithRemedy("a duplicate name makes the extracted tree depend on entry order")
		}
		if prev, ok := folded[strings.ToLower(clean)]; ok {
			return nil, refuse("archive_case_collision",
				"the release archive carries %q and %q, which collide on a case-insensitive filesystem", prev, clean).
				WithRemedy("rename one; the same archive must extract identically on Linux, macOS and Windows")
		}
		st.Files++
		if st.Files > MaxFiles {
			return nil, refuse("archive_too_many_files",
				"the release archive exceeds the %d file cap", MaxFiles).
				WithRemedy("a release is source, never a build tree or a weights bundle")
		}

		if clean == DeclName {
			if st.Files != 1 {
				return nil, refuse("archive_declaration_misplaced",
					"%s must be the first entry of a release archive, found at entry %d", DeclName, st.Files)
			}
			raw, err := io.ReadAll(io.LimitReader(tr, 16<<20))
			if err != nil {
				return nil, refuse("archive_unreadable", "cannot read %s: %s", DeclName, err)
			}
			if err := json.Unmarshal(raw, &st.Decl); err != nil {
				return nil, refuse("archive_declaration_malformed", "%s is not a release declaration: %s", DeclName, err)
			}
			for _, d := range st.Decl.Files {
				c, e := safeName(d.Path)
				if e != nil {
					return nil, e
				}
				declared[c] = d
			}
			haveDecl = true
			seen[clean] = true
			folded[strings.ToLower(clean)] = clean
			if err := os.MkdirAll(dest, 0o755); err != nil {
				return nil, exit.Internalf("cannot create %s: %s", dest, err)
			}
			if err := os.WriteFile(filepath.Join(dest, DeclName), raw, 0o644); err != nil {
				return nil, exit.Internalf("cannot write %s: %s", DeclName, err)
			}
			st.Bytes += int64(len(raw))
			continue
		}
		if !haveDecl {
			return nil, refuse("archive_missing_declaration",
				"the release archive's first entry is %q, not %s", clean, DeclName).
				WithRemedy("every release archive declares its file set first; nothing is extracted on faith")
		}
		want, ok := declared[clean]
		if !ok {
			return nil, refuse("archive_undeclared_entry",
				"the release archive carries %q, which %s does not declare", clean, DeclName).
				WithRemedy("the declared file set is the release; an undeclared entry is a smuggled one")
		}

		n, sum, e := writeEntry(tr, filepath.Join(dest, filepath.FromSlash(clean)), hdr.FileInfo().Mode(), MaxExpanded-st.Bytes)
		if e != nil {
			if cr.err != nil {
				return nil, cr.err
			}
			return nil, e
		}
		if sum != want.SHA256 {
			return nil, refuse("archive_file_digest_mismatch",
				"%q does not match the digest %s declares", clean, DeclName).
				WithRemedy("declared %s, extracted %s", short(want.SHA256), short(sum))
		}
		st.Bytes += n
		seen[clean] = true
		folded[strings.ToLower(clean)] = clean
	}

	if cr.err != nil {
		return nil, cr.err
	}
	if !haveDecl {
		return nil, refuse("archive_missing_declaration", "the release archive carries no %s", DeclName).
			WithRemedy("every release archive declares its file set first; nothing is extracted on faith")
	}
	for p := range declared {
		if !seen[p] {
			return nil, refuse("archive_incomplete",
				"%s declares %q but the archive does not carry it", DeclName, p).
				WithRemedy("the release is incomplete; re-publish it")
		}
	}
	// Drain so the digest covers the whole file, including any trailer.
	if _, err := io.Copy(io.Discard, cr); err != nil && cr.err != nil {
		return nil, cr.err
	}
	st.Compressed = cr.n
	st.Digest = "sha256:" + hex.EncodeToString(cr.sum.Sum(nil))
	return st, nil
}

func writeEntry(r io.Reader, dst string, mode os.FileMode, budget int64) (int64, string, *exit.Error) {
	if budget <= 0 {
		return 0, "", refuse("archive_expanded_cap",
			"the release archive expands past the %s cap", bytesText(MaxExpanded)).
			WithRemedy("a release is source; %s of expanded bytes is not one", bytesText(MaxExpanded))
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return 0, "", exit.Internalf("cannot create %s: %s", filepath.Dir(dst), err)
	}
	f, err := os.OpenFile(dst, os.O_CREATE|os.O_EXCL|os.O_WRONLY, mode.Perm()|0o200)
	if err != nil {
		return 0, "", exit.Internalf("cannot write %s: %s", dst, err)
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(f, h), io.LimitReader(r, budget+1))
	if err != nil {
		return 0, "", refuse("archive_unreadable", "cannot extract %s: %s", filepath.Base(dst), err)
	}
	if n > budget {
		return 0, "", refuse("archive_expanded_cap",
			"the release archive expands past the %s cap", bytesText(MaxExpanded)).
			WithRemedy("a release is source; %s of expanded bytes is not one", bytesText(MaxExpanded))
	}
	return n, hex.EncodeToString(h.Sum(nil)), nil
}

// safeName normalizes one archive path and refuses everything that could write
// outside the staging root or extract differently on another host.
func safeName(name string) (string, *exit.Error) {
	if name == "" {
		return "", refuse("archive_path_invalid", "the release archive carries an entry with no name")
	}
	if strings.ContainsRune(name, 0) || strings.Contains(name, `\`) {
		return "", refuse("archive_path_invalid",
			"the release archive carries an entry with an invalid path %q", name).
			WithRemedy("paths are slash-separated and carry no NUL")
	}
	if strings.HasPrefix(name, "/") || filepath.IsAbs(name) || strings.Contains(name, ":") {
		return "", refuse("archive_absolute_path",
			"the release archive carries an absolute path %q", name).
			WithRemedy("a release archive extracts under one root; absolute paths write outside it")
	}
	if len(name) > MaxPathLen {
		return "", refuse("archive_path_too_long",
			"the release archive carries a %d-byte path, past the %d cap: %q", len(name), MaxPathLen, short(name))
	}
	clean := path.Clean(strings.TrimSuffix(name, "/"))
	if clean == "." || clean == ".." || strings.HasPrefix(clean, "../") {
		return "", refuse("archive_traversal",
			"the release archive carries a traversal path %q", name).
			WithRemedy("`..` in a release path escapes the staging root")
	}
	for _, c := range strings.Split(clean, "/") {
		if c == ".." {
			return "", refuse("archive_traversal",
				"the release archive carries a traversal path %q", name).
				WithRemedy("`..` in a release path escapes the staging root")
		}
		if len(c) > MaxComponent {
			return "", refuse("archive_path_too_long",
				"the release archive carries a %d-byte path component, past the %d cap", len(c), MaxComponent)
		}
	}
	return clean, nil
}

func bytesText(n int64) string { return render.Bytes(n) }

func short(s string) string {
	if len(s) > 19 {
		return s[:19] + "…"
	}
	return s
}
