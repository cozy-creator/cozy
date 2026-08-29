package wheel

// Exact wheel inspection for endpoint publication. Creator never builds, repairs,
// renames, or retags a custom wheel: it reads the bytes the author named, validates
// the portable wheel envelope and RECORD, and declares the resulting WheelFact.

import (
	"archive/zip"
	"crypto/sha256"
	"encoding/base64"
	"encoding/csv"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/exit"
)

const MaxWheelBytes int64 = 512 << 20

// Fact is the exact portable WheelFact wire shape frozen by th-075.
type Fact struct {
	Digest       string   `json:"digest"`
	Distribution string   `json:"distribution"`
	Filename     string   `json:"filename"`
	ImportRoots  []string `json:"import_roots"`
	Length       int64    `json:"length"`
	Tags         []string `json:"tags"`
	Version      string   `json:"version"`
	Native       bool     `json:"-"`
}

type InspectClass string

const (
	ProjectWheel InspectClass = "project"
	CustomWheel  InspectClass = "custom"
)

// Inspect reads one wheel exactly as supplied. Project wheels must be one pure
// py3-none-any payload with no native/build member. Custom wheels may contain native
// binaries but never native source or a build recipe; they remain prebuilt inputs.
func Inspect(file string, class InspectClass) (Fact, *exit.Error) {
	out := Fact{ImportRoots: []string{}, Tags: []string{}}
	abs, err := filepath.Abs(file)
	if err != nil {
		return out, exit.Usagef("wheel path %q is not resolvable: %s", file, err)
	}
	info, err := os.Stat(abs)
	if err != nil {
		return out, exit.Named(exit.NotFound, "wheel_absent", "%s is not readable: %v", abs, err)
	}
	if !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > MaxWheelBytes {
		return out, exit.Named(exit.Validation, "wheel_size_invalid",
			"%s is not a non-empty regular wheel at or below %d B", abs, MaxWheelBytes)
	}
	filename := filepath.Base(abs)
	distribution, version, filenameTags, e := parseWheelFilename(filename)
	if e != nil {
		return out, e
	}

	f, err := os.Open(abs)
	if err != nil {
		return out, exit.Named(exit.Structural, "wheel_unreadable", "%s: %v", abs, err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return out, exit.Named(exit.Structural, "wheel_unreadable", "%s: %v", abs, err)
	}
	zr, err := zip.NewReader(f, info.Size())
	if err != nil {
		return out, exit.Named(exit.Validation, "wheel_zip_invalid", "%s is not a ZIP wheel: %v", filename, err)
	}
	if len(zr.File) == 0 || len(zr.File) > maxZipEntries {
		return out, exit.Named(exit.Validation, "wheel_zip_invalid", "%s has %d ZIP entries", filename, len(zr.File))
	}

	seen := map[string]bool{}
	members := map[string][]byte{}
	var wheelDoc, metadataDoc, recordDoc []byte
	var distInfo string
	rootCandidates := map[string]bool{}
	for _, member := range zr.File {
		name := member.Name
		if e := checkName(strings.TrimSuffix(name, "/")); e != nil {
			return out, exit.Named(exit.Validation, "wheel_entry_invalid", "%s: %s", name, e.Message)
		}
		folded := strings.ToLower(name)
		if seen[folded] {
			return out, exit.Named(exit.Validation, "wheel_entry_invalid", "%s is duplicated or differs only by case", name)
		}
		seen[folded] = true
		if member.FileInfo().IsDir() {
			continue
		}
		if member.Mode()&os.ModeSymlink != 0 || !member.Mode().IsRegular() {
			return out, exit.Named(exit.Validation, "wheel_entry_invalid", "%s is not a regular file", name)
		}
		ext := strings.ToLower(path.Ext(name))
		if class == ProjectWheel && compiledExt[ext] {
			return out, refuseCompiled(name)
		}
		if class == CustomWheel && nativeSourceExt[ext] {
			return out, exit.Named(exit.Validation, "custom_wheel_build_input",
				"%s contains native source; custom wheels are exact prebuilt binaries", name)
		}
		if class == CustomWheel && compiledExt[ext] && !nativeSourceExt[ext] {
			out.Native = true
		}
		if buildInputName(name) {
			code := "project_wheel_build_input"
			if class == CustomWheel {
				code = "custom_wheel_build_input"
			}
			return out, exit.Named(exit.Validation, code,
				"%s contains a build input; Creator never builds or repairs endpoint wheels", name)
		}
		if ext == ".pth" {
			return out, exit.Named(exit.Validation, "wheel_path_injection",
				"%s is executable site-path injection and has no closed import-root fact", name)
		}

		body, e := wheelMember(member)
		if e != nil {
			return out, e
		}
		members[name] = body
		switch {
		case strings.HasSuffix(name, ".dist-info/WHEEL"):
			if wheelDoc != nil {
				return out, wheelStructure("more than one .dist-info/WHEEL")
			}
			wheelDoc, distInfo = body, strings.TrimSuffix(name, "/WHEEL")
		case strings.HasSuffix(name, ".dist-info/METADATA"):
			if metadataDoc != nil {
				return out, wheelStructure("more than one .dist-info/METADATA")
			}
			metadataDoc = body
		case strings.HasSuffix(name, ".dist-info/RECORD"):
			if recordDoc != nil {
				return out, wheelStructure("more than one .dist-info/RECORD")
			}
			recordDoc = body
		}
		root := importRoot(name)
		if root != "" {
			rootCandidates[root] = true
		}
	}
	if wheelDoc == nil || metadataDoc == nil || recordDoc == nil || distInfo == "" {
		return out, wheelStructure("WHEEL, METADATA, and RECORD must each appear exactly once")
	}
	if !strings.HasPrefix(path.Base(distInfo), strings.ReplaceAll(distribution, "-", "_")+"-") {
		return out, wheelStructure(".dist-info directory disagrees with filename distribution")
	}
	metadataName, metadataVersion, e := metadataIdentity(metadataDoc)
	if e != nil {
		return out, e
	}
	if normalize(metadataName) != distribution || metadataVersion != version {
		return out, exit.Named(exit.Validation, "wheel_identity_mismatch",
			"filename says %s==%s while METADATA says %s==%s",
			distribution, version, normalize(metadataName), metadataVersion)
	}
	wheelTags, pure, e := wheelHeaders(wheelDoc)
	if e != nil {
		return out, e
	}
	if !equalStrings(filenameTags, wheelTags) {
		return out, exit.Named(exit.Validation, "wheel_tag_mismatch",
			"filename tags %v disagree with WHEEL tags %v", filenameTags, wheelTags)
	}
	if class == ProjectWheel && (!pure || !equalStrings(wheelTags, []string{Tag})) {
		return out, exit.Named(exit.Validation, "project_wheel_not_pure",
			"project wheel must declare Root-Is-Purelib: true and exactly Tag: %s", Tag)
	}
	if e := verifyRecord(recordDoc, members, distInfo+"/RECORD"); e != nil {
		return out, e
	}

	for root := range rootCandidates {
		out.ImportRoots = append(out.ImportRoots, root)
	}
	sort.Strings(out.ImportRoots)
	out.Digest = "sha256:" + hex.EncodeToString(h.Sum(nil))
	out.Distribution, out.Filename, out.Length = distribution, filename, info.Size()
	out.Tags, out.Version = wheelTags, version
	return out, nil
}

var nativeSourceExt = map[string]bool{
	".c": true, ".h": true, ".cc": true, ".cpp": true, ".cxx": true, ".hpp": true,
	".pyx": true, ".pxd": true, ".pxi": true, ".rs": true, ".f90": true, ".cu": true,
}

func parseWheelFilename(filename string) (string, string, []string, *exit.Error) {
	if filepath.Base(filename) != filename || !strings.HasSuffix(filename, ".whl") {
		return "", "", nil, exit.Named(exit.Validation, "wheel_filename_invalid", "%q is not a wheel basename", filename)
	}
	parts := strings.Split(strings.TrimSuffix(filename, ".whl"), "-")
	if len(parts) != 5 && len(parts) != 6 {
		return "", "", nil, exit.Named(exit.Validation, "wheel_filename_invalid",
			"%q does not have the PEP 427 distribution-version[-build]-python-abi-platform shape", filename)
	}
	distribution := normalize(strings.ReplaceAll(parts[0], "_", "-"))
	version := parts[1]
	if !reName.MatchString(distribution) || version == "" {
		return "", "", nil, exit.Named(exit.Validation, "wheel_filename_invalid", "%q has an invalid distribution or version", filename)
	}
	n := len(parts)
	var tags []string
	for _, py := range strings.Split(parts[n-3], ".") {
		for _, abi := range strings.Split(parts[n-2], ".") {
			for _, platform := range strings.Split(parts[n-1], ".") {
				tags = append(tags, py+"-"+abi+"-"+platform)
			}
		}
	}
	tags = sortedUnique(tags)
	if len(tags) == 0 {
		return "", "", nil, exit.Named(exit.Validation, "wheel_filename_invalid", "%q names no wheel tag", filename)
	}
	return distribution, version, tags, nil
}

func wheelMember(member *zip.File) ([]byte, *exit.Error) {
	if member.UncompressedSize64 > uint64(MaxWheelBytes) {
		return nil, exit.Named(exit.Validation, "wheel_entry_too_large", "%s expands past %d B", member.Name, MaxWheelBytes)
	}
	r, err := member.Open()
	if err != nil {
		return nil, exit.Named(exit.Validation, "wheel_zip_invalid", "%s does not open: %v", member.Name, err)
	}
	defer r.Close()
	body, err := io.ReadAll(io.LimitReader(r, MaxWheelBytes+1))
	if err != nil || int64(len(body)) > MaxWheelBytes {
		return nil, exit.Named(exit.Validation, "wheel_entry_too_large", "%s cannot be read under the wheel bound", member.Name)
	}
	return body, nil
}

func wheelStructure(message string) *exit.Error {
	return exit.Named(exit.Validation, "wheel_structure_invalid", "%s", message)
}

func metadataIdentity(body []byte) (string, string, *exit.Error) {
	name, version := "", ""
	for _, line := range strings.Split(strings.ReplaceAll(string(body), "\r\n", "\n"), "\n") {
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		switch strings.ToLower(strings.TrimSpace(key)) {
		case "name":
			if name != "" {
				return "", "", wheelStructure("METADATA repeats Name")
			}
			name = strings.TrimSpace(value)
		case "version":
			if version != "" {
				return "", "", wheelStructure("METADATA repeats Version")
			}
			version = strings.TrimSpace(value)
		}
	}
	if name == "" || version == "" {
		return "", "", wheelStructure("METADATA has no exact Name and Version")
	}
	return name, version, nil
}

func wheelHeaders(body []byte) ([]string, bool, *exit.Error) {
	var tags []string
	pure := false
	pureSeen := false
	for _, line := range strings.Split(strings.ReplaceAll(string(body), "\r\n", "\n"), "\n") {
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		switch strings.ToLower(strings.TrimSpace(key)) {
		case "tag":
			value = strings.TrimSpace(value)
			if len(strings.Split(value, "-")) != 3 {
				return nil, false, exit.Named(exit.Validation, "wheel_tag_invalid", "%q is not a PEP 425 tag", value)
			}
			tags = append(tags, value)
		case "root-is-purelib":
			if pureSeen {
				return nil, false, wheelStructure("WHEEL repeats Root-Is-Purelib")
			}
			pureSeen = true
			pure = strings.EqualFold(strings.TrimSpace(value), "true")
		}
	}
	if !pureSeen || len(tags) == 0 {
		return nil, false, wheelStructure("WHEEL has no Root-Is-Purelib or Tag")
	}
	return sortedUnique(tags), pure, nil
}

func verifyRecord(body []byte, members map[string][]byte, recordName string) *exit.Error {
	reader := csv.NewReader(strings.NewReader(string(body)))
	reader.FieldsPerRecord = -1
	seen := map[string]bool{}
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil || len(row) != 3 || row[0] == "" || seen[row[0]] {
			return exit.Named(exit.Validation, "wheel_record_invalid", "RECORD is malformed, duplicated, or incomplete")
		}
		seen[row[0]] = true
		member, ok := members[row[0]]
		if !ok {
			return exit.Named(exit.Validation, "wheel_record_invalid", "RECORD names absent member %s", row[0])
		}
		if row[0] == recordName {
			if row[1] != "" || row[2] != "" {
				return exit.Named(exit.Validation, "wheel_record_invalid", "RECORD gives itself a digest or length")
			}
			continue
		}
		length, err := strconv.ParseInt(row[2], 10, 64)
		sum := sha256.Sum256(member)
		want := "sha256=" + base64.RawURLEncoding.EncodeToString(sum[:])
		if err != nil || length != int64(len(member)) || row[1] != want {
			return exit.Named(exit.Validation, "wheel_record_invalid", "RECORD identity disagrees for %s", row[0])
		}
	}
	if len(seen) != len(members) || !seen[recordName] {
		return exit.Named(exit.Validation, "wheel_record_invalid",
			"RECORD covers %d of %d wheel members", len(seen), len(members))
	}
	return nil
}

func validImportRoot(root string) bool {
	if root == "" || root == "." || strings.Contains(root, ".") {
		return false
	}
	for i, r := range root {
		if !(r == '_' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || i > 0 && r >= '0' && r <= '9') {
			return false
		}
	}
	return true
}

func importRoot(name string) string {
	parts := strings.Split(name, "/")
	if len(parts) >= 3 && strings.HasSuffix(parts[0], ".data") &&
		(parts[1] == "purelib" || parts[1] == "platlib") {
		if validImportRoot(parts[2]) {
			return parts[2]
		}
		return ""
	}
	if len(parts) > 1 {
		if validImportRoot(parts[0]) && !strings.HasSuffix(parts[0], ".dist-info") &&
			!strings.HasSuffix(parts[0], ".data") {
			return parts[0]
		}
		return ""
	}
	base := parts[0]
	if strings.HasSuffix(base, ".py") {
		base = strings.TrimSuffix(base, ".py")
	} else if ext := strings.ToLower(path.Ext(base)); compiledExt[ext] {
		base = strings.SplitN(base, ".", 2)[0]
	} else {
		return ""
	}
	if validImportRoot(base) {
		return base
	}
	return ""
}

func sortedUnique(values []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		if value != "" && !seen[value] {
			seen[value] = true
			out = append(out, value)
		}
	}
	sort.Strings(out)
	return out
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (f Fact) String() string {
	return fmt.Sprintf("%s==%s %s %dB", f.Distribution, f.Version, f.Digest, f.Length)
}
