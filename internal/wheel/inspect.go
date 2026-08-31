package wheel

// Publication needs only a wheel's bounded name and version. Tensorhub owns
// project-wheel and package-environment policy after the uploaded bytes cross its trust boundary.

import (
	"archive/zip"
	"bufio"
	"bytes"
	"io"
	"net/textproto"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

const (
	MaxWheelBytes int64 = 512 << 20
	maxZipEntries       = 0xfffe
)

var (
	reName      = regexp.MustCompile(`^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$`)
	reNormalize = regexp.MustCompile(`[-_.]+`)
)

// Identity is the minimum fact Creator needs before offering a wheel to
// Tensorhub. It does not decide whether the wheel is safe or compatible.
type Identity struct {
	Distribution string
	Filename     string
	Length       int64
	Version      string
}

// InspectIdentity verifies that one bounded wheel's filename and METADATA name
// and version agree. Native and platform-tagged dependency wheels are admitted.
func InspectIdentity(file string) (Identity, *exit.Error) {
	var out Identity
	abs, err := filepath.Abs(file)
	if err != nil {
		return out, exit.Usagef("wheel path %q is not resolvable: %s", file, err)
	}
	info, err := os.Stat(abs)
	if err != nil || !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > MaxWheelBytes {
		return out, exit.Named(exit.Validation, "wheel_size_invalid",
			"%s is not a non-empty regular wheel at or below %d B", abs, MaxWheelBytes)
	}
	filename := filepath.Base(abs)
	distribution, version, problem := parseWheelFilename(filename)
	if problem != nil {
		return out, problem
	}
	f, err := os.Open(abs)
	if err != nil {
		return out, exit.Named(exit.Structural, "wheel_unreadable", "%s: %v", abs, err)
	}
	defer f.Close()
	zr, err := zip.NewReader(f, info.Size())
	if err != nil || len(zr.File) == 0 || len(zr.File) > maxZipEntries {
		return out, exit.Named(exit.Validation, "wheel_zip_invalid", "%s is not a bounded ZIP wheel", filename)
	}
	var metadata []byte
	for _, member := range zr.File {
		if !strings.HasSuffix(member.Name, ".dist-info/METADATA") {
			continue
		}
		if metadata != nil {
			return out, wheelStructure("more than one .dist-info/METADATA")
		}
		metadata, problem = wheelMember(member)
		if problem != nil {
			return out, problem
		}
	}
	if metadata == nil {
		return out, wheelStructure("one .dist-info/METADATA is required")
	}
	metadataName, metadataVersion, problem := metadataIdentity(metadata)
	if problem != nil {
		return out, problem
	}
	if normalize(metadataName) != distribution || metadataVersion != version {
		return out, exit.Named(exit.Validation, "wheel_identity_mismatch",
			"filename says %s==%s while METADATA says %s==%s",
			distribution, version, normalize(metadataName), metadataVersion)
	}
	out.Distribution, out.Filename = distribution, filename
	out.Length, out.Version = info.Size(), version
	return out, nil
}

func parseWheelFilename(filename string) (string, string, *exit.Error) {
	if filepath.Base(filename) != filename || !strings.HasSuffix(filename, ".whl") {
		return "", "", exit.Named(exit.Validation, "wheel_filename_invalid", "%q is not a wheel basename", filename)
	}
	parts := strings.Split(strings.TrimSuffix(filename, ".whl"), "-")
	if len(parts) != 5 && len(parts) != 6 {
		return "", "", exit.Named(exit.Validation, "wheel_filename_invalid",
			"%q does not have the PEP 427 distribution-version[-build]-python-abi-platform shape", filename)
	}
	distribution := normalize(strings.ReplaceAll(parts[0], "_", "-"))
	version := parts[1]
	if !reName.MatchString(distribution) || version == "" {
		return "", "", exit.Named(exit.Validation, "wheel_filename_invalid", "%q has an invalid distribution or version", filename)
	}
	tags := []string{}
	n := len(parts)
	for _, py := range strings.Split(parts[n-3], ".") {
		for _, abi := range strings.Split(parts[n-2], ".") {
			for _, platform := range strings.Split(parts[n-1], ".") {
				tags = append(tags, py+"-"+abi+"-"+platform)
			}
		}
	}
	if len(sortedUnique(tags)) == 0 {
		return "", "", exit.Named(exit.Validation, "wheel_filename_invalid", "%q names no wheel tag", filename)
	}
	return distribution, version, nil
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
	header, err := textproto.NewReader(bufio.NewReader(bytes.NewReader(body))).ReadMIMEHeader()
	if err != nil && err != io.EOF {
		return "", "", wheelStructure("METADATA header block is malformed")
	}
	names, versions := header.Values("Name"), header.Values("Version")
	if len(names) > 1 {
		return "", "", wheelStructure("METADATA repeats Name")
	}
	if len(versions) > 1 {
		return "", "", wheelStructure("METADATA repeats Version")
	}
	name, version := "", ""
	if len(names) == 1 {
		name = strings.TrimSpace(names[0])
	}
	if len(versions) == 1 {
		version = strings.TrimSpace(versions[0])
	}
	if name == "" || version == "" {
		return "", "", wheelStructure("METADATA has no exact Name and Version")
	}
	return name, version, nil
}

func normalize(name string) string {
	return strings.ToLower(reNormalize.ReplaceAllString(name, "-"))
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
