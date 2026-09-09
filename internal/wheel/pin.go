package wheel

import (
	"archive/zip"
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/csv"
	"io"
	"net/textproto"
	"os"
	"path/filepath"
	"sort"
	"strconv"

	"github.com/cozy-creator/cozy/internal/exit"
)

// PinDependencies derives a private project wheel with exact flattened requirements.
// Every implementation/resource member keeps its original bytes. The new METADATA
// and RECORD participate in the ordinary wheel hash; no extra runtime manifest exists.
func PinDependencies(source, target string, requirements []string) *exit.Error {
	return pinDependencies(source, target, requirements, nil)
}

// PinDependenciesPreserving seals private requirements while retaining the
// project's original compatibility declarations for image-owned distributions.
// Those distributions are supplied by the worker image and must not be replaced
// by the versions selected on the publishing machine.
func PinDependenciesPreserving(source, target string, requirements []string,
	preserve func(string) bool,
) *exit.Error {
	return pinDependencies(source, target, requirements, preserve)
}

func pinDependencies(source, target string, requirements []string,
	preserve func(string) bool,
) *exit.Error {
	if _, problem := InspectIdentity(source); problem != nil {
		return problem
	}
	reader, err := zip.OpenReader(source)
	if err != nil {
		return wheelStructure("cannot read project wheel")
	}
	defer reader.Close()
	if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
		return exit.Internalf("cannot stage pinned project wheel")
	}
	file, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return exit.Internalf("cannot stage pinned project wheel")
	}
	defer file.Close()
	writer := zip.NewWriter(file)
	var record bytes.Buffer
	csvWriter := csv.NewWriter(&record)
	recordName := ""
	files := append([]*zip.File(nil), reader.File...)
	sort.Slice(files, func(i, j int) bool { return files[i].Name < files[j].Name })
	for _, member := range files {
		if distInfoMember(member.Name, "RECORD.jws") || distInfoMember(member.Name, "RECORD.p7s") {
			return wheelStructure("signed project metadata cannot be privately repinned")
		}
		if distInfoMember(member.Name, "RECORD") {
			recordName = member.Name
			continue
		}
		if member.FileInfo().IsDir() {
			continue
		}
		hash := sha256.New()
		var size int64
		if distInfoMember(member.Name, "METADATA") {
			body, problem := wheelMember(member)
			if problem != nil {
				return problem
			}
			body, err = pinnedMetadata(body, requirements, preserve)
			if err != nil {
				return wheelStructure("cannot seal private project requirements")
			}
			header := member.FileHeader
			header.CRC32, header.CompressedSize, header.UncompressedSize, header.CompressedSize64, header.UncompressedSize64 = 0, 0, 0, 0, 0
			out, createErr := writer.CreateHeader(&header)
			if createErr != nil {
				return exit.Internalf("cannot write private project metadata")
			}
			count, writeErr := out.Write(body)
			if writeErr != nil || count != len(body) {
				return exit.Internalf("cannot write private project metadata")
			}
			size = int64(len(body))
			hash.Write(body)
		} else {
			incoming, openErr := member.Open()
			if openErr != nil {
				return wheelStructure("cannot verify a project member")
			}
			size, err = io.Copy(hash, io.LimitReader(incoming, MaxWheelBytes+1))
			incoming.Close()
			if err != nil || size > MaxWheelBytes {
				return wheelStructure("project member exceeds the wheel bound")
			}
			if err = writer.Copy(member); err != nil {
				return exit.Internalf("cannot copy private project member")
			}
		}
		if err := csvWriter.Write([]string{member.Name, "sha256=" + base64.RawURLEncoding.EncodeToString(hash.Sum(nil)), strconv.FormatInt(size, 10)}); err != nil {
			return exit.Internalf("cannot record private project member")
		}
	}
	if recordName == "" {
		return wheelStructure("private project wheel has no RECORD")
	}
	_ = csvWriter.Write([]string{recordName, "", ""})
	csvWriter.Flush()
	if csvWriter.Error() != nil {
		return exit.Internalf("cannot seal private project RECORD")
	}
	out, err := writer.Create(recordName)
	if err != nil {
		return exit.Internalf("cannot write private project RECORD")
	}
	if _, err = out.Write(record.Bytes()); err != nil {
		return exit.Internalf("cannot write private project RECORD")
	}
	if err = writer.Close(); err != nil {
		return exit.Internalf("cannot close pinned project wheel")
	}
	if err = file.Sync(); err != nil {
		return exit.Internalf("cannot persist pinned project wheel")
	}
	if _, problem := InspectIdentity(target); problem != nil {
		return problem
	}
	return nil
}

func pinnedMetadata(raw []byte, requirements []string, preserve func(string) bool) ([]byte, error) {
	reader := bufio.NewReader(bytes.NewReader(raw))
	headers, err := textproto.NewReader(reader).ReadMIMEHeader()
	if err != nil && err != io.EOF {
		return nil, err
	}
	body, err := io.ReadAll(reader)
	if err != nil {
		return nil, err
	}
	originalRequirements := headers.Values("Requires-Dist")
	headers.Del("Requires-Dist")
	headers.Del("Provides-Extra")
	if preserve != nil {
		for _, requirement := range originalRequirements {
			if preserve(requirement) {
				headers.Add("Requires-Dist", requirement)
			}
		}
	}
	for _, requirement := range requirements {
		headers.Add("Requires-Dist", requirement)
	}
	keys := make([]string, 0, len(headers))
	for key := range headers {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var output bytes.Buffer
	for _, key := range keys {
		for _, value := range headers[key] {
			output.WriteString(key + ": " + value + "\n")
		}
	}
	output.WriteByte('\n')
	output.Write(body)
	return output.Bytes(), nil
}
