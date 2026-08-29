// Package wheel contains Cozy's fixed pure-wheel helper and exact wheel inspector.
//
// Pack is a fixed first-party helper for controlled local materialization and fixtures. Public
// `cozy package publish` instead runs `uv build --wheel` in private staging and passes the one
// result through Inspect. Pack does not consult `[build-system]`, import the project, or fire a
// PEP 517 hook. It copies the canonical path-sorted source tree,
// synthesizes METADATA/WHEEL/RECORD from declared metadata under fixed field order, fixed
// timestamps and normalized permissions, and emits a py3-none-any wheel.
//
// Inputs: the canonical tree's path (a directory of regular files), plus the
// distribution identity — from `[project]` when the tree declares it, otherwise from the
// caller, because a package's identity is its RELEASE, not a line in its tree.
// Outputs: the wheel file, its `project_wheel_digest`, and the `tree_digest` it was
// computed from. Refusals are typed and named; each one names the door it closed.
//
// DETERMINISM IS THE GATE, and it is structural, not hoped for. The wheel bytes are a
// function of (canonical tree, distribution identity, packer version) and of NOTHING
// else — no clock, no locale, no timezone, no umask, no host, no working directory, no
// map iteration order, and no compressor. The zip container is written by hand, byte by
// byte, rather than through a library whose framing choices could shift under a toolchain
// upgrade: two seats that must produce the same digest do not share a build of Go.
//
// Two consequences of that, stated rather than hidden:
//   - Entries are STORED, never deflated. A compressor's output is an implementation
//     detail of whoever compiled the packer; the digest would then be a function of the
//     toolchain, and the hub/laptop equality gate (§4) would fail for no semantic reason.
//     The wheel rides in CAS and over compressing transports, so the bytes are paid for
//     once, in a place that dedups them.
//   - File modes are NORMALIZED to 0644 and the execute bit is not carried. A wheel is
//     imported, not run; the canonical tree normalizes modes at intake for the same
//     reason (§1 stage 1).
package wheel

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/exit"
)

const (
	// PackerVersion is part of the digest's definition. Anything that changes the bytes
	// this package emits for an unchanged tree changes this string, in the same commit.
	PackerVersion = "cozy-wheel 1"
	// Tag is fixed: pure-python at launch. A tree that would need another tag is refused
	// by the compiled-extension arm, never quietly retagged.
	Tag = "py3-none-any"

	// The DOS timestamp every entry carries: 1980-01-01 00:00:00, the zero of the zip
	// epoch. No entry carries a clock reading.
	fixedDOSTime = 0x0000
	fixedDOSDate = 0x0021

	zipVersionNeeded = 20        // 2.0 — stored/deflated, no zip64
	zipVersionMadeBy = 3<<8 | 20 // unix creator, 2.0
	fixedFileMode    = 0o644
	maxZipEntries    = 0xfffe
	maxZipBytes      = 0xfffffffe // zip64 is refused rather than emitted
)

var reEscapeVersion = regexp.MustCompile(`[^\w\d.+]+`)

// Request is what a caller must settle before a wheel exists.
type Request struct {
	// Tree is the canonical source tree's root — the build input and identity source.
	Tree string
	// Name and Version are the distribution identity. They are REQUIRED unless the tree
	// declares them in `[project]`; when both exist they must agree.
	Name    string
	Version string
	// OutDir is where the wheel is written. The filename is derived, never supplied: a
	// wheel's name is part of its meaning to pip.
	OutDir string
}

// Result is what both callers record. `Digest` is `project_wheel_digest`.
type Result struct {
	Path          string
	Filename      string
	Digest        string
	TreeDigest    string
	Bytes         int64
	Files         int
	Name          string
	Version       string
	Tag           string
	PackerVersion string
	Entries       []string
	Fact          Fact
}

// Pack turns a canonical source tree into one wheel. Refusal order is fixed and part of
// the contract: declared metadata, then the build-system door, then the tree itself, then
// identity, then the package's own application contract. A tree that trips several arms
// always reports the same one.
func Pack(req Request) (*Result, *exit.Error) {
	root, err := filepath.Abs(req.Tree)
	if err != nil {
		return nil, exit.Named(exit.Usage, "tree_missing", "%s: %v", req.Tree, err)
	}

	decl, e := readDeclaration(root)
	if e != nil {
		return nil, e
	}
	if e := checkBackend(decl); e != nil {
		return nil, e
	}
	entries, e := walk(root)
	if e != nil {
		return nil, e
	}
	name, version, e := identity(decl, req)
	if e != nil {
		return nil, e
	}
	if e := checkApplication(decl, entries); e != nil {
		return nil, e
	}

	distInfo := escape(name) + "-" + reEscapeVersion.ReplaceAllString(version, "_") + ".dist-info"
	meta, e := metadataBody(name, version, decl)
	if e != nil {
		return nil, e
	}

	// Entry order is FIXED: the payload path-sorted, then the dist-info in the order a
	// reader needs it, with RECORD last because it describes everything before it.
	members := make([]entry, 0, len(entries)+3)
	members = append(members, entries...)
	members = append(members,
		entry{Path: distInfo + "/METADATA", Data: meta},
		entry{Path: distInfo + "/WHEEL", Data: wheelBody()},
	)
	members = append(members, entry{Path: distInfo + "/RECORD", Data: record(members, distInfo)})

	body, e := writeZip(members)
	if e != nil {
		return nil, e
	}
	if e := verify(body, members); e != nil {
		return nil, e
	}

	filename := escape(name) + "-" + reEscapeVersion.ReplaceAllString(version, "_") + "-" + Tag + ".whl"
	outDir := req.OutDir
	if outDir == "" {
		outDir = "."
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return nil, exit.Named(exit.Structural, "output_unwritable", "%s: %v", outDir, err)
	}
	out := filepath.Join(outDir, filename)
	if err := os.WriteFile(out, body, 0o644); err != nil {
		return nil, exit.Named(exit.Structural, "output_unwritable", "%s: %v", out, err)
	}
	fact, e := Inspect(out, ProjectWheel)
	if e != nil {
		_ = os.Remove(out)
		return nil, e
	}

	sum := sha256.Sum256(body)
	names := make([]string, len(members))
	for i, m := range members {
		names[i] = m.Path
	}
	return &Result{
		Path:          out,
		Filename:      filename,
		Digest:        "sha256:" + hex.EncodeToString(sum[:]),
		TreeDigest:    treeDigest(entries),
		Bytes:         int64(len(body)),
		Files:         len(entries),
		Name:          name,
		Version:       version,
		Tag:           Tag,
		PackerVersion: PackerVersion,
		Entries:       names,
		Fact:          fact,
	}, nil
}

// record is the RECORD file: every member, path-sorted, with its digest and size. RECORD
// names itself with neither, because a file cannot carry its own hash.
func record(members []entry, distInfo string) []byte {
	rows := make([]string, 0, len(members)+1)
	for _, m := range members {
		sum := sha256.Sum256(m.Data)
		rows = append(rows, fmt.Sprintf("%s,sha256=%s,%d",
			m.Path, base64.RawURLEncoding.EncodeToString(sum[:]), len(m.Data)))
	}
	sort.Strings(rows)
	rows = append(rows, distInfo+"/RECORD,,")
	return []byte(strings.Join(rows, "\n") + "\n")
}

// writeZip emits the container by hand: stored entries, no data descriptors, no extra
// fields, no comments, one fixed timestamp and one fixed mode. Every field below is a
// constant or a function of the member bytes.
func writeZip(members []entry) ([]byte, *exit.Error) {
	if len(members) > maxZipEntries {
		return nil, exit.Named(exit.Validation, "wheel_too_large",
			"%d entries exceeds the %d this packer emits without zip64", len(members), maxZipEntries)
	}
	var buf bytes.Buffer
	type placed struct {
		e      entry
		crc    uint32
		offset uint32
	}
	local := make([]placed, 0, len(members))

	for _, m := range members {
		if len(m.Data) > maxZipBytes || buf.Len() > maxZipBytes {
			return nil, exit.Named(exit.Validation, "wheel_too_large",
				"%s crosses the 4 GiB boundary; this packer refuses rather than emitting zip64", m.Path)
		}
		off := uint32(buf.Len())
		crc := crc32.ChecksumIEEE(m.Data)

		h := make([]byte, 30)
		binary.LittleEndian.PutUint32(h[0:], 0x04034b50)
		binary.LittleEndian.PutUint16(h[4:], zipVersionNeeded)
		binary.LittleEndian.PutUint16(h[6:], 0) // flags: no data descriptor, no utf8 bit
		binary.LittleEndian.PutUint16(h[8:], 0) // method: stored
		binary.LittleEndian.PutUint16(h[10:], fixedDOSTime)
		binary.LittleEndian.PutUint16(h[12:], fixedDOSDate)
		binary.LittleEndian.PutUint32(h[14:], crc)
		binary.LittleEndian.PutUint32(h[18:], uint32(len(m.Data)))
		binary.LittleEndian.PutUint32(h[22:], uint32(len(m.Data)))
		binary.LittleEndian.PutUint16(h[26:], uint16(len(m.Path)))
		binary.LittleEndian.PutUint16(h[28:], 0) // extra: none
		buf.Write(h)
		buf.WriteString(m.Path)
		buf.Write(m.Data)

		local = append(local, placed{e: m, crc: crc, offset: off})
	}

	dirStart := uint32(buf.Len())
	for _, p := range local {
		h := make([]byte, 46)
		binary.LittleEndian.PutUint32(h[0:], 0x02014b50)
		binary.LittleEndian.PutUint16(h[4:], zipVersionMadeBy)
		binary.LittleEndian.PutUint16(h[6:], zipVersionNeeded)
		binary.LittleEndian.PutUint16(h[8:], 0)
		binary.LittleEndian.PutUint16(h[10:], 0)
		binary.LittleEndian.PutUint16(h[12:], fixedDOSTime)
		binary.LittleEndian.PutUint16(h[14:], fixedDOSDate)
		binary.LittleEndian.PutUint32(h[16:], p.crc)
		binary.LittleEndian.PutUint32(h[20:], uint32(len(p.e.Data)))
		binary.LittleEndian.PutUint32(h[24:], uint32(len(p.e.Data)))
		binary.LittleEndian.PutUint16(h[28:], uint16(len(p.e.Path)))
		binary.LittleEndian.PutUint16(h[30:], 0) // extra
		binary.LittleEndian.PutUint16(h[32:], 0) // comment
		binary.LittleEndian.PutUint16(h[34:], 0) // disk
		binary.LittleEndian.PutUint16(h[36:], 0) // internal attrs
		binary.LittleEndian.PutUint32(h[38:], fixedFileMode<<16)
		binary.LittleEndian.PutUint32(h[42:], p.offset)
		buf.Write(h)
		buf.WriteString(p.e.Path)
	}
	dirSize := uint32(buf.Len()) - dirStart

	end := make([]byte, 22)
	binary.LittleEndian.PutUint32(end[0:], 0x06054b50)
	binary.LittleEndian.PutUint16(end[4:], 0)
	binary.LittleEndian.PutUint16(end[6:], 0)
	binary.LittleEndian.PutUint16(end[8:], uint16(len(local)))
	binary.LittleEndian.PutUint16(end[10:], uint16(len(local)))
	binary.LittleEndian.PutUint32(end[12:], dirSize)
	binary.LittleEndian.PutUint32(end[16:], dirStart)
	binary.LittleEndian.PutUint16(end[20:], 0) // no archive comment
	buf.Write(end)

	return buf.Bytes(), nil
}

// verify reads the emitted bytes back through a general zip reader and checks the two
// properties the packer claims but a hand-written writer could quietly break: every name
// is relative and normalized, and every payload is exactly what went in. A wheel that
// fails this is never written.
func verify(body []byte, members []entry) *exit.Error {
	internal := func(format string, args ...any) *exit.Error {
		return exit.Named(exit.Internal, "packer_self_check", format, args...)
	}
	r, err := zip.NewReader(bytes.NewReader(body), int64(len(body)))
	if err != nil {
		return internal("the emitted wheel does not read back as a zip: %v", err)
	}
	if len(r.File) != len(members) {
		return internal("wrote %d entries, read back %d", len(members), len(r.File))
	}
	for i, f := range r.File {
		if e := checkName(f.Name); e != nil {
			return internal("entry %q is not a safe wheel path: %s", f.Name, e.Message)
		}
		if f.Name != members[i].Path {
			return internal("entry %d is %q, expected %q", i, f.Name, members[i].Path)
		}
		rc, err := f.Open()
		if err != nil {
			return internal("entry %q does not open: %v", f.Name, err)
		}
		got := sha256.New()
		_, err = io.Copy(got, rc)
		rc.Close()
		if err != nil {
			return internal("entry %q does not read: %v", f.Name, err)
		}
		want := sha256.Sum256(members[i].Data)
		if !bytes.Equal(got.Sum(nil), want[:]) {
			return internal("entry %q reads back with different bytes", f.Name)
		}
	}
	return nil
}
