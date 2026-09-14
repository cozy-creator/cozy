package inputasset

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
)

const MaxRootInputBytes = int64(256 << 20)

// CaptureTree freezes a bounded file closure into request-owned staging. Only
// regular files enter the canonical native manifest; source paths never reach Runtime.
func CaptureTree(destination, field, directory string, maximum int64) (records.AssetBinding, *exit.Error) {
	info, err := os.Lstat(directory)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return records.AssetBinding{}, exit.New(exit.Validation, "input tree must be a real directory")
	}
	relative, err := filepath.Rel(directory, destination)
	if err != nil || relative == "." || filepath.IsLocal(relative) {
		return records.AssetBinding{}, exit.New(exit.Validation, "input capture cannot be inside its source tree")
	}
	root, err := os.OpenRoot(directory)
	if err != nil {
		return records.AssetBinding{}, exit.Internalf("cannot open input tree: %s", err)
	}
	defer root.Close()
	var names []string
	original := map[string]fs.FileInfo{}
	err = fs.WalkDir(root.FS(), ".", func(name string, entry fs.DirEntry, problem error) error {
		if problem != nil {
			return problem
		}
		info, problem := entry.Info()
		if problem != nil {
			return problem
		}
		if !info.IsDir() && !info.Mode().IsRegular() {
			return fs.ErrInvalid
		}
		original[name] = info
		if info.Mode().IsRegular() {
			names = append(names, name)
		}
		return nil
	})
	if err != nil || len(names) == 0 {
		return records.AssetBinding{}, exit.New(exit.Validation, "input tree must contain regular files only")
	}
	sort.Strings(names)
	members := make([]resultfiles.TreeMember, 0, len(names))
	var total int64
	for _, name := range names {
		input, err := root.Open(filepath.FromSlash(name))
		if err != nil {
			return records.AssetBinding{}, exit.Internalf("cannot open input tree member: %s", err)
		}
		member, problem := captureFile(destination, input, name, min(maximum, MaxRootInputBytes)-total)
		input.Close()
		if problem != nil {
			return records.AssetBinding{}, problem
		}
		current, err := root.Lstat(filepath.FromSlash(name))
		before := original[name]
		if err != nil || !os.SameFile(before, current) || current.Size() != before.Size() || !current.ModTime().Equal(before.ModTime()) {
			return records.AssetBinding{}, exit.New(exit.Conflict, "input tree changed during capture")
		}
		total += member.Length
		members = append(members, member)
	}
	count := 0
	err = fs.WalkDir(root.FS(), ".", func(name string, entry fs.DirEntry, problem error) error {
		if problem != nil {
			return problem
		}
		current, problem := entry.Info()
		if problem != nil {
			return problem
		}
		before, ok := original[name]
		if !ok || !os.SameFile(before, current) || current.Mode() != before.Mode() || current.Size() != before.Size() || !current.ModTime().Equal(before.ModTime()) {
			return fs.ErrInvalid
		}
		count++
		return nil
	})
	if err != nil || count != len(original) {
		return records.AssetBinding{}, exit.New(exit.Conflict, "input tree changed its captured closure")
	}
	snapshot, problem := finishSnapshot(destination, members, total)
	if problem != nil {
		return records.AssetBinding{}, problem
	}
	return records.AssetBinding{FieldPath: field, LocalPath: snapshot.Path, Digest: snapshot.Manifest.Digest, Length: snapshot.Manifest.Length, MediaType: resultfiles.TreeMediaType, MaxBytes: maximum, Snapshot: snapshot}, nil
}

func CaptureFile(destination string, binding records.AssetBinding) (records.AssetBinding, *exit.Error) {
	info, err := os.Lstat(binding.LocalPath)
	if err != nil || !info.Mode().IsRegular() {
		return binding, exit.New(exit.Validation, "input file must be regular and cannot be a symlink")
	}
	input, err := os.Open(binding.LocalPath)
	if err != nil {
		return binding, exit.Internalf("cannot capture input file: %s", err)
	}
	defer input.Close()
	member, problem := captureFile(destination, input, "payload", min(binding.MaxBytes, MaxRootInputBytes))
	if problem != nil {
		return binding, problem
	}
	if member.Digest != binding.Digest || member.Length != binding.Length {
		return binding, exit.New(exit.Conflict, "input file changed before capture")
	}
	snapshot, problem := finishSnapshot(destination, []resultfiles.TreeMember{member}, member.Length)
	if problem != nil {
		return binding, problem
	}
	binding.Snapshot = snapshot
	binding.LocalPath = filepath.Join(snapshot.Path+".files", strings.TrimPrefix(member.Digest, "sha256:"))
	return binding, nil
}

func captureFile(destination string, input *os.File, name string, maximum int64) (resultfiles.TreeMember, *exit.Error) {
	info, err := input.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() < 0 || info.Size() > maximum {
		return resultfiles.TreeMember{}, exit.New(exit.Validation, "input exceeds its bounded file capacity")
	}
	directory := destination + ".files"
	if err := os.MkdirAll(directory, 0700); err != nil {
		return resultfiles.TreeMember{}, exit.Internalf("cannot create input snapshot: %s", err)
	}
	output, err := os.CreateTemp(directory, ".capture-")
	if err != nil {
		return resultfiles.TreeMember{}, exit.Internalf("cannot stage input snapshot: %s", err)
	}
	temporary := output.Name()
	defer func() { output.Close(); _ = os.Remove(temporary) }()
	hash := sha256.New()
	written, err := io.Copy(io.MultiWriter(output, hash), io.LimitReader(input, maximum+1))
	current, statErr := input.Stat()
	if err != nil || statErr != nil || written != info.Size() || current.Size() != info.Size() || !current.ModTime().Equal(info.ModTime()) {
		return resultfiles.TreeMember{}, exit.New(exit.Conflict, "input changed while its bytes were captured")
	}
	if err := output.Sync(); err != nil {
		return resultfiles.TreeMember{}, exit.Internalf("cannot sync captured input: %s", err)
	}
	if err := output.Close(); err != nil {
		return resultfiles.TreeMember{}, exit.Internalf("cannot close captured input: %s", err)
	}
	digest := "sha256:" + hex.EncodeToString(hash.Sum(nil))
	if err := os.Rename(temporary, filepath.Join(directory, strings.TrimPrefix(digest, "sha256:"))); err != nil {
		return resultfiles.TreeMember{}, exit.Internalf("cannot commit captured input: %s", err)
	}
	return resultfiles.TreeMember{Path: name, Digest: digest, Length: written}, nil
}

func finishSnapshot(destination string, members []resultfiles.TreeMember, total int64) (*records.ByteInputSnapshot, *exit.Error) {
	entries := make([]map[string]any, 0, len(members))
	for _, member := range members {
		entries = append(entries, map[string]any{"path": member.Path, "kind": "file", "blob": map[string]any{"sha256": strings.TrimPrefix(member.Digest, "sha256:"), "length": member.Length}})
	}
	raw, err := json.Marshal(map[string]any{"entries": entries})
	if err != nil {
		return nil, exit.Internalf("cannot encode captured tree: %s", err)
	}
	raw, err = canonical.NormalizeJCS(raw)
	if err != nil {
		return nil, exit.New(exit.Validation, "captured tree is not canonical")
	}
	if _, problem := resultfiles.ParseTreeManifest(raw, total); problem != nil {
		return nil, problem
	}
	manifest, err := os.OpenFile(destination, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return nil, exit.Internalf("cannot stage input manifest: %s", err)
	}
	defer manifest.Close()
	if _, err := manifest.Write(raw); err != nil {
		return nil, exit.Internalf("cannot write input manifest: %s", err)
	}
	if err := manifest.Sync(); err != nil {
		return nil, exit.Internalf("cannot sync input manifest: %s", err)
	}
	for _, directory := range []string{destination + ".files", filepath.Dir(destination)} {
		root, err := os.Open(directory)
		if err != nil {
			return nil, exit.Internalf("cannot open input directory: %s", err)
		}
		err = root.Sync()
		root.Close()
		if err != nil {
			return nil, exit.Internalf("cannot sync input directory: %s", err)
		}
	}
	digest, _ := canonical.Spell(canonical.Digest(raw))
	return &records.ByteInputSnapshot{Manifest: records.ArtifactObjectRef{Digest: digest, Length: int64(len(raw))}, ContentBytes: total, Path: destination, Body: raw}, nil
}
