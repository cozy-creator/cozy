package machinev1

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/cozy-creator/cozy/internal/localpackage"
	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

const writeChunk = 1 << 20

// Write puts one content-addressed object on the machine, resuming from the bytes it already
// holds: a header-only Write asks, and the rest streams from there.
func Write(ctx context.Context, client pb.MachineClient, digest string, length int64, open func() (io.ReadSeekCloser, error)) error {
	for {
		held, err := write(ctx, client, &pb.WriteFrame{Digest: digest, Length: uint64(length)}, nil)
		if err == nil && held == uint64(length) {
			return nil
		}
		if err == nil {
			body, openErr := open()
			if openErr != nil {
				return openErr
			}
			if _, err = body.Seek(int64(held), io.SeekStart); err == nil {
				header := &pb.WriteFrame{Digest: digest, Length: uint64(length), Offset: held}
				held, err = write(ctx, client, header, body)
			}
			body.Close()
			if err == nil && held != uint64(length) {
				return fmt.Errorf("the machine holds %d of %s's %d bytes", held, digest, length)
			}
		}
		if !expiredCap(err) || ctx.Err() != nil {
			return err
		}
		// The fresh RPC mints a new cap and probes the actual staged length. A lost key
		// never reaches this retry path, and an upload has no inference to replay.
	}
}

func write(ctx context.Context, client pb.MachineClient, header *pb.WriteFrame, body io.Reader) (uint64, error) {
	stream, err := client.Write(ctx)
	if err != nil {
		return 0, err
	}
	if err := stream.Send(header); err != nil {
		return 0, closeWrite(stream, err)
	}
	if body != nil {
		buffer := make([]byte, writeChunk)
		for {
			n, readErr := io.ReadFull(body, buffer)
			if n > 0 {
				if err := stream.Send(&pb.WriteFrame{Data: buffer[:n]}); err != nil {
					return 0, closeWrite(stream, err)
				}
			}
			if readErr == io.EOF || readErr == io.ErrUnexpectedEOF {
				break
			}
			if readErr != nil {
				return 0, readErr
			}
		}
	}
	result, err := stream.CloseAndRecv()
	if err != nil {
		return 0, err
	}
	return result.GetHeld(), nil
}

// closeWrite answers the machine's refusal when a send fails because it ended the stream.
func closeWrite(stream pb.Machine_WriteClient, sent error) error {
	if _, err := stream.CloseAndRecv(); err != nil {
		return err
	}
	return sent
}

// Object names written bytes.
type Object struct {
	Name   string `json:"name,omitempty"`
	Digest string `json:"digest"`
	Length int64  `json:"length"`
}

// WriteFile writes the file at path, named by its sha256.
func WriteFile(ctx context.Context, client pb.MachineClient, path string) (Object, error) {
	file, err := os.Open(path)
	if err != nil {
		return Object{}, err
	}
	hash := sha256.New()
	length, err := io.Copy(hash, file)
	file.Close()
	if err != nil {
		return Object{}, err
	}
	object := Object{Digest: "sha256:" + hex.EncodeToString(hash.Sum(nil)), Length: length}
	open := func() (io.ReadSeekCloser, error) { return os.Open(path) }
	return object, Write(ctx, client, object.Digest, length, open)
}

// WriteBytes writes a small document, named by its sha256.
func WriteBytes(ctx context.Context, client pb.MachineClient, data []byte) (Object, error) {
	sum := sha256.Sum256(data)
	object := Object{Digest: "sha256:" + hex.EncodeToString(sum[:]), Length: int64(len(data))}
	open := func() (io.ReadSeekCloser, error) { return nopCloser{bytes.NewReader(data)}, nil }
	return object, Write(ctx, client, object.Digest, object.Length, open)
}

type nopCloser struct{ io.ReadSeeker }

func (nopCloser) Close() error { return nil }

// localManifest is the LocalSource manifest the machine installs from (machine.proto).
type localManifest struct {
	Package        string   `json:"package"`
	Release        string   `json:"release"`
	PythonRequires string   `json:"python_requires,omitempty"`
	PythonVersion  string   `json:"python_version,omitempty"`
	Source         Object   `json:"source"`
	Wheels         []Object `json:"wheels,omitempty"`
	Requirements   *Object  `json:"requirements,omitempty"`
}

// LocalSource writes an unpublished package (its source archive, vendored wheels and locked
// requirements) and then its manifest. A run names the manifest's digest; unchanged code
// writes nothing new and reopens the machine's installation.
func LocalSource(ctx context.Context, client pb.MachineClient, installation localpackage.Installation) (string, error) {
	manifest := localManifest{Package: installation.Package, Release: installation.Release,
		PythonRequires: installation.PythonRequires, PythonVersion: installation.PythonVersion}
	sourced := false
	for _, file := range installation.Files {
		object, err := WriteFile(ctx, client, file.Path)
		if err != nil {
			return "", fmt.Errorf("writing %s: %w", file.Filename, err)
		}
		if file.Kind == "source" {
			manifest.Source, sourced = object, true
			continue
		}
		object.Name = file.Filename
		manifest.Wheels = append(manifest.Wheels, object)
	}
	if !sourced {
		return "", fmt.Errorf("%s has no source archive to write", installation.Package)
	}
	if len(installation.DependencyRequirements) > 0 {
		requirements, err := WriteBytes(ctx, client, installation.DependencyRequirements)
		if err != nil {
			return "", fmt.Errorf("writing the locked requirements: %w", err)
		}
		manifest.Requirements = &requirements
	}
	document, err := json.Marshal(manifest)
	if err != nil {
		return "", err
	}
	written, err := WriteBytes(ctx, client, document)
	return written.Digest, err
}
