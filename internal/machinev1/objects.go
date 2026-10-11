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
	"sync"

	"github.com/cozy-creator/cozy/internal/localpackage"
	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

const writeChunk = 1 << 20

// Write puts one content-addressed object on the machine, resuming from the bytes it already
// holds: a header-only Write asks, and the rest streams from there.
func Write(ctx context.Context, client pb.MachineClient, digest string, length int64, open func() (io.ReadSeekCloser, error)) error {
	_, err := writeObject(ctx, client, digest, length, open)
	return err
}

// writeObject is Write, answering whether any byte was sent (false: the machine held it all).
func writeObject(ctx context.Context, client pb.MachineClient, digest string, length int64, open func() (io.ReadSeekCloser, error)) (bool, error) {
	held, err := write(ctx, client, &pb.WriteFrame{Digest: digest, Length: uint64(length)}, nil)
	if err != nil || held == uint64(length) {
		return false, err
	}
	body, err := open()
	if err != nil {
		return false, err
	}
	defer body.Close()
	if _, err := body.Seek(int64(held), io.SeekStart); err != nil {
		return false, err
	}
	header := &pb.WriteFrame{Digest: digest, Length: uint64(length), Offset: held}
	if held, err = write(ctx, client, header, body); err != nil {
		return true, err
	}
	if held != uint64(length) {
		return true, fmt.Errorf("the machine holds %d of %s's %d bytes", held, digest, length)
	}
	return true, nil
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
	object, _, err := writeFile(ctx, client, path)
	return object, err
}

func writeFile(ctx context.Context, client pb.MachineClient, path string) (Object, bool, error) {
	object, err := fileObject(path)
	if err != nil {
		return Object{}, false, err
	}
	open := func() (io.ReadSeekCloser, error) { return os.Open(path) }
	sent, err := writeObject(ctx, client, object.Digest, object.Length, open)
	return object, sent, err
}

// WriteBytes writes a small document, named by its sha256.
func WriteBytes(ctx context.Context, client pb.MachineClient, data []byte) (Object, error) {
	object, _, err := writeBytes(ctx, client, data)
	return object, err
}

func writeBytes(ctx context.Context, client pb.MachineClient, data []byte) (Object, bool, error) {
	sum := sha256.Sum256(data)
	object := Object{Digest: "sha256:" + hex.EncodeToString(sum[:]), Length: int64(len(data))}
	open := func() (io.ReadSeekCloser, error) { return nopCloser{bytes.NewReader(data)}, nil }
	sent, err := writeObject(ctx, client, object.Digest, object.Length, open)
	return object, sent, err
}

type nopCloser struct{ io.ReadSeeker }

func (nopCloser) Close() error { return nil }

// localManifest is the LocalSource manifest the machine installs from (machine.proto): a live
// project, file by file, and the wheels its lock selects from a Tensorhub index.
type localManifest struct {
	Package        string            `json:"package"`
	Release        string            `json:"release"`
	PythonRequires string            `json:"python_requires,omitempty"`
	Files          []member          `json:"files"`
	Project        string            `json:"project,omitempty"`
	Path           string            `json:"path,omitempty"`
	Wheels         []member          `json:"wheels,omitempty"`
	Callees        map[string]string `json:"callees,omitempty"`
	Indexes        map[string]string `json:"indexes,omitempty"`
	Locals         []string          `json:"locals,omitempty"`
}

type member struct {
	Name       string `json:"name"`
	Digest     string `json:"digest"`
	Length     int64  `json:"length"`
	Executable bool   `json:"executable,omitempty"`
}

// Held is what one machine was last sent of one package, by name: its tree's files and the
// wheels its environment was built from. Nothing it names at the same digest is offered again.
type Held map[string]string

// LocalSource writes an unpublished package (its source tree and its lock's Tensorhub wheels)
// and its manifest, and answers the manifest's digest, which a run names. The machine keeps
// one tree per package: only a file `held` does not name at its digest is offered, one round
// trip when the machine has the object already and two when it does not, eight at a time.
// In place, the machine reads the files where they are (this computer's): only their digests go.
// `sent` is false when no byte moved.
func LocalSource(ctx context.Context, client pb.MachineClient, installation localpackage.Installation, held Held, inPlace bool) (manifestDigest string, sent bool, now Held, err error) {
	manifest := localManifest{Package: installation.Package, Release: installation.Release,
		PythonRequires: installation.PythonRequires, Callees: installation.Callees,
		Files: []member{}, Project: installation.Project, Indexes: installation.Indexes, Locals: installation.Locals}
	if inPlace {
		manifest.Path = installation.Root
	}
	type upload struct {
		object Object
		open   func() (io.ReadSeekCloser, error)
	}
	var uploads []upload
	now = Held{}
	for _, file := range installation.Files {
		manifest.Files = append(manifest.Files, member{file.Name, file.Digest, file.Length, file.Executable})
		if now[file.Name] = file.Digest; held[file.Name] != file.Digest && !inPlace {
			uploads = append(uploads, upload{Object{Digest: file.Digest, Length: file.Length}, func() (io.ReadSeekCloser, error) { return os.Open(file.Path) }})
		}
	}
	for _, wheel := range installation.Wheels {
		manifest.Wheels = append(manifest.Wheels, member{Name: wheel.Name, Digest: wheel.Digest, Length: wheel.Length})
		name := "wheel:" + wheel.Name
		if now[name] = wheel.Digest; held[name] != wheel.Digest {
			uploads = append(uploads, upload{Object{Digest: wheel.Digest, Length: wheel.Length}, func() (io.ReadSeekCloser, error) { return os.Open(wheel.Path) }})
		}
	}
	document, err := json.Marshal(manifest)
	if err != nil {
		return "", false, nil, err
	}
	sum := sha256.Sum256(document)
	written := Object{Digest: "sha256:" + hex.EncodeToString(sum[:]), Length: int64(len(document))}
	uploads = append(uploads, upload{written, func() (io.ReadSeekCloser, error) { return nopCloser{bytes.NewReader(document)}, nil }})
	var (
		wait  sync.WaitGroup
		mu    sync.Mutex
		first error
		slots = make(chan struct{}, 8)
	)
	for _, u := range uploads {
		wait.Add(1)
		go func() {
			defer wait.Done()
			slots <- struct{}{}
			defer func() { <-slots }()
			wrote, err := writeObject(ctx, client, u.object.Digest, u.object.Length, u.open)
			mu.Lock()
			defer mu.Unlock()
			sent = sent || wrote
			if err != nil && first == nil {
				first = fmt.Errorf("writing %s: %w", u.object.Digest, err)
			}
		}()
	}
	wait.Wait()
	return written.Digest, sent, now, first
}

// fileObject names the file at path by its sha256 and length.
func fileObject(path string) (Object, error) {
	file, err := os.Open(path)
	if err != nil {
		return Object{}, err
	}
	defer file.Close()
	hash := sha256.New()
	length, err := io.Copy(hash, file)
	if err != nil {
		return Object{}, err
	}
	return Object{Digest: "sha256:" + hex.EncodeToString(hash.Sum(nil)), Length: length}, nil
}
