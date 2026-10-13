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

// Write puts one content-addressed object of length bytes, read from r, on the machine for
// this client's signer. Writers of one object to one machine take turns in this process, so
// concurrent runs move its bytes once: the next finds it held and sends nothing. A transfer
// the connection drops resumes from what the machine holds, again while each attempt adds
// bytes; one that adds none ends with its error.
func (c *Client) Write(ctx context.Context, digest string, length uint64, r io.ReaderAt) error {
	_, err := c.write(ctx, digest, length, r)
	return err
}

// write is Write, answering whether any byte was sent (false: the machine held it all).
func (c *Client) write(ctx context.Context, digest string, length uint64, r io.ReaderAt) (sent bool, err error) {
	done, err := writers.take(ctx, c.worker+"\x00"+string(c.signer.Public)+"\x00"+digest)
	if err != nil {
		return false, err
	}
	defer done()
	conn, machine := c.uploads()
	header := &pb.WriteFrame{Digest: digest, Length: length}
	held, err := send(ctx, machine, header, nil)
	for err == nil && held < length {
		sent = true
		header.Offset = held
		var next uint64
		next, err = send(ctx, machine, header, io.NewSectionReader(r, int64(held), int64(length-held)))
		if err != nil && retryable(err) && ctx.Err() == nil {
			c.dropUploads(conn)
			conn, machine = c.uploads()
			header.Offset = 0
			if probed, problem := send(ctx, machine, header, nil); problem == nil && probed > held {
				next, err = probed, nil
			}
		}
		if err == nil && next <= held {
			err = fmt.Errorf("the machine holds %d of %s's %d bytes after a write", next, digest, length)
		}
		held = next
	}
	return sent, err
}

// send is one Write call: the header, then body's bytes. It answers the bytes the machine holds.
func send(ctx context.Context, machine pb.MachineClient, header *pb.WriteFrame, body io.Reader) (uint64, error) {
	stream, err := machine.Write(ctx)
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

// turns lets one writer at a time hold a key; the others wait for it, or for their context.
type turns struct {
	mu   sync.Mutex
	keys map[string]*turn
}

type turn struct {
	free  chan struct{}
	users int
}

// writers serializes the writers of one object to one machine, process-wide.
var writers turns

func (t *turns) take(ctx context.Context, key string) (func(), error) {
	t.mu.Lock()
	k := t.keys[key]
	if k == nil {
		k = &turn{free: make(chan struct{}, 1)}
		k.free <- struct{}{}
		if t.keys == nil {
			t.keys = map[string]*turn{}
		}
		t.keys[key] = k
	}
	k.users++
	t.mu.Unlock()
	leave := func() {
		t.mu.Lock()
		if k.users--; k.users == 0 {
			delete(t.keys, key)
		}
		t.mu.Unlock()
	}
	select {
	case <-k.free:
		return func() { k.free <- struct{}{}; leave() }, nil
	case <-ctx.Done():
		leave()
		return nil, ctx.Err()
	}
}

// Object names written bytes.
type Object struct {
	Name   string `json:"name,omitempty"`
	Digest string `json:"digest"`
	Length int64  `json:"length"`
}

// WriteFile writes the file at path, named by its sha256.
func (c *Client) WriteFile(ctx context.Context, path string) (Object, error) {
	file, err := os.Open(path)
	if err != nil {
		return Object{}, err
	}
	defer file.Close()
	object, err := objectOf(file)
	if err != nil {
		return Object{}, err
	}
	_, err = c.write(ctx, object.Digest, uint64(object.Length), file)
	return object, err
}

// WriteBytes writes a small document, named by its sha256.
func (c *Client) WriteBytes(ctx context.Context, data []byte) (Object, error) {
	object, _, err := c.writeBytes(ctx, data)
	return object, err
}

func (c *Client) writeBytes(ctx context.Context, data []byte) (Object, bool, error) {
	sum := sha256.Sum256(data)
	object := Object{Digest: "sha256:" + hex.EncodeToString(sum[:]), Length: int64(len(data))}
	sent, err := c.write(ctx, object.Digest, uint64(object.Length), bytes.NewReader(data))
	return object, sent, err
}

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
// trip when the machine has the object already and two when it does not, eight at a time, then
// the manifest.
// In place, the machine reads the files where they are (this computer's): only their digests go.
// `sent` is false when no byte moved.
func (c *Client) LocalSource(ctx context.Context, installation localpackage.Installation, held Held, inPlace bool) (manifestDigest string, sent bool, now Held, err error) {
	manifest := localManifest{Package: installation.Package, Release: installation.Release,
		PythonRequires: installation.PythonRequires, Callees: installation.Callees,
		Files: []member{}, Project: installation.Project, Indexes: installation.Indexes, Locals: installation.Locals}
	if inPlace {
		manifest.Path = installation.Root
	}
	type upload struct {
		object Object
		path   string
	}
	var uploads []upload
	now = Held{}
	for _, file := range installation.Files {
		manifest.Files = append(manifest.Files, member{file.Name, file.Digest, file.Length, file.Executable})
		if now[file.Name] = file.Digest; held[file.Name] != file.Digest && !inPlace {
			uploads = append(uploads, upload{Object{Digest: file.Digest, Length: file.Length}, file.Path})
		}
	}
	for _, wheel := range installation.Wheels {
		manifest.Wheels = append(manifest.Wheels, member{Name: wheel.Name, Digest: wheel.Digest, Length: wheel.Length})
		name := "wheel:" + wheel.Name
		if now[name] = wheel.Digest; held[name] != wheel.Digest {
			uploads = append(uploads, upload{Object{Digest: wheel.Digest, Length: wheel.Length}, wheel.Path})
		}
	}
	document, err := json.Marshal(manifest)
	if err != nil {
		return "", false, nil, err
	}
	var (
		wait  sync.WaitGroup
		mu    sync.Mutex
		first error
		slots = make(chan struct{}, 8)
	)
	for _, u := range uploads {
		wait.Go(func() {
			slots <- struct{}{}
			defer func() { <-slots }()
			wrote, err := c.writePath(ctx, u.object, u.path)
			mu.Lock()
			defer mu.Unlock()
			sent = sent || wrote
			if err != nil && first == nil {
				first = fmt.Errorf("writing %s: %w", u.object.Digest, err)
			}
		})
	}
	wait.Wait()
	if first != nil {
		return "", sent, nil, first
	}
	written, wrote, err := c.writeBytes(ctx, document)
	return written.Digest, sent || wrote, now, err
}

// writePath writes the file at path as object, the digest and length it was measured at.
func (c *Client) writePath(ctx context.Context, object Object, path string) (bool, error) {
	file, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer file.Close()
	return c.write(ctx, object.Digest, uint64(object.Length), file)
}

// fileObject names the file at path by its sha256 and length.
func fileObject(path string) (Object, error) {
	file, err := os.Open(path)
	if err != nil {
		return Object{}, err
	}
	defer file.Close()
	return objectOf(file)
}

func objectOf(r io.Reader) (Object, error) {
	hash := sha256.New()
	length, err := io.Copy(hash, r)
	if err != nil {
		return Object{}, err
	}
	return Object{Digest: "sha256:" + hex.EncodeToString(hash.Sum(nil)), Length: length}, nil
}
