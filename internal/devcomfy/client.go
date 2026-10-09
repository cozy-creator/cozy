// Package devcomfy is an operator-owned Comfy client, outside Python execution.
// SSH host verification and the rental's ordinary keepalive remain mandatory.
package devcomfy

import (
	"bytes"
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/flock"
)

//go:embed observer.py
var observerSource string

//go:embed remote.py
var driverSource string

type Input struct {
	GraphJSON     string `json:"graph_json"`
	OutputRoot    string `json:"output_root"`
	Port          int    `json:"port"`
	TimeoutS      int    `json:"timeout_s"`
	Trace         bool   `json:"trace"`
	TraceRoot     string `json:"trace_root"`
	ExpectedSteps int    `json:"expected_steps"`
}

func ParseInput(raw []byte) (Input, error) {
	in := Input{Port: 8188, TimeoutS: 3600, ExpectedSteps: 8}
	if len(raw) > 5<<20 {
		return in, errors.New("Comfy input exceeds 5 MiB")
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&in); err != nil {
		return in, err
	}
	if d.Decode(new(any)) != io.EOF {
		return in, errors.New("Comfy input must contain exactly one JSON document")
	}
	var graph map[string]json.RawMessage
	if json.Unmarshal([]byte(in.GraphJSON), &graph) != nil || len(graph) == 0 || len(in.GraphJSON) > 4<<20 {
		return in, errors.New("graph_json must contain a nonempty Comfy API graph below 4 MiB")
	}
	if in.Port < 1 || in.Port > 65535 || in.TimeoutS < 1 || in.TimeoutS > 7200 || (in.ExpectedSteps != 8 && in.ExpectedSteps != 30) {
		return in, errors.New("invalid Comfy port, timeout or expected step count")
	}
	if !strings.HasPrefix(in.OutputRoot, "/") || (in.Trace && !strings.HasPrefix(in.TraceRoot, "/")) {
		return in, errors.New("remote output_root and traced trace_root must be absolute")
	}
	return in, nil
}

type File struct {
	Name   string `json:"name"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}
type Result struct {
	Status                 string   `json:"status"`
	PromptID               string   `json:"prompt_id"`
	Detail                 string   `json:"detail,omitempty"`
	ServerExecutionSeconds *float64 `json:"server_execution_seconds,omitempty"`
	Files                  []File   `json:"files,omitempty"`
	Operation              string   `json:"operation"`
	OutputDirectory        string   `json:"output_directory,omitempty"`
}

type SSH struct {
	Host, Port, Key, KnownHosts, Python string
	Environment                         []string
}

type transportError struct {
	err   error
	retry bool
}

func (e *transportError) Error() string { return e.err.Error() }
func (e *transportError) Unwrap() error { return e.err }

// IdentityError ends observation rather than retrying a changed rental identity.
type IdentityError struct{ Err error }

func (e *IdentityError) Error() string { return e.Err.Error() }
func (e *IdentityError) Unwrap() error { return e.Err }

type limitedBuffer struct {
	bytes.Buffer
	limit int
}

func (b *limitedBuffer) Write(p []byte) (int, error) {
	if len(p) > b.limit-b.Len() {
		return 0, errors.New("Comfy control response exceeds its bound")
	}
	return b.Buffer.Write(p)
}

func quote(s string) string { return "'" + strings.ReplaceAll(s, "'", "'\"'\"'") + "'" }

func (s SSH) command(ctx context.Context, request map[string]any) *exec.Cmd {
	bootstrap := "import json,sys; _cozy_request=json.loads(sys.stdin.readline()); exec(compile(sys.stdin.read(), '<cozy-dev-comfy>', 'exec'))"
	command := exec.CommandContext(ctx, "ssh", "-F", "none", "-T", "-i", s.Key,
		"-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes",
		"-o", "UserKnownHostsFile="+strconv.Quote(s.KnownHosts), "-o", "ConnectTimeout=10",
		"-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3", "-p", s.Port,
		"root@"+s.Host, "CUDA_VISIBLE_DEVICES='' "+quote(s.Python)+" -c "+quote(bootstrap))
	command.Env = s.Environment
	raw, _ := json.Marshal(request)
	command.Stdin = io.MultiReader(bytes.NewReader(append(raw, '\n')), strings.NewReader(driverSource))
	return command
}

func (s SSH) call(ctx context.Context, request map[string]any) (Result, error) {
	var result Result
	call, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	command := s.command(call, request)
	stdout, stderr := limitedBuffer{limit: 8 << 20}, limitedBuffer{limit: 1 << 20}
	command.Stdout, command.Stderr = &stdout, &stderr
	if err := command.Run(); err != nil {
		var exited *exec.ExitError
		retry := (errors.As(err, &exited) && exited.ExitCode() == 255) || (call.Err() != nil && ctx.Err() == nil)
		for _, reason := range []string{"host key verification failed", "remote host identification has changed", "permission denied", "bad permissions", "no such identity"} {
			if strings.Contains(strings.ToLower(stderr.String()), reason) {
				retry = false
			}
		}
		return result, &transportError{fmt.Errorf("Comfy SSH observation: %w: %s", err, tail(stderr.String(), 2000)), retry}
	}
	if stdout.Len() > 8<<20 {
		return result, errors.New("Comfy response exceeds its bound")
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		return result, fmt.Errorf("invalid Comfy receipt: %w", err)
	}
	switch result.Status {
	case "prepared", "running", "observer_stopped", "success", "failed", "timed_out", "error":
	default:
		return result, errors.New("invalid Comfy operation state")
	}
	return result, nil
}

func tail(s string, n int) string {
	if len(s) > n {
		return s[len(s)-n:]
	}
	return s
}

type Options struct {
	SSH                                                         SSH
	Input                                                       Input
	Rental, Worker, Boot, Key, StateRoot, ReceiptDir, OutputDir string
	Keepalive                                                   func(context.Context) error
	Progress                                                    func(Result)
}

func Operation(rental, key string) string {
	sum := sha256.Sum256([]byte(rental + "\x00" + key))
	return hex.EncodeToString(sum[:])
}

func save(path string, value any) error {
	raw, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	file, err := os.CreateTemp(filepath.Dir(path), ".comfy-receipt-")
	if err != nil {
		return err
	}
	name := file.Name()
	defer os.Remove(name)
	if err = file.Chmod(0600); err == nil {
		_, err = file.Write(append(raw, '\n'))
	}
	if err == nil {
		err = file.Sync()
	}
	if closed := file.Close(); err == nil {
		err = closed
	}
	if err != nil {
		return err
	}
	if err = os.Rename(name, path); err != nil {
		return err
	}
	dir, err := os.Open(filepath.Dir(path))
	if err != nil {
		return err
	}
	defer dir.Close()
	return dir.Sync()
}

// Run never controls the Comfy server, buys a rental, or executes a Python job.
// Repeating the same operation resumes its remote observer and original prompt.
func Run(parent context.Context, o Options) (result Result, returned error) {
	if o.Key == "" || len(o.Key) > 256 || o.Keepalive == nil {
		return result, errors.New("Comfy requires an idempotency key and rental keepalive")
	}
	op := Operation(o.Rental, o.Key)
	if err := os.MkdirAll(o.ReceiptDir, 0700); err != nil {
		return result, err
	}
	lock, err := os.OpenFile(filepath.Join(o.ReceiptDir, "observer.lock"), os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return result, err
	}
	defer lock.Close()
	if err = flock.Exclusive(lock); err != nil {
		return result, errors.New("this Comfy operation already has a local observer")
	}
	if err := os.MkdirAll(o.OutputDir, 0700); err != nil {
		return result, err
	}
	identity := struct {
		Rental, Worker, Boot, StateRoot string
		Input                Input
	}{o.Rental, o.Worker, o.Boot, o.StateRoot, o.Input}
	raw, _ := json.Marshal(identity)
	receipt := filepath.Join(o.ReceiptDir, "request.json")
	if prior, err := os.ReadFile(receipt); err == nil {
		var held any
		if json.Unmarshal(prior, &held) != nil {
			return result, errors.New("retained Comfy request is unreadable")
		}
		canonical, _ := json.Marshal(held)
		var current any
		_ = json.Unmarshal(raw, &current)
		wanted, _ := json.Marshal(current)
		if !bytes.Equal(canonical, wanted) {
			return result, errors.New("idempotency conflict: rental boot or Comfy inputs changed")
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return result, err
	} else if err = save(receipt, identity); err != nil {
		return result, err
	}
	ctx, cancel := context.WithTimeout(parent, time.Duration(o.Input.TimeoutS)*time.Second+15*time.Minute)
	defer cancel()
	// Failure to renew does not abandon observation. Retry until the command's
	// bounded deadline; identity refusals are returned by the pinned connection.
	keepDone := make(chan struct{})
	identityFailure := make(chan error, 1)
	go func() {
		defer close(keepDone)
		delay := time.Duration(0)
		for {
			if delay > 0 {
				select {
				case <-ctx.Done():
					return
				case <-time.After(delay):
				}
			}
			call, stop := context.WithTimeout(ctx, 10*time.Second)
			err := o.Keepalive(call)
			stop()
			_ = save(filepath.Join(o.ReceiptDir, "keepalive.json"), map[string]any{"at": time.Now().UTC(), "error": fmt.Sprint(err)})
			var identity *IdentityError
			if errors.As(err, &identity) {
				identityFailure <- err
				cancel()
				return
			}
			if err == nil {
				delay = 60 * time.Second
			} else {
				delay = 5 * time.Second
			}
			if ctx.Err() != nil {
				return
			}
		}
	}()
	defer func() {
		cancel()
		<-keepDone
		select {
		case failed := <-identityFailure:
			returned = failed
		default:
		}
	}()
	preparedPath := filepath.Join(o.ReceiptDir, "remote-prepared.json")
	_, preparedErr := os.Stat(preparedPath)
	if preparedErr != nil && !errors.Is(preparedErr, os.ErrNotExist) {
		return result, preparedErr
	}
	request := map[string]any{"action": "prepare", "operation": op, "state_root": o.StateRoot, "payload": o.Input, "observer_source": observerSource, "driver_source": driverSource, "require_existing": preparedErr == nil}
	delay := time.Second
	for {
		var err error
		result, err = o.SSH.call(ctx, request)
		if err == nil {
			if request["action"] == "prepare" && result.Status == "prepared" {
				if err = save(preparedPath, map[string]any{"operation": op, "remote_state": o.StateRoot}); err != nil {
					return result, err
				}
				request["action"], request["require_existing"] = "start", true
				continue
			}
			break
		}
		var transport *transportError
		if !errors.As(err, &transport) || !transport.retry {
			return result, err
		}
		if ctx.Err() != nil {
			select {
			case failed := <-identityFailure:
				return result, failed
			default:
			}
			return result, fmt.Errorf("Comfy observation interrupted; repeat the same key to recover: %w", err)
		}
		_ = save(filepath.Join(o.ReceiptDir, "last-error.json"), map[string]any{"at": time.Now().UTC(), "error": err.Error()})
		select {
		case <-ctx.Done():
			select {
			case failed := <-identityFailure:
				return result, failed
			default:
			}
			return result, ctx.Err()
		case <-time.After(delay):
		}
		if delay < 15*time.Second {
			delay *= 2
		}
	}
	request = map[string]any{"action": "status", "operation": op, "state_root": o.StateRoot}
	for {
		result.Operation = op
		result.OutputDirectory = o.OutputDir
		_ = save(filepath.Join(o.ReceiptDir, "status.json"), result)
		if o.Progress != nil {
			o.Progress(result)
		}
		if result.Status != "running" && result.Status != "observer_stopped" {
			break
		}
		if result.Status == "observer_stopped" {
			return result, errors.New("Comfy observer stopped; repeat the same command to resume its original prompt")
		}
		select {
		case <-ctx.Done():
			return result, ctx.Err()
		case <-time.After(3 * time.Second):
		}
		next, err := o.SSH.call(ctx, request)
		if err == nil {
			result = next
		} else {
			var transport *transportError
			if !errors.As(err, &transport) || !transport.retry {
				return result, err
			}
			_ = save(filepath.Join(o.ReceiptDir, "last-error.json"), map[string]any{"error": err.Error()})
		}
	}
	var total int64
	if len(result.Files) > 8192 {
		return result, errors.New("Comfy artifact count exceeds its bound")
	}
	for _, file := range result.Files {
		_, validDigest := hex.DecodeString(file.SHA256)
		if !filepath.IsLocal(file.Name) || strings.Contains(file.Name, "\\") || file.Bytes < 0 || file.Bytes > (64<<30)-total || len(file.SHA256) != 64 || validDigest != nil {
			return result, errors.New("invalid retained Comfy artifact manifest")
		}
		total += file.Bytes
		if err := o.fetch(ctx, op, file); err != nil {
			return result, err
		}
	}
	if err := save(filepath.Join(o.OutputDir, "comfy-result.json"), result); err != nil {
		return result, err
	}
	return result, nil
}

func digest(path string) (string, int64, error) {
	f, e := os.Open(path)
	if e != nil {
		return "", 0, e
	}
	defer f.Close()
	h := sha256.New()
	n, e := io.Copy(h, f)
	return hex.EncodeToString(h.Sum(nil)), n, e
}

func (o Options) fetch(ctx context.Context, operation string, file File) error {
	target := filepath.Join(o.OutputDir, filepath.FromSlash(file.Name))
	if info, err := os.Lstat(target); err == nil {
		if !info.Mode().IsRegular() {
			return errors.New("Comfy output destination is not a regular file")
		}
		sha, n, err := digest(target)
		if err == nil && sha == file.SHA256 && n == file.Bytes {
			return nil
		}
		return fmt.Errorf("refusing to overwrite different artifact %s", target)
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(target), 0700); err != nil {
		return err
	}
	base, err := filepath.EvalSymlinks(o.OutputDir)
	if err != nil {
		return err
	}
	parent, err := filepath.EvalSymlinks(filepath.Dir(target))
	if err != nil {
		return err
	}
	rel, err := filepath.Rel(base, parent)
	if err != nil || (!filepath.IsLocal(rel) && rel != ".") {
		return errors.New("Comfy artifact parent escapes the output directory")
	}
	partial := target + ".partial-" + operation[:12]
	if info, err := os.Lstat(partial); err == nil && !info.Mode().IsRegular() {
		return errors.New("partial Comfy artifact is not a regular file")
	}
	f, err := os.OpenFile(partial, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
	if err != nil {
		return err
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return err
	}
	offset := info.Size()
	if offset > file.Bytes {
		f.Close()
		return errors.New("partial artifact exceeds its retained length")
	}
	request := map[string]any{"action": "fetch", "operation": operation, "state_root": o.StateRoot, "name": file.Name, "offset": offset}
	cmd := o.SSH.command(ctx, request)
	var stderr bytes.Buffer
	cmd.Stdout = f
	cmd.Stderr = &stderr
	err = cmd.Run()
	syncErr := f.Sync()
	closeErr := f.Close()
	if err != nil {
		return fmt.Errorf("Comfy artifact transfer interrupted; repeat the same key: %w: %s", err, tail(stderr.String(), 1000))
	}
	if syncErr != nil {
		return syncErr
	}
	if closeErr != nil {
		return closeErr
	}
	sha, n, err := digest(partial)
	if err != nil {
		return err
	}
	if sha != file.SHA256 || n != file.Bytes {
		return errors.New("retained Comfy artifact digest or length differs")
	}
	return publishVerified(partial, target)
}

// Both names are siblings on the same filesystem. A hard link publishes the
// verified bytes atomically while refusing an independently created destination.
func publishVerified(partial, target string) error {
	if err := os.Link(partial, target); err != nil {
		return fmt.Errorf("cannot publish Comfy artifact without replacing another file; verified partial retained: %w", err)
	}
	directory, err := os.Open(filepath.Dir(target))
	if err != nil {
		return err
	}
	defer directory.Close()
	if err = directory.Sync(); err != nil {
		return err
	}
	if err = os.Remove(partial); err != nil {
		return nil
	} // Published bytes are authoritative; extra link is harmless.
	return directory.Sync()
}
