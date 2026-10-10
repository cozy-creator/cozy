package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/devcomfy"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/rental"
)

type DevCmd struct {
	Comfy DevComfyCmd `cmd:"" help:"Submit or resume one Comfy graph on an owned development rental; no Python executor network access."`
}

type DevComfyCmd struct {
	Rental         string `required:"" help:"Existing owned development rental; never rent or replace one."`
	Input          string `required:"" predictor:"file" help:"JSON containing graph_json, output_root and optional trace settings."`
	IdempotencyKey string `required:"" name:"idempotency-key" help:"Stable identity; repeat the same inputs/key to recover without resubmission."`
	SSHKnownHosts  string `required:"" name:"ssh-known-hosts" predictor:"file" help:"Existing verified SSH host-key file; strict checking is mandatory."`
	SSHKey         string `name:"ssh-key" predictor:"file" help:"Private SSH key; defaults to the configured or managed rental identity."`
	Python         string `default:"python3" help:"Remote Python3.11+ executable for the trusted service observer."`
	RemoteState    string `default:"/root/.cozy/dev/comfy" name:"remote-state" help:"Remote directory retaining this development operation's receipts."`
	Out            string `help:"Local artifact directory; defaults to this operation under Cozy outputs."`
}

func (c *DevComfyCmd) Run(r *Runtime) error {
	return r.call(handleDevComfy, nil, nil, values("--rental", c.Rental, "--input", c.Input,
		"--idempotency-key", c.IdempotencyKey, "--ssh-known-hosts", c.SSHKnownHosts,
		"--ssh-key", c.SSHKey, "--python", c.Python, "--remote-state", c.RemoteState, "--out", c.Out), false)
}

func handleDevComfy(ctx *Context) *exit.Error {
	name := ctx.Inv.Value("--rental")
	adoptRentalHub(ctx, name)
	layout, store, problem := rentalStores(ctx)
	if problem != nil {
		return problem
	}
	defer store.Close()
	address, problem := readDevelopmentRentalSSH(ctx, store, name)
	if problem != nil {
		return problem
	}
	expand := func(value string) (string, *exit.Error) {
		p, e := config.ExpandHome(value)
		if e != nil {
			return "", exit.Usagef("cannot resolve path: %s", e)
		}
		p, e = filepath.Abs(p)
		if e != nil {
			return "", exit.Usagef("cannot resolve path: %s", e)
		}
		return p, nil
	}
	inputPath, problem := expand(ctx.Inv.Value("--input"))
	if problem != nil {
		return problem
	}
	inputFile, err := os.Open(inputPath)
	if err != nil {
		return exit.Usagef("cannot read Comfy input: %s", err)
	}
	raw, err := io.ReadAll(io.LimitReader(inputFile, (5<<20)+1))
	inputFile.Close()
	if err != nil {
		return exit.Usagef("cannot read Comfy input: %s", err)
	}
	in, err := devcomfy.ParseInput(raw)
	if err != nil {
		return exit.Usagef("invalid Comfy input: %s", err)
	}
	known, problem := expand(ctx.Inv.Value("--ssh-known-hosts"))
	if problem != nil {
		return problem
	}
	if info, err := os.Stat(known); err != nil || !info.Mode().IsRegular() {
		return exit.New(exit.Credential, "SSH known-hosts must be an existing verified regular file")
	}
	key := ctx.Inv.Value("--ssh-key")
	if key == "" {
		public := ctx.Cfg.RentalsSSHPublicKey
		if public != "" {
			if !filepath.IsAbs(public) && !strings.HasPrefix(public, "~/") {
				public = filepath.Join(ctx.Cfg.Home, public)
			}
			key = strings.TrimSuffix(public, ".pub")
		} else {
			key = filepath.Join(ctx.Cfg.Home, "auth", "rental-ssh")
		}
	}
	key, problem = expand(key)
	if problem != nil {
		return problem
	}
	if info, err := os.Stat(key); err != nil || !info.Mode().IsRegular() || info.Mode().Perm()&0077 != 0 {
		return exit.New(exit.Credential, "SSH key must be an existing private regular file")
	}
	operation := devcomfy.Operation(address.Row.ID, ctx.Inv.Value("--idempotency-key"))
	out := ctx.Inv.Value("--out")
	if out == "" {
		out = filepath.Join(layout.Outputs, "dev-comfy", operation)
	}
	out, problem = expand(out)
	if problem != nil {
		return problem
	}
	observed, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	keepalive := func(call context.Context) error {
		_, problem := rental.KeepAlive(call, layout, address.Row)
		if problem != nil {
			if problem.Code != exit.Unavailable {
				return &devcomfy.IdentityError{Err: problem}
			}
			return problem
		}
		return nil
	}
	// Refuse identity/auth failures before sending a Comfy graph to the SSH host.
	call, cancel := context.WithTimeout(observed, 15*time.Second)
	err = keepalive(call)
	cancel()
	if err != nil {
		return exit.As(err)
	}
	last := ""
	result, err := devcomfy.Run(observed, devcomfy.Options{
		SSH:   devcomfy.SSH{Host: address.Host, Port: strconv.Itoa(address.Port), Key: key, KnownHosts: known, Python: ctx.Inv.Value("--python"), Environment: ctx.Cfg.Tool()},
		Input: in, Rental: address.Row.ID, Worker: address.Row.ExpectedWorkerID, Boot: address.Row.ExpectedWorkerBootID,
		Key: ctx.Inv.Value("--idempotency-key"), StateRoot: ctx.Inv.Value("--remote-state"),
		ReceiptDir: filepath.Join(layout.Root, "dev", "comfy", operation), OutputDir: out, Keepalive: keepalive,
		Progress: func(r devcomfy.Result) {
			signature := r.Status + "/" + r.PromptID
			if signature == last {
				return
			}
			last = signature
			if ctx.Mode().JSON {
				_ = json.NewEncoder(ctx.Err).Encode(map[string]any{"type": "comfy.observation", "operation": operation, "status": r.Status, "prompt_id": r.PromptID})
			} else {
				fmt.Fprintf(ctx.Err, "Comfy %s: %s\n", r.PromptID, r.Status)
			}
		},
	})
	if err != nil {
		return exit.New(exit.Unavailable, "%s", err).WithRemedy("repeat the same cozy dev comfy command and idempotency key; accepted Comfy work was not canceled")
	}
	if result.Status != "success" {
		ctx.exitCode = 1
	}
	var seconds any
	if result.ServerExecutionSeconds != nil {
		seconds = *result.ServerExecutionSeconds
	}
	return emit(ctx, compactRecord([]output.Field{{K: "operation", V: result.Operation}, {K: "status", V: result.Status},
		{K: "prompt_id", V: result.PromptID}, {K: "server_execution_seconds", V: seconds},
		{K: "detail", V: result.Detail}, {K: "output_directory", V: out}, {K: "files", V: result.Files}},
		"status", "prompt_id", "server_execution_seconds", "output_directory", "detail"))
}
