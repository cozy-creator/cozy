package cli

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/tfs"
)

// The model transfer verbs (cl-012). `model upload` is th-002's declare-first protocol
// driven from this side; `model download` is its inverse into the local canonical store. Neither owns a
// byte or a protocol: the byte plane is TensorFS's (internal/tfs) and the protocol is
// the hub's (internal/hub). What these own is the argument surface, the progress
// accounting, and the exit code.

// tooling opens the byte plane and the hub together, with the credential this
// invocation carries. A verb that could not get either says so before it moves.
func tooling(ctx *Context) (*tfs.Tool, *hub.Client, home.Layout, *exit.Error) {
	layout, e := home.Open(ctx.Cfg.Home)
	if e != nil {
		return nil, nil, layout, e
	}
	tool, e := tfs.Open(ctx.Cfg)
	if e != nil {
		return nil, nil, layout, e
	}
	return tool, client(ctx), layout, nil
}

func progress(ctx *Context) func(string) {
	return func(line string) { _ = output.Progress(ctx.Err, line) }
}

func localTensorFS(ctx *Context) (*tfs.Tool, home.Layout, *exit.Error) {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return nil, layout, problem
	}
	tool, problem := tfs.Open(ctx.Cfg)
	return tool, layout, problem
}

// handleModelList lists what this computer's machine holds, from its Status: one row per
// repository release lane, with the repository's bytes (TensorFS measures per repository).
func handleModelList(ctx *Context) *exit.Error {
	host, problem := localMachineHost(ctx)
	if problem != nil {
		return problem
	}
	readCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	frame, problem := host.ReadStatus(readCtx)
	if problem != nil {
		return problem
	}
	if frame == nil {
		return exit.Named(exit.Unavailable, "machine.stopped", "this computer's machine is not running; it lists the models it holds").
			WithNext("cozy machine start")
	}
	list := output.List{
		Name: "models", Fields: []string{"model", "release", "lane", "size", "shared"},
		AllFields: []string{"model", "kind", "release", "lane", "manifest_id", "yanked", "size", "shared"},
		Bytes:     []string{"size", "shared", "unique"},
		Machine:   []string{"unique"},
	}
	for _, model := range frame.GetModels() {
		kind := "catalog"
		if strings.HasPrefix(model.GetRepository(), "local/") {
			kind = "local"
		}
		total, unique := int64(model.GetTotalBytes()), int64(model.GetUniqueBytes())
		row := func(release, lane, manifest string, yanked bool) map[string]string {
			return map[string]string{"model": model.GetRepository(), "kind": kind, "release": release, "lane": lane,
				"manifest_id": manifest, "yanked": fmt.Sprint(yanked), "size": output.Int(total),
				"shared": output.Int(total - unique), "unique": output.Int(unique)}
		}
		if len(model.GetCheckpoints()) == 0 {
			list.Rows = append(list.Rows, row("", "", "", false))
		}
		for _, checkpoint := range model.GetCheckpoints() {
			list.Rows = append(list.Rows, row(checkpoint.GetRelease(), checkpoint.GetLane(), checkpoint.GetManifest(), checkpoint.GetYanked()))
		}
	}
	list.Aggregates = []output.Field{{K: "store", V: storeUsage{Size: int64(frame.GetModelsBytes()), Models: len(frame.GetModels())}}}
	return emit(ctx, list)
}

// storeUsage is the model list's footer: the bytes the machine's models hold together, shared
// content counted once.
type storeUsage struct {
	Size   int64 `json:"size"`
	Models int   `json:"models"`
}

func (s storeUsage) Human() string {
	noun := "models"
	if s.Models == 1 {
		noun = "model"
	}
	return fmt.Sprintf("%s in %d %s", output.Bytes(s.Size), s.Models, noun)
}

// handleMachineModelDownload freezes a catalog selection before durable acceptance and
// downloads it into a machine's own store: the named rental's, or without --rental this
// computer's machine, the one local runs read.
func handleMachineModelDownload(ctx *Context) *exit.Error {
	machine := strings.TrimSpace(ctx.Inv.Value("--rental"))
	if machine == "" {
		machine = machines.Local
	}
	if ctx.Inv.Bool("--rental-only") || ctx.Inv.Value("--idempotency-key") != "" {
		return exit.Usagef("machine model download does not accept --rental-only or --idempotency-key")
	}
	model, problem := resolveRemoteModel(ctx, client(ctx), "", launch.Slot{}, ctx.Inv.Args[0], ctx.Inv.Value("--lane"), nil)
	if problem != nil {
		return problem
	}
	selection := records.RentalInstallSelection{Models: []records.ModelRef{model}}
	return enqueueRentalInstall(ctx, machine, selection, ctx.Inv.Bool("--await"))
}
