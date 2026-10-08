package cli

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

func handleRentalPackageInstall(ctx *Context) *exit.Error {
	ref, plan, problem := resolveRegistryPackage(ctx, ctx.Inv.Args[0], ctx.Inv.Value("--version"))
	if problem != nil {
		return problem
	}
	// No model bindings are resolved or forwarded by package installation.
	return enqueueRentalInstall(ctx, ctx.Inv.Value("--rental"), records.RentalInstallSelection{Package: ref.String(), Release: plan.Release}, false)
}

func enqueueRentalInstall(ctx *Context, rentalName string, selection records.RentalInstallSelection, await bool) *exit.Error {
	if rentalName != machines.Local {
		ep, problem := foregroundRental(ctx, rentalName)
		if problem != nil {
			return problem
		}
		if ep != nil {
			if taken, problem := foregroundInstall(ctx, ep, rentalName, selection); taken || problem != nil {
				return problem
			}
		}
	}
	_, store, problem := rentalStores(ctx)
	if problem != nil {
		return problem
	}
	rentalID, problem := machines.InstallTarget(store, rentalName)
	store.Close()
	if problem != nil {
		return problem
	}
	state, _, problem := ensureDaemon(ctx)
	if problem != nil {
		return problem
	}
	ctx.Daemon = state
	client, problem := localapi.Open(ctx.Cfg, state)
	if problem != nil {
		return problem
	}
	result, problem := client.PrepareRentalPackage(rentalID, selection)
	if problem != nil {
		return problem
	}
	if await {
		return awaitRentalInstall(ctx, client, rentalName, result)
	}
	fields := []output.Field{{K: "id", V: result.ID}, {K: "rental", V: result.RentalID}, {K: "status", V: result.State}, {K: "target", V: rentalInstallTarget(result.Selection)}, {K: "package", V: result.Selection.Package}, {K: "release", V: result.Selection.Release}, {K: "models", V: result.Selection.Models}}
	record := compactRecord(fields, "id", "rental", "status", "target")
	record.Notes = []string{"installation accepted; Creator delivers it when the machine is ready"}
	if len(selection.Models) == 0 {
		record.Notes = append(record.Notes, "no model weights were requested")
	}
	return emit(ctx, record)
}

// foregroundInstall makes one installation on a machine this command reaches as an explicit
// endpoint (a rental under a daemon that predates cozy.machine.v1, or the machine a foreground
// run used): its warm run, watched here to its outcome, with nothing queued. Interrupting stops
// only the watch. false: the machine takes no such warm run, and the daemon's queue keeps it.
func foregroundInstall(ctx *Context, ep *machineendpoint.Endpoint, machine string, selection records.RentalInstallSelection) (bool, *exit.Error) {
	began := time.Now()
	install, problem := foregroundPrewarm(ctx, ep, machine, selection)
	if problem != nil {
		return true, problem
	}
	return true, emitInstalled(ctx, install, began)
}

// foregroundPrewarm is foregroundInstall's warm run without its record: the settled
// installation, shown by the rental's id as the daemon's queue shows it.
func foregroundPrewarm(ctx *Context, ep *machineendpoint.Endpoint, machine string, selection records.RentalInstallSelection) (records.RentalInstall, *exit.Error) {
	l := home.Paths(ctx.Cfg.Home)
	st, problem := records.Open(l.DB)
	if problem != nil {
		return records.RentalInstall{}, problem
	}
	defer st.Close()
	runs := endpointRuns(ctx, ep, l, st)
	defer runs.cancel()
	selection.Hub = either(selection.Hub, ctx.Cfg.HubURL)
	install := records.RentalInstall{ID: records.NewID("rental-install"), RentalID: ep.Name(), State: "installing", Selection: selection}
	watch := &installWatch{machine: machine, shownAt: time.Now()}
	result, problem := runs.prewarmV1(runs.ctx, install, func(p machines.InstallProgress) {
		watch.report(ctx, p.Stage, p.TransferredBytes, p.TotalBytes)
	})
	if problem != nil {
		named := *problem
		named.Message = machine + ": " + problem.Message
		return install, &named
	}
	install.Result, install.RentalID, install.State = result, either(ep.WorkspaceID, ep.Name()), "succeeded"
	return install, nil
}

// awaitRentalInstall watches one accepted installation until it settles, saying on stderr
// what the machine reports. Interrupting stops only the watch; the installation goes on.
func awaitRentalInstall(ctx *Context, client *localapi.Client, machine string, install records.RentalInstall) *exit.Error {
	began := time.Now()
	settled, problem := watchRentalInstall(ctx, client, machine, install)
	if problem != nil {
		return problem
	}
	install.Result = settled.Result
	return emitInstalled(ctx, install, began)
}

// emitInstalled is a succeeded installation; an upload names the checkpoint it put in its
// destination.
func emitInstalled(ctx *Context, install records.RentalInstall, began time.Time) *exit.Error {
	fields := []output.Field{{K: "id", V: install.ID}, {K: "rental", V: install.RentalID}, {K: "status", V: "succeeded"},
		{K: "target", V: rentalInstallTarget(install.Selection)}, {K: "elapsed", V: time.Since(began).Round(time.Second).String()}}
	shown := []string{"id", "rental", "status", "target", "elapsed"}
	var result struct {
		Models []struct {
			Published *struct{ Destination, Checkpoint string } `json:"published"`
		} `json:"models"`
		// A warm set: each member with the level it holds and why that is short of its ask.
		Set []struct {
			Package, Entrypoint, Level string
			HeldBack                   string `json:"held_back"`
		} `json:"set"`
	}
	_ = json.Unmarshal(install.Result, &result)
	for _, member := range result.Set {
		if selection := install.Selection; member.Package == selection.Package && member.Entrypoint == selection.Entrypoint {
			fields, shown = append(fields, output.Field{K: "level", V: member.Level}), append(shown, "level")
			if member.HeldBack != "" {
				fields, shown = append(fields, output.Field{K: "held_back", V: member.HeldBack}), append(shown, "held_back")
			}
		}
	}
	if len(result.Models) == 1 && result.Models[0].Published != nil {
		published := result.Models[0].Published
		fields = append(fields, output.Field{K: "destination", V: published.Destination}, output.Field{K: "checkpoint", V: published.Checkpoint})
		shown = append(shown, "destination", "checkpoint")
	}
	return emit(ctx, compactRecord(fields, shown...))
}

func watchRentalInstall(ctx *Context, client *localapi.Client, machine string, install records.RentalInstall) (records.RentalInstall, *exit.Error) {
	watch := &installWatch{machine: machine, shownAt: time.Now()}
	for {
		status, problem := client.RentalInstall(install.RentalID, install.ID)
		if problem != nil {
			return install, problem
		}
		switch status.State {
		case "succeeded":
			return status.RentalInstall, nil
		case "failed":
			return install, exit.Named(exit.Failed, either(status.ErrorCode, "rental_install.failed"), "%s: %s", machine, status.Error)
		}
		stage, done, total := status.State, uint64(0), uint64(0)
		if p := status.Progress; p != nil {
			stage, done, total = p.Stage, p.TransferredBytes, p.TotalBytes
		}
		watch.report(ctx, stage, done, total)
		time.Sleep(time.Second)
	}
}

// installWatch says on stderr what a machine reports of one installation: each new stage, and
// its bytes at most every 10 s.
type installWatch struct {
	machine, stage string
	shownAt        time.Time
	shownBytes     uint64
}

func (w *installWatch) report(ctx *Context, stage string, done, total uint64) {
	if stage == w.stage && (done <= w.shownBytes || time.Since(w.shownAt) < 10*time.Second) {
		return
	}
	line := w.machine + ": " + stage
	if total > 0 {
		line += fmt.Sprintf(" %s of %s (%d%%)", output.Bytes(int64(done)), output.Bytes(int64(total)), done*100/total)
	}
	if stage == w.stage {
		line += fmt.Sprintf(", %s/s", output.Bytes(int64(float64(done-w.shownBytes)/time.Since(w.shownAt).Seconds())))
	}
	_ = output.Progress(ctx.Err, line)
	w.stage, w.shownAt, w.shownBytes = stage, time.Now(), done
}

func rentalInstallTarget(selection records.RentalInstallSelection) string {
	if selection.Warm != "" {
		return selection.Package + "/" + selection.Entrypoint + " warm=" + selection.Warm
	}
	if selection.Package != "" {
		return selection.Package + "@" + selection.Release
	}
	models := make([]string, 0, len(selection.Models))
	for _, model := range selection.Models {
		if selection.Destination != "" {
			models = append(models, either(model.Source, either(model.Model, model.Manifest))+" -> "+selection.Destination)
			continue
		}
		name := model.Model
		if model.Release != "" {
			name += "@" + model.Release
		}
		if model.Lane != "" {
			name += "/" + model.Lane
		}
		if model.Release == "" {
			name += "#" + model.Manifest
		}
		models = append(models, name)
	}
	return strings.Join(models, ", ")
}
