package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

// handleRentalPackageInstall asks the Hub nothing: the machine reads the release at the hub the
// selection names, and one it already holds is the answer with no read anywhere. No model
// bindings are resolved or forwarded by package installation.
func handleRentalPackageInstall(ctx *Context) *exit.Error {
	ref, release, problem := registryPackageRef(ctx.Inv.Args[0], ctx.Inv.Value("--version"))
	if problem != nil {
		return problem
	}
	selection := records.RentalInstallSelection{Package: ref.String(), Release: release, Hub: ctx.Cfg.HubURL}
	return enqueueRentalInstall(ctx, ctx.Inv.Value("--rental"), selection, !ctx.Inv.Bool("--no-wait"))
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
	client, rentalID, problem := rentalInstallClient(ctx, rentalName)
	if problem != nil {
		return problem
	}
	if selection, problem = olderDaemonSelection(ctx, client, selection); problem != nil {
		return problem
	}
	result, problem := client.PrepareRentalPackage(rentalID, selection)
	if problem != nil {
		return problem
	}
	if await {
		began := time.Now()
		settled, detached, problem := watchRentalInstall(ctx, client, rentalName, result)
		if problem != nil {
			return problem
		}
		if !detached {
			return emitInstalled(ctx, settled, began)
		}
		result = settled
	}
	fields := []output.Field{{K: "id", V: result.ID}, {K: "rental", V: result.RentalID}, {K: "status", V: result.State}, {K: "target", V: result.Selection.Target()}, {K: "package", V: result.Selection.Package}, {K: "release", V: result.Selection.Release}, {K: "models", V: result.Selection.Models}}
	record := compactRecord(fields, "id", "rental", "status", "target")
	record.Notes = []string{"installation accepted; Creator delivers it when the machine is ready"}
	if selection.Package != "" && rentalName != machines.Local {
		record.Next = []string{"cozy package list --rental=" + rentalName}
	}
	if len(selection.Models) == 0 {
		record.Notes = append(record.Notes, "no model weights were requested")
	}
	return emit(ctx, record)
}

// rentalInstallClient is the daemon that queues installations, and the rental's id there.
func rentalInstallClient(ctx *Context, rentalName string) (*localapi.Client, string, *exit.Error) {
	_, store, problem := rentalStores(ctx)
	if problem != nil {
		return nil, "", problem
	}
	rentalID, problem := machines.InstallTarget(store, rentalName)
	store.Close()
	if problem != nil {
		return nil, "", problem
	}
	client, problem := dial(ctx)
	return client, rentalID, problem
}

// olderDaemonSelection is the selection a daemon from before package_verbs takes: an exact
// release, which this command then reads at the hub; local code and removals it cannot queue.
func olderDaemonSelection(ctx *Context, daemon *localapi.Client, selection records.RentalInstallSelection) (records.RentalInstallSelection, *exit.Error) {
	if selection.Package == "" || selection.Release != "" && selection.Local == "" && selection.Remove == "" {
		return selection, nil
	}
	if caps, problem := daemon.Capabilities(); problem != nil || caps.PackageVerbs {
		return selection, problem
	}
	if selection.Local != "" || selection.Remove != "" {
		return selection, exit.Named(exit.Unavailable, "daemon.package_verbs_unavailable",
			"this computer's daemon predates installing local code on, and removing packages from, a rental").
			WithRemedy("restart it with `cozy down`; running work and rentals continue")
	}
	ref, problem := hub.ParseRef(selection.Package)
	if problem != nil {
		return selection, problem
	}
	hctx, cancel := hub.Context()
	defer cancel()
	card, problem := client(ctx.forHub(selection.Hub)).PackageCard(hctx, ref)
	if problem != nil {
		return selection, packageHubProblem(ctx, selection.Package, problem)
	}
	selection.Release, problem = newestPackageRelease(card.Releases)
	return selection, problem
}

// settleRentalInstall queues one selection on the rental and follows it to its outcome: what
// `package update` and `package remove` do once per package.
func settleRentalInstall(ctx *Context, rentalName string, selection records.RentalInstallSelection) (records.RentalInstall, *exit.Error) {
	if rentalName != machines.Local {
		ep, problem := foregroundRental(ctx, rentalName)
		if problem != nil {
			return records.RentalInstall{}, problem
		}
		if ep != nil {
			return foregroundPrewarm(ctx, ep, rentalName, selection)
		}
	}
	client, rentalID, problem := rentalInstallClient(ctx, rentalName)
	if problem != nil {
		return records.RentalInstall{}, problem
	}
	if selection, problem = olderDaemonSelection(ctx, client, selection); problem != nil {
		return records.RentalInstall{}, problem
	}
	queued, problem := client.PrepareRentalPackage(rentalID, selection)
	if problem != nil {
		return records.RentalInstall{}, problem
	}
	settled, detached, problem := watchRentalInstall(ctx, client, rentalName, queued)
	if problem == nil && detached {
		problem = exit.New(exit.Canceled, "detached from %s; it continues on %s", queued.Selection.Target(), rentalName)
	}
	return settled, problem
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

// emitInstalled is a succeeded installation; an upload names the checkpoint it put in its
// destination.
func emitInstalled(ctx *Context, install records.RentalInstall, began time.Time) *exit.Error {
	var result struct {
		// A package installation: the release the machine installed.
		Package, Release string
		Models           []struct {
			Published *struct{ Destination, Checkpoint string } `json:"published"`
		} `json:"models"`
		// A warm set: each member with the level it holds and why that is short of its ask.
		Set []struct {
			Package, Entrypoint, Level string
			HeldBack                   string `json:"held_back"`
		} `json:"set"`
	}
	_ = json.Unmarshal(install.Result, &result)
	target := install.Selection.Target()
	if result.Package != "" && result.Release != "" {
		target = result.Package + "@" + result.Release
	}
	fields := []output.Field{{K: "id", V: install.ID}, {K: "rental", V: install.RentalID}, {K: "status", V: "succeeded"},
		{K: "target", V: target}, {K: "elapsed", V: time.Since(began).Round(time.Second).String()}}
	shown := []string{"id", "rental", "status", "target", "elapsed"}
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

// watchRentalInstall follows one accepted installation until it settles, saying on stderr what
// the machine reports. Interrupting detaches: the installation goes on, and its latest state is
// answered with true.
func watchRentalInstall(ctx *Context, client *localapi.Client, machine string, install records.RentalInstall) (records.RentalInstall, bool, *exit.Error) {
	interrupted, restore, _, problem := liveWatchContext(ctx, context.Background(), nil)
	if problem != nil {
		return install, false, problem
	}
	defer restore()
	watch := &installWatch{machine: machine, shownAt: time.Now()}
	for {
		status, problem := client.RentalInstall(install.RentalID, install.ID)
		if problem != nil {
			return install, false, problem
		}
		switch status.State {
		case "succeeded":
			return status.RentalInstall, false, nil
		case "failed":
			return install, false, exit.Named(exit.Failed, either(status.ErrorCode, "rental_install.failed"), "%s: %s", machine, status.Error)
		}
		stage, done, total := status.State, uint64(0), uint64(0)
		if p := status.Progress; p != nil {
			stage, done, total = p.Stage, p.TransferredBytes, p.TotalBytes
		}
		watch.report(ctx, stage, done, total)
		select {
		case <-interrupted.Done():
			if !ctx.Mode().JSON {
				fmt.Fprintf(ctx.Err, "\ndetached from installation %s; it continues on %s\n", install.ID, machine)
			}
			return status.RentalInstall, true, nil
		case <-time.After(200 * time.Millisecond):
		}
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
