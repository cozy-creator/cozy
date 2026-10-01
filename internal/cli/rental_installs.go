package cli

import (
	"fmt"
	"strings"
	"time"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

func handleRentalPackageInstall(ctx *Context) *exit.Error {
	adoptRentalHub(ctx, ctx.Inv.Value("--rental"))
	ref, plan, problem := resolveRegistryPackage(ctx, ctx.Inv.Args[0], ctx.Inv.Value("--version"))
	if problem != nil {
		return problem
	}
	// No model bindings are resolved or forwarded by package installation.
	return enqueueRentalInstall(ctx, ctx.Inv.Value("--rental"), records.RentalInstallSelection{Package: ref.String(), Release: plan.Release}, false)
}

func enqueueRentalInstall(ctx *Context, rentalName string, selection records.RentalInstallSelection, await bool) *exit.Error {
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

// awaitRentalInstall watches one accepted installation until it settles, saying on stderr
// what the machine reports. Interrupting stops only the watch; the installation goes on.
func awaitRentalInstall(ctx *Context, client *localapi.Client, machine string, install records.RentalInstall) *exit.Error {
	began := time.Now()
	lastStage, shownAt, shownBytes := "", began, uint64(0)
	for {
		status, problem := client.RentalInstall(install.RentalID, install.ID)
		if problem != nil {
			return problem
		}
		switch status.State {
		case "succeeded":
			fields := []output.Field{{K: "id", V: status.ID}, {K: "rental", V: status.RentalID}, {K: "status", V: status.State},
				{K: "target", V: rentalInstallTarget(status.Selection)}, {K: "elapsed", V: time.Since(began).Round(time.Second).String()}}
			return emit(ctx, compactRecord(fields, "id", "rental", "status", "target", "elapsed"))
		case "failed":
			return exit.Named(exit.Failed, either(status.ErrorCode, "rental_install.failed"), "%s: %s", machine, status.Error)
		}
		stage, done, total := status.State, uint64(0), uint64(0)
		if p := status.Progress; p != nil {
			stage, done, total = p.Stage, p.TransferredBytes, p.TotalBytes
		}
		if stage != lastStage || done > shownBytes && time.Since(shownAt) >= 10*time.Second {
			line := machine + ": " + stage
			if total > 0 {
				line += fmt.Sprintf(" %s of %s (%d%%)", output.Bytes(int64(done)), output.Bytes(int64(total)), done*100/total)
			}
			if stage == lastStage {
				line += fmt.Sprintf(", %s/s", output.Bytes(int64(float64(done-shownBytes)/time.Since(shownAt).Seconds())))
			}
			_ = output.Progress(ctx.Err, line)
			lastStage, shownAt, shownBytes = stage, time.Now(), done
		}
		time.Sleep(time.Second)
	}
}

func rentalInstallTarget(selection records.RentalInstallSelection) string {
	if selection.Package != "" {
		return selection.Package + "@" + selection.Release
	}
	models := make([]string, 0, len(selection.Models))
	for _, model := range selection.Models {
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
