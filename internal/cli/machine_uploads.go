package cli

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/records"
)

// handleRunUpload is `cozy run upload <run>[#<output>] <org/model>`: a retained output
// goes from the rental holding it to a private checkpoint; no release is published.
func handleRunUpload(ctx *Context) *exit.Error {
	run, slot, _ := strings.Cut(ctx.Inv.Args[0], "#")
	ref, problem := hub.ParseRef(ctx.Inv.Args[1])
	if problem != nil {
		return problem
	}
	if taken, problem := uploadV1Output(ctx, run, slot, ref.String()); taken || problem != nil {
		return problem
	}
	return exit.Named(exit.NotFound, "run_output.absent", "run %s is not a cozy.machine.v1 run this host recorded", run).
		WithRemedy("run the job again with --upload-to %s", ref.String())
}

// uploadV1Output puts a cozy.machine.v1 run's weights output in destination: a warm run on the
// machine that holds it, queued as an installation. false: the run is not a v1 run.
func uploadV1Output(ctx *Context, run, slot, destination string) (bool, *exit.Error) {
	_, store, problem := rentalStores(ctx)
	if problem != nil {
		return true, problem
	}
	defer store.Close()
	request, problem := store.RequestByReference(run)
	if problem != nil || request == nil {
		return false, problem
	}
	if v1, problem := store.RunV1(request.ID); problem != nil || !v1 {
		return false, problem
	}
	link, problem := store.MachineExecution(request.ID)
	if problem != nil || link == nil {
		return true, problem
	}
	products, problem := store.Products(request.ID)
	if problem != nil {
		return true, problem
	}
	var output *records.Product
	for _, product := range records.Fold(products) {
		if product.MediaType == modelManifestMedia && (slot == "" || product.Output == slot) {
			if output != nil {
				return true, exit.Usagef("run %s has more than one weights output: name it as %s#<output>", run, run)
			}
			held := product
			output = &held
		}
	}
	if output == nil {
		return true, exit.Named(exit.NotFound, "run_output.absent", "run %s has no weights output %q", run, slot)
	}
	model := records.ModelRef{Slot: output.Output, Manifest: output.Digest, ManifestLength: output.Length}
	selection := records.RentalInstallSelection{Models: []records.ModelRef{model}, Destination: destination, Hub: request.Hub}
	if !machineendpoint.IsName(link.MachineID) {
		return true, enqueueRentalInstall(ctx, link.MachineID, selection, ctx.Inv.Bool("--await"))
	}
	// A foreground run's machine is the explicit endpoint it recorded; no daemon queue names it.
	ep, problem := store.RequestMachineEndpoint(request.ID, link.MachineID)
	if problem == nil && ep == nil {
		problem = exit.New(exit.NotFound, "run %s's machine endpoint is no longer recorded", run)
	}
	if problem != nil {
		return true, problem
	}
	label := link.MachineID
	if rented, _ := store.RentalRow(ep.WorkspaceID); rented != nil && rented.MachineName != "" {
		label = rented.MachineName
	}
	if taken, problem := foregroundInstall(ctx, ep, label, selection); taken || problem != nil {
		return true, problem
	}
	return true, exit.Named(exit.Structural, "machine.upload_unsupported", "the machine run %s ran on takes no model uploads", run)
}

// modelManifestMedia is a weights output's product: its bytes are the output's manifest.
const modelManifestMedia = "application/vnd.cozy.model-manifest"
