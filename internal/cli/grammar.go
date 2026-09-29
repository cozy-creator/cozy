package cli

import "strconv"

// CLI is the complete public command grammar. Kong derives parsing and help from
// this tree; there is no parallel command manifest or string handler registry.
type CLI struct {
	Tensorhub *string  `help:"Use this hub (a name from cozy hub list, or a URL) for this command only." placeholder:"HUB"`
	JSON      bool     `help:"Emit JSON instead of human-readable output."`
	Full      bool     `help:"Include complete values and all available fields."`
	Fields    []string `help:"Select result fields." sep:","`

	Package PackageCmd `cmd:"" group:"Packages" help:"Install the source-code that generates media."`
	Model   ModelCmd   `cmd:"" group:"Models" help:"Download the tensors that are the AI's mind."`
	Auth    AuthCmd    `cmd:"" group:"Authentication" help:"Authenticate this machine to Tensorhub."`
	Hub     HubCmd     `cmd:"" group:"Authentication" help:"Choose which Tensorhub commands use; one daemon serves them all."`
	Run     RunCmd     `cmd:"" group:"Runs" help:"Run a package function on a local or rented machine."`
	Rental  RentalCmd  `cmd:"" group:"Rentals" help:"Rent a more powerful GPU in the cloud (alias: rent)."`
	Machine MachineCmd `cmd:"" group:"Rentals" help:"This computer as a machine: the same Host a rental runs."`
	Rent    RentalCmd  `cmd:"" hidden:"" help:"Alias of cozy rental."`
	Cache   CacheCmd   `cmd:"" group:"Lifecycle" help:"Manage cached operation results on this machine."`
	Up      UpCmd      `cmd:"" group:"Lifecycle" help:"Start the cozy-daemon and localhost web-ui."`
	Down    DownCmd    `cmd:"" group:"Lifecycle" help:"Stop cozy-daemon and localhost web-ui."`
	Daemon  DaemonCmd  `cmd:"" group:"Lifecycle" help:"Read the cozy-daemon's own log."`

	Completion CompletionCmd `cmd:"" group:"Lifecycle" help:"Print a bash, zsh or fish tab-completion script."`
}

type MachineCmd struct {
	Install MachineInstallCmd `cmd:"" help:"Install a worker cohort's Host and Runtime as this computer's machine."`
	Show    MachineShowCmd    `cmd:"" help:"Show this computer's machine."`
	Stop    MachineStopCmd    `cmd:"" help:"Stop this computer's machine Host; the next local run starts it again."`
}

type MachineInstallCmd struct {
	Host          string `name:"host" predictor:"file" help:"Development only: install this cozy-machine executable; default: the published machine agent."`
	RuntimeWheel  string `name:"runtime-wheel" predictor:"file" help:"The cohort's Runtime wheel; default: the published Runtime."`
	TensorFSWheel string `name:"tensorfs-wheel" predictor:"file" help:"The cohort's TensorFS wheel, with --runtime-wheel."`
}

func (c *MachineInstallCmd) Run(r *Runtime) error {
	return r.call(handleMachineInstall, nil, nil, values("--host", c.Host, "--runtime-wheel", c.RuntimeWheel, "--tensorfs-wheel", c.TensorFSWheel), false)
}

type MachineShowCmd struct{}

func (c *MachineShowCmd) Run(r *Runtime) error {
	return r.call(handleMachineShow, nil, nil, nil, false)
}

type MachineStopCmd struct{}

func (c *MachineStopCmd) Run(r *Runtime) error {
	return r.call(handleMachineStop, nil, nil, nil, false)
}

type DaemonCmd struct {
	Log DaemonLogCmd `cmd:"" help:"Print the cozy-daemon log ($COZY_HOME/daemon.log)."`
}

type CacheCmd struct {
	Prune CachePruneCmd `cmd:"" help:"Free unused cached operation results on this machine."`
}

type CachePruneCmd struct{}

func (c *CachePruneCmd) Run(r *Runtime) error {
	return r.call(handleCachePrune, nil, nil, nil, true)
}

type DaemonLogCmd struct {
	Follow bool `short:"f" help:"Keep printing as the daemon writes."`
}

func (c *DaemonLogCmd) Run(r *Runtime) error {
	return r.call(handleDaemonLog, nil, bools("--follow", c.Follow), nil, false)
}

type AuthCmd struct {
	Login               AuthLoginCmd               `cmd:"" help:"Register or authenticate this machine by email."`
	Logout              AuthLogoutCmd              `cmd:"" help:"Revoke this machine and erase its local key."`
	RevokeOtherMachines AuthRevokeOtherMachinesCmd `cmd:"" help:"Revoke every other machine after email verification."`
	Current             AuthCurrent                `cmd:"" default:"1" hidden:""`
}

type AuthCurrent struct{}

func (c *AuthCurrent) Run(r *Runtime) error {
	return r.call(handleAuthStatus, nil, nil, nil, false)
}

type AuthLoginCmd struct {
	Email string `arg:"" name:"email" help:"Tensorhub account email."`
}

func (c *AuthLoginCmd) Run(r *Runtime) error {
	return r.call(handleAuthLogin, []string{c.Email}, nil, nil, false)
}

type AuthLogoutCmd struct{}

func (c *AuthLogoutCmd) Run(r *Runtime) error {
	return r.call(handleAuthLogout, nil, nil, nil, false)
}

type AuthRevokeOtherMachinesCmd struct{}

func (c *AuthRevokeOtherMachinesCmd) Run(r *Runtime) error {
	return r.call(handleAuthRevokeOtherMachines, nil, nil, nil, false)
}

type HubCmd struct {
	List   HubListCmd   `cmd:"" default:"1" help:"List named hubs and which one is current."`
	Use    HubUseCmd    `cmd:"" help:"Make a hub current for later commands; existing work keeps its own hub."`
	Add    HubAddCmd    `cmd:"" help:"Name a Tensorhub URL."`
	Remove HubRemoveCmd `cmd:"" help:"Forget a hub name."`
}

type HubListCmd struct{}

func (c *HubListCmd) Run(r *Runtime) error {
	return r.call(handleHubList, nil, nil, nil, false)
}

type HubUseCmd struct {
	Hub string `arg:"" name:"hub" help:"A hub name or Tensorhub URL."`
}

func (c *HubUseCmd) Run(r *Runtime) error {
	return r.call(handleHubUse, []string{c.Hub}, nil, nil, false)
}

type HubAddCmd struct {
	Name string `arg:"" name:"name" help:"Hub name, such as local."`
	URL  string `arg:"" name:"url" help:"Tensorhub base URL, such as http://127.0.0.1:8819."`
}

func (c *HubAddCmd) Run(r *Runtime) error {
	return r.call(handleHubAdd, []string{c.Name, c.URL}, nil, nil, false)
}

type HubRemoveCmd struct {
	Name string `arg:"" name:"name" help:"Hub name."`
}

func (c *HubRemoveCmd) Run(r *Runtime) error {
	return r.call(handleHubRemove, []string{c.Name}, nil, nil, false)
}

type PackageCmd struct {
	UpdateAll PackageUpdateAllCmd `cmd:"" help:"Update installed published packages to newer releases without downloading model weights; local and development installs are skipped."`
	Search    PackageSearchCmd    `cmd:"" help:"Search for AI magic."`
	Install   PackageInstallCmd   `cmd:"" help:"Install a published package or explicit local directory."`
	Recover   PackageRecoverCmd   `cmd:"" help:"Repair package inventory from an explicit Creator database backup." hidden:""`
	Remove    PackageRemoveCmd    `cmd:"" help:"Delete source-code."`
	List      PackageListCmd      `cmd:"" help:"List installed packages."`
	Info      PackageInfoCmd      `cmd:"" help:"Show a published package's releases."`
	Lock      PackageLockCmd      `cmd:"" help:"Lock this package's dependencies for the current Tensorhub and account."`
	Publish   PackagePublishCmd   `cmd:"" help:"Publish a package release."`
	Yank      PackageYankCmd      `cmd:"" help:"Permanently yank a package release."`

	Bind     PackageBindCmd     `cmd:"" help:"Set an owner model override for one package slot."`
	Unbind   PackageUnbindCmd   `cmd:"" help:"Remove an owner override and use the package default."`
	Bindings PackageBindingsCmd `cmd:"" help:"Show each model slot's default ladder, owner override, and which one runs."`
}

type PackageBindCmd struct {
	Ref  string   `arg:"" name:"package" help:"Published package name (org/name)."`
	Slot string   `arg:"" name:"slot-path" help:"Declared slot path, e.g. generate.models.model."`
	To   string   `arg:"" name:"model" help:"org/model@release — the model release the slot loads."`
	GPU  []string `name:"gpu" help:"<GPU>=<lane>: the lane that fits that GPU class; repeat in order of preference, '*'=<lane> last as the catch-all."`
}

func (c *PackageBindCmd) Run(r *Runtime) error {
	return r.call(handlePackageBind, []string{c.Ref, c.Slot, c.To}, nil, values("--gpu", c.GPU), false)
}

type PackageUnbindCmd struct {
	Ref  string `arg:"" name:"package" help:"Published package name (org/name)."`
	Slot string `arg:"" name:"slot-path" help:"Model slot whose owner override to remove."`
}

func (c *PackageUnbindCmd) Run(r *Runtime) error {
	return r.call(handlePackageUnbind, []string{c.Ref, c.Slot}, nil, nil, false)
}

type PackageBindingsCmd struct {
	Ref     string `arg:"" name:"package" help:"Published package name (org/name)."`
	Version string `help:"Package release whose authored defaults to show instead of the newest, e.g. 1.2.3."`
}

func (c *PackageBindingsCmd) Run(r *Runtime) error {
	return r.call(handlePackageBindings, []string{c.Ref}, nil, values("--version", c.Version), false)
}

type PackageSearchCmd struct {
	Query []string `arg:"" optional:"" name:"query" help:"Search text or an exact org/name."`
	Limit int      `help:"Maximum results." default:"20"`
}

func (c *PackageSearchCmd) Run(r *Runtime) error {
	return r.call(handlePackageSearch, c.Query, nil,
		values("--limit", intText(c.Limit)), false)
}

type PackageInstallCmd struct {
	Ref      string `arg:"" name:"package-or-directory" predictor:"dir-or-ref" help:"Published org/name or explicit directory such as . or ./project."`
	Version  string `help:"Install this release instead of the newest, e.g. 1.2.3."`
	Editable bool   `help:"Keep an explicit local directory live for development."`
	Rental   string `predictor:"rental" help:"Install published code and dependencies on this existing rental name or id, without model weights."`
}

type PackageUpdateAllCmd struct{}

func (c *PackageUpdateAllCmd) Run(r *Runtime) error {
	return r.call(handlePackageUpdateAll, nil, nil, nil, false)
}

type PackageRecoverCmd struct {
	Database string `arg:"" name:"database" help:"Explicit prior Creator records database." type:"path"`
}

type PackageInfoCmd struct {
	Ref string `arg:"" name:"package" help:"Published package name (org/name)."`
}

func (c *PackageInfoCmd) Run(r *Runtime) error {
	return r.call(handlePackageInfo, []string{c.Ref}, nil, nil, false)
}

type PackageYankCmd struct {
	Ref     string `arg:"" name:"package" help:"Published package name (org/name)."`
	Version string `help:"Immutable N.M.P release to yank." required:""`
}

func (c *PackageYankCmd) Run(r *Runtime) error {
	return r.call(handlePackageYank, []string{c.Ref}, nil, values("--version", c.Version), false)
}

func (c *PackageInstallCmd) Run(r *Runtime) error {
	return r.call(handleInstall, []string{c.Ref}, bools("--editable", c.Editable),
		values("--version", c.Version, "--rental", c.Rental), false)
}

func (c *PackageRecoverCmd) Run(r *Runtime) error {
	return r.call(handlePackageRecover, []string{c.Database}, nil, nil, false)
}

type PackageRemoveCmd struct {
	Refs []string `arg:"" name:"package" predictor:"package" help:"Installed package ref."`
}

func (c *PackageRemoveCmd) Run(r *Runtime) error {
	// The daemon is ensured inside the verb, after the removal: a daemon started here
	// sweeps unreferenced installs on boot, and the verb would then find nothing to
	// report for a package whose pin was already gone.
	return r.call(handleRm, c.Refs, nil, nil, false)
}

type PackageListCmd struct{}

func (c *PackageListCmd) Run(r *Runtime) error {
	return r.call(handleLs, nil, nil, nil, false)
}

type PackageLockCmd struct {
	UpgradePackage []string `name:"upgrade-package" help:"Allow this dependency to move to its newest compatible version; repeat for each."`
}

func (c *PackageLockCmd) Run(r *Runtime) error {
	return r.call(handlePackageLock, nil, nil, map[string][]string{"--upgrade-package": c.UpgradePackage}, false)
}

type PackagePublishCmd struct {
	Wheel []string `name:"wheel" help:"Publish this prebuilt project wheel instead of building one; repeat for each Python/platform variant." type:"path"`
}

func (c *PackagePublishCmd) Run(r *Runtime) error {
	return r.call(handlePackagePublish, nil, nil, map[string][]string{"--wheel": c.Wheel}, false)
}

type ModelCmd struct {
	Info     ModelInfoCmd     `cmd:"" help:"Show all releases, lane sizes, and exact checkpoint refs."`
	Search   ModelSearchCmd   `cmd:"" help:"Search models, showing each model's latest available release."`
	Family   ModelFamilyCmd   `cmd:"" help:"Set a model repository's discovery family."`
	Download ModelDownloadCmd `cmd:"" help:"Download a model into a local destination or an owned rental store."`
	Remove   ModelRemoveCmd   `cmd:"" help:"Remove local model repositories and reclaim their bytes."`
	GC       ModelGCCmd       `cmd:"" name:"gc" help:"Reclaim the bytes no local model references."`
	List     ModelListCmd     `cmd:"" help:"List local model releases."`
	Upload   ModelUploadCmd   `cmd:"" help:"Ingest a model with its required metadata and upload an owner-only checkpoint."`
	Publish  ModelPublishCmd  `cmd:"" help:"Update a release's mutable lane pointers."`
	Retarget ModelRetargetCmd `cmd:"" help:"Move one existing release lane to another retained checkpoint."`
	Yank     ModelYankCmd     `cmd:"" help:"Yank a model release."`
	Delete   ModelDeleteCmd   `cmd:"" help:"Delete a Tensorhub model whose releases are yanked, or one unreleased checkpoint."`
}

type ModelInfoCmd struct {
	Model string `arg:"" name:"model" help:"Model repository as org/name, optionally followed by @release."`
}

func (c *ModelInfoCmd) Run(r *Runtime) error {
	return r.call(handleModelInfo, []string{c.Model}, nil, nil, false)
}

type ModelSearchCmd struct {
	Query  []string `arg:"" optional:"" name:"query" help:"Search text or an exact org/name."`
	Limit  int      `help:"Maximum matching models; one row per model." default:"20"`
	Family string   `help:"Only show one recognized model family."`
}

func (c *ModelSearchCmd) Run(r *Runtime) error {
	return r.call(handleModelSearch, c.Query, nil,
		values("--limit", intText(c.Limit), "--family", c.Family), false)
}

type ModelFamilyCmd struct {
	Model  string `arg:"" name:"model" help:"Model repository as org/name."`
	Family string `arg:"" optional:"" name:"family" help:"Recognized family value."`
	Clear  bool   `help:"Clear the repository family."`
}

func (c *ModelFamilyCmd) Run(r *Runtime) error {
	return r.call(handleModelFamily, []string{c.Model}, bools("--clear", c.Clear),
		values("--family", c.Family), true)
}

type ModelDownloadCmd struct {
	Source         string `arg:"" name:"source" predictor:"file-or-ref" help:"Provider source, Tensorhub model (#lane or @release/lane), local alias, or explicit local file."`
	Ref            string `arg:"" optional:"" name:"model" help:"Local destination (local/name); omit it to download into this computer's machine."`
	Lane           string `help:"Select an input lane when source is a Tensorhub model release."`
	Rental         string `help:"Download directly into this owned rental (no local destination)."`
	RentalOnly     bool   `help:"Require a remote rental instead of local capacity."`
	Await          bool   `help:"Watch the accepted run until it settles."`
	IdempotencyKey string `help:"Stable request identity for exact replay; otherwise start a new run."`
}

func (c *ModelDownloadCmd) Run(r *Runtime) error {
	return r.call(handleModelDownload, []string{c.Source, c.Ref}, bools(
		"--rental-only", c.RentalOnly, "--await", c.Await),
		values("--rental", c.Rental, "--lane", c.Lane, "--idempotency-key", c.IdempotencyKey), false)
}

type ModelRemoveCmd struct {
	Refs []string `arg:"" name:"model" help:"Local model repository name."`
}

func (c *ModelRemoveCmd) Run(r *Runtime) error {
	return r.call(handleModelRemove, c.Refs, nil, nil, false)
}

type ModelGCCmd struct{}

func (c *ModelGCCmd) Run(r *Runtime) error {
	return r.call(handleModelGC, nil, nil, nil, false)
}

type ModelListCmd struct{}

func (c *ModelListCmd) Run(r *Runtime) error {
	return r.call(handleModelList, nil, nil, nil, false)
}

type ModelUploadCmd struct {
	Source         string   `arg:"" name:"source" predictor:"file-or-ref" help:"Provider source, Tensorhub model (#lane or @release/lane), local alias, or explicit local file."`
	Ref            string   `arg:"" name:"model" help:"Tensorhub destination (org/name)."`
	Lane           string   `help:"Select an input lane when source is a Tensorhub model release."`
	Rental         *string  `predictor:"rental" help:"Use this existing rental name or id; never buy a replacement."`
	RentalOnly     bool     `help:"Require a remote rental instead of local capacity."`
	Await          bool     `help:"Watch the accepted run until it settles."`
	IdempotencyKey string   `help:"Stable request identity for exact replay; otherwise a rented re-run of the same ingest reattaches or resumes."`
	SourceProfiles []string `name:"source-profile" help:"Reviewed TensorFS source profile to convert on the rental (repeatable; several compose one model). Default: the one profile the source headers match."`
}

func (c *ModelUploadCmd) Run(r *Runtime) error {
	rentalName, problem := rentalArgument(c.Rental)
	if problem != nil {
		return problem
	}
	return r.call(handleModelUpload, []string{c.Source, c.Ref}, bools(
		"--rental-only", c.RentalOnly, "--await", c.Await),
		values("--rental", rentalName, "--lane", c.Lane, "--idempotency-key", c.IdempotencyKey,
			"--source-profile", c.SourceProfiles), false)
}

type ModelPublishCmd struct {
	Ref        string   `arg:"" name:"model" help:"Tensorhub model repository (org/name)."`
	Release    string   `help:"Mutable release label." required:""`
	Lanes      []string `name:"lane" help:"Set a lane pointer as name=checkpoint-id."`
	RemoveLane []string `help:"Remove a lane pointer by name; yank the release instead of removing its last lane."`
}

func (c *ModelPublishCmd) Run(r *Runtime) error {
	return r.call(handleModelPublish, []string{c.Ref}, nil,
		values("--release", c.Release, "--lane", c.Lanes, "--remove-lane", c.RemoveLane), false)
}

type ModelRetargetCmd struct {
	Ref     string `arg:"" name:"model" help:"Tensorhub model repository (org/name)."`
	Release string `help:"Mutable release label." required:""`
	Lane    string `help:"Existing lane to move; retarget never cuts one." required:""`
	To      string `help:"Retained checkpoint id (sha256:<64 hex>)." required:""`
}

func (c *ModelRetargetCmd) Run(r *Runtime) error {
	return r.call(handleModelRetarget, []string{c.Ref}, nil,
		values("--release", c.Release, "--lane", c.Lane, "--to", c.To), false)
}

type ModelYankCmd struct {
	Ref     string `arg:"" name:"model" help:"Tensorhub model repository (org/name)."`
	Release string `help:"Release label to yank." required:""`
}

func (c *ModelYankCmd) Run(r *Runtime) error {
	return r.call(handleModelYank, []string{c.Ref}, nil, values("--release", c.Release), false)
}

type ModelDeleteCmd struct {
	Ref string `arg:"" name:"model" help:"Tensorhub model (org/name), or one unreleased checkpoint (org/name#sha256:<checkpoint>)."`
	Yes bool   `help:"Confirm the deletion; it cannot be undone."`
}

func (c *ModelDeleteCmd) Run(r *Runtime) error {
	return r.call(handleModelDelete, []string{c.Ref}, bools("--yes", c.Yes), nil, false)
}

type RunCmd struct {
	Upload  RunUploadCmd  `cmd:"" help:"Upload a run's retained output, from the rental holding it, as a private checkpoint in org/model without running it again."`
	Execute RunExecuteCmd `cmd:"" default:"withargs" hidden:""`
	Cancel  RunCancelCmd  `cmd:"" help:"Cancel a queued or running run."`
	Pause   RunPauseCmd   `cmd:"" help:"Pause a private transaction and retain its work. Rentals still shut down after 15 idle minutes; cozy rental keepalive <name> resets the deadline once."`
	Resume  RunResumeCmd  `cmd:"" help:"Resume a paused transaction from its captured code and retained work."`
	List    RunListCmd    `cmd:"" help:"List current and past runs."`
	Watch   RunWatchCmd   `cmd:"" help:"Watch one recorded run until it settles."`
	Play    RunPlayCmd    `cmd:"" help:"Print a link that plays a run's output in any browser, live as it grows, straight from its rented machine."`
	Show    RunShowCmd    `cmd:"" help:"Show one run's execution evidence: setup and inference stages, per-step times, the GPUs it ran on and each GPU's attention kernels (served, or why not, and compile time), and each child call's function, label, GPUs, stages and steps."`
}

type RunExecuteCmd struct {
	Target          string   `arg:"" name:"target" predictor:"callable" help:"Package callable org/package[/function], or a single-entrypoint Python script."`
	Input           []string `arg:"" optional:"" name:"input" help:"Primary value (a conversion job takes <input-model> [<org/model> destination]), field=value payload, model.<param>=reference overrides (Tensorhub, hf://, or civitai://), and kernel.attention=[component=]backend for a request-scoped development override."`
	Out             string   `help:"Output directory." type:"path" predictor:"dir"`
	Timeout         string   `help:"Request deadline."`
	PayloadFile     string   `name:"input" help:"Read the whole payload from a JSON file, e.g. --input=request.json; inline fields override file values." type:"path"`
	Assets          []string `name:"asset" predictor:"binding-file" help:"Attach a file or label=file to a declared Assets input; field-path=file binds a named payload asset."`
	AssetFidelity   []string `name:"asset-fidelity" help:"Set a declared asset hint as label-or-index=auto|low|medium|high (repeatable)."`
	LoRAs           []string `name:"lora" sep:"none" help:"Apply an ordered LoRA as model-parameter:component=reference[,strength] (repeatable)."`
	AttentionKernel string   `name:"attention-kernel" help:"Development override for this request: backend (all sites) or [model/]component=backend. Example: model/fl2va_dit=kitchen-int8. No fallback; Runtime validates hardware, compiled mode and parallelism."`
	Rental          *string  `predictor:"rental" help:"Run only on this existing rental name or id; never buy a replacement."`
	RentNew         bool     `help:"Buy a fresh managed rental for this run; do not reuse existing machines."`
	RentalOnly      bool     `help:"Require a remote rental even when local capacity is ready."`
	IdempotencyKey  string   `help:"Stable request identity for safe retries."`
	Retry           string   `help:"Retry with current code while retaining compatible work from this prior run."`
	Trees           []string `name:"input-tree" predictor:"binding-dir" help:"Bind a job input tree as ref=directory."`
	Org             string   `help:"Job publication organization (defaults to local)."`
	UploadTo        string   `help:"Upload the job's declared weight outputs as private checkpoints to org/model; no release is published."`
	AllowUpload     []string `help:"Allow this rented transaction to upload checkpoints, and publish releases, only to org/model (repeatable)."`
	SourceProfiles  []string `name:"source-profile" help:"Map a foreign model input to a reviewed TensorFS source profile as slot=profile (repeatable)."`
	Await           bool     `help:"Show progress and wait for the result; --json writes JSONL events to stderr and one result to stdout."`
	Describe        bool     `help:"Print the callable's request contract instead of running it."`
}

func (c *RunExecuteCmd) Run(r *Runtime) error {
	rentalName, problem := rentalArgument(c.Rental)
	if problem != nil {
		return problem
	}
	args := append([]string{c.Target}, c.Input...)
	return r.call(handleRunExecute, args, bools(
		"--await", c.Await,
		"--rental-only", c.RentalOnly, "--rent-new", c.RentNew, "--describe", c.Describe), values(
		"--rental", rentalName, "--out", c.Out, "--timeout", c.Timeout,
		"--attention-kernel", c.AttentionKernel, "--lora", c.LoRAs,
		"--in", c.PayloadFile, "--asset", c.Assets, "--asset-fidelity", c.AssetFidelity,
		"--idempotency-key", c.IdempotencyKey, "--retry", c.Retry, "--input", c.Trees, "--org", c.Org,
		"--upload-to", c.UploadTo, "--allow-upload", c.AllowUpload, "--source-profile", c.SourceProfiles), !c.Describe)
}

type RunPlayCmd struct {
	ID      string `arg:"" name:"run" help:"Run number or id."`
	Output  string `default:"video" help:"The output to play: its name, or name/index for one item of a list."`
	Expires string `default:"24h" help:"How long the link works; anyone holding it may watch this output until then."`
}

func (c *RunPlayCmd) Run(r *Runtime) error {
	return r.call(handleRunPlay, []string{c.ID}, nil, values("--output", c.Output, "--expires", c.Expires), true)
}

type RunUploadCmd struct {
	Output      string `arg:"" name:"run[#output]" help:"Run number or id; name the output when the run retains more than one."`
	Destination string `arg:"" name:"model" help:"Tensorhub model repository (org/name) for the private checkpoint."`
	Await       bool   `help:"Wait until the checkpoint is uploaded."`
}

func (c *RunUploadCmd) Run(r *Runtime) error {
	return r.call(handleRunUpload, []string{c.Output, c.Destination}, bools("--await", c.Await), nil, true)
}

type RunCancelCmd struct {
	ID string `arg:"" name:"run" help:"Run id."`
}

type RunPauseCmd struct {
	ID string `arg:"" name:"run" help:"Run id."`
}

func (c *RunPauseCmd) Run(r *Runtime) error {
	return r.call(handleRunPause, []string{c.ID}, nil, nil, true)
}

type RunResumeCmd struct {
	ID string `arg:"" name:"run" help:"Run id."`
}

func (c *RunResumeCmd) Run(r *Runtime) error {
	return r.call(handleRunResume, []string{c.ID}, nil, nil, true)
}

func (c *RunCancelCmd) Run(r *Runtime) error {
	return r.call(handleRunCancel, []string{c.ID}, nil, nil, true)
}

type RunListCmd struct {
	AllHubs bool   `help:"List runs on every hub, not only the current one."`
	State   string `help:"Filter by lifecycle state."`
	Package string `predictor:"package" help:"Filter by package."`
	Limit   *int   `help:"Maximum runs; snapshots default to 50, 0 reads all history. Live lists load more while scrolling."`
	Watch   bool   `help:"Refresh continuously (requires a terminal)."`
	NoWatch bool   `help:"Print one snapshot even in a terminal."`
}

func (c *RunListCmd) Run(r *Runtime) error {
	limit := ""
	if c.Limit != nil {
		limit = strconv.Itoa(*c.Limit)
	}
	return r.call(handleRunList, nil, bools("--watch", c.Watch, "--no-watch", c.NoWatch, "--all-hubs", c.AllHubs), values(
		"--state", c.State, "--package", c.Package, "--limit", limit), true)
}

type RunWatchCmd struct {
	ID string `arg:"" name:"run" help:"Run id."`
}

func (c *RunWatchCmd) Run(r *Runtime) error {
	return r.call(handleRunWatch, []string{c.ID}, nil, nil, true)
}

type RunShowCmd struct {
	ID   string `arg:"" name:"run" help:"Run number or id."`
	Call string `help:"Show one of the run's calls: its number, label, function or call id."`
}

func (c *RunShowCmd) Run(r *Runtime) error {
	return r.call(handleRunShow, []string{c.ID}, nil, values("--call", c.Call), true)
}

// RentalCmd has no default subcommand: bare `cozy rental` prints its verbs, the way
// bare `cozy package` and `cozy model` do.
type RentalCmd struct {
	Keepalive RentalKeepaliveCmd `cmd:"" help:"Explicitly reset this rental's fixed 15-minute idle deadline once."`
	Update    RentalUpdateCmd    `cmd:"" help:"Update this private rental's Runtime while retaining its files; active work is never interrupted."`
	SSHInfo   RentalSSHInfoCmd   `cmd:"" name:"ssh-info" help:"Read the current SSH endpoint of an attached development rental."`
	List      RentalListCmd      `cmd:"" help:"List rented machines, live on a terminal."`
	Show      RentalShowCmd      `cmd:"" help:"Show one rental: state, rate, accrued spend and every other fact."`
	Logs      RentalLogsCmd      `cmd:"" help:"Print the provider's boot log of a rental's attempt, kept after its pod is gone."`
	New       RentalNewCmd       `cmd:"" help:"Start a private rental, or list available machine types."`
	End       RentalEndCmd       `cmd:"" help:"End a private rental and stop billing."`
	Prune     RentalPruneCmd     `cmd:"" help:"Free unused cached operation results on a private rental."`
}

type RentalUpdateCmd struct {
	Rental        string `arg:"" predictor:"rental" help:"Existing rental name or id."`
	RuntimeWheel  string `name:"runtime-wheel" predictor:"file" help:"Development only: install this local native Runtime wheel on your private rental."`
	TensorFSWheel string `name:"tensorfs-wheel" predictor:"file" help:"Development only: pair --runtime-wheel with this local native TensorFS wheel."`
	Runtime       string `name:"runtime-version" help:"Install this published Runtime release; with no wheel or version, the newest pair."`
	TensorFS      string `name:"tensorfs-version" help:"Install this published TensorFS release."`
}

func (c *RentalUpdateCmd) Run(r *Runtime) error {
	return r.call(handleRentalUpdate, []string{c.Rental}, nil, values("--runtime-wheel", c.RuntimeWheel, "--tensorfs-wheel", c.TensorFSWheel,
		"--runtime-version", c.Runtime, "--tensorfs-version", c.TensorFS), false)
}

type RentalNewCmd struct {
	Development    *bool    `help:"Enable SSH maintenance access on the selected worker image (default); false disables it."`
	SSHPublicKey   string   `name:"ssh-public-key" predictor:"file" help:"SSH public-key file for this development rental."`
	Image          string   `name:"image" help:"Boot this hub-registered worker image (tag, digest, or kind: cuda, cpu-torch, cpu) instead of the machine's default."`
	SKU            string   `arg:"" optional:"" name:"machine-slug" help:"Machine type from the rental catalog, such as h100-sxm5-80gb."`
	GPUs           int      `name:"gpus" default:"1" help:"GPUs on the machine; any count the catalog lists. Keep to an even count for parallelism."`
	Models         []string `name:"model" help:"Size disk for a Hub model (org/model@release/lane) or a provider source to ingest (hf://org/repo@commit, civitai://version); repeat for several."`
	SourceProfiles []string `name:"source-profile" help:"Reviewed TensorFS source profile of the one --model provider source (repeatable; several compose one model)."`
	DiskGB         int      `name:"disk-gb" help:"Container disk to rent, in GB; the Hub picks an offer whose disk allows it."`
	IdempotencyKey string   `help:"Stable paid-operation identity."`
	Timeout        string   `help:"Caller wait deadline; does not release the rental."`
}

type RentalPruneCmd struct {
	ID string `arg:"" name:"rental" predictor:"rental" help:"Rental machine name or id."`
}

func (c *RentalPruneCmd) Run(r *Runtime) error {
	return r.call(handleRentalPrune, []string{c.ID}, nil, nil, true)
}

func (c *RentalNewCmd) Run(r *Runtime) error {
	flags := map[string]bool{}
	if c.Development != nil {
		flags["--development"] = *c.Development
	}
	gpus := ""
	if c.GPUs != 1 {
		gpus = strconv.Itoa(c.GPUs)
	}
	disk := ""
	if c.DiskGB != 0 {
		disk = strconv.Itoa(c.DiskGB)
	}
	return r.call(handleRent, []string{c.SKU}, flags, values(
		"--gpus", gpus, "--idempotency-key", c.IdempotencyKey, "--timeout", c.Timeout, "--model", c.Models,
		"--source-profile", c.SourceProfiles, "--disk-gb", disk, "--ssh-public-key", c.SSHPublicKey, "--image", c.Image), false)
}

type RentalEndCmd struct {
	ID string `arg:"" name:"rental" predictor:"rental" help:"Private rental id."`
}

func (c *RentalEndCmd) Run(r *Runtime) error {
	return r.call(handleRentRelease, []string{c.ID}, nil, nil, true)
}

type RentalListCmd struct {
	AllHubs bool `help:"List rentals on every hub, not only the current one."`
	Watch   bool `help:"Refresh continuously (requires a terminal)."`
	NoWatch bool `help:"Print one snapshot even in a terminal."`
}

func (c *RentalListCmd) Run(r *Runtime) error {
	return r.call(handleRentalList, nil, bools("--watch", c.Watch, "--no-watch", c.NoWatch, "--all-hubs", c.AllHubs), nil, false)
}

type RentalShowCmd struct {
	Rental string `arg:"" name:"rental" predictor:"rental" help:"Rental machine name or id."`
}

func (c *RentalShowCmd) Run(r *Runtime) error {
	return r.call(handleRentalShow, []string{c.Rental}, nil, nil, false)
}

type RentalLogsCmd struct {
	Rental  string `arg:"" name:"rental" predictor:"rental" help:"Rental machine name or id."`
	Follow  bool   `short:"f" help:"Keep printing while the rental boots, onto any attempt a replan starts."`
	Attempt int    `help:"Attempt number, from 1; default the latest."`
}

func (c *RentalLogsCmd) Run(r *Runtime) error {
	attempt := ""
	if c.Attempt != 0 {
		attempt = intText(c.Attempt)
	}
	return r.call(handleRentalLogs, []string{c.Rental}, bools("--follow", c.Follow), values("--attempt", attempt), false)
}

type UpCmd struct{}

func (c *UpCmd) Run(r *Runtime) error {
	return r.call(handleUp, nil, nil, nil, false)
}

type DownCmd struct {
	All   bool `help:"Cancel all work, end all rentals, then stop Cozy." xor:"down-mode"`
	Force bool `help:"Disconnect without canceling work or ending rentals; daemon-owned local work may be interrupted." xor:"down-mode"`
}

func (c *DownCmd) Run(r *Runtime) error {
	return r.call(handleDown, nil, bools("--all", c.All, "--force", c.Force), nil, false)
}

// SSHInfo reads current provider mapping from Hub; it stores no endpoint locally.
type RentalSSHInfoCmd struct {
	Rental string `arg:"" name:"rental" predictor:"rental" help:"Attached machine name or rental id."`
}

func (c *RentalSSHInfoCmd) Run(r *Runtime) error {
	return r.call(handleRentalSSHInfo, []string{c.Rental}, nil, nil, false)
}

type RentalKeepaliveCmd struct {
	Rental string `arg:"" predictor:"rental" help:"Existing rental name or id."`
}

func (c *RentalKeepaliveCmd) Run(r *Runtime) error {
	return r.call(handleRentalKeepalive, []string{c.Rental}, nil, nil, true)
}
