package cli

// CLI is the complete public command grammar. Kong derives parsing and help from
// this tree; there is no parallel command manifest or string handler registry.
type CLI struct {
	JSON   bool     `help:"Emit JSON instead of human-readable output."`
	Full   bool     `help:"Include complete values and all available fields."`
	Fields []string `help:"Select result fields." sep:","`

	Package PackageCmd `cmd:"" group:"Packages" help:"Install the source-code that generates media."`
	Model   ModelCmd   `cmd:"" group:"Models" help:"Download the tensors that are the AI's mind."`
	Auth    AuthCmd    `cmd:"" group:"Authentication" help:"Authenticate this machine to Tensorhub."`
	Run     RunCmd     `cmd:"" group:"Runs" help:"Run a package function on a local or rented machine."`
	Rental  RentalCmd  `cmd:"" group:"Rentals" help:"Rent a more powerful GPU in the cloud."`
	Cache   CacheCmd   `cmd:"" group:"Lifecycle" help:"Manage cached operation results on this machine."`
	Volume  VolumeCmd  `cmd:"" group:"Rentals" help:"Manage an optional repo-object cache in a datacenter you rent in."`
	Up      UpCmd      `cmd:"" group:"Lifecycle" help:"Start the cozy-daemon and localhost web-ui."`
	Down    DownCmd    `cmd:"" group:"Lifecycle" help:"Stop cozy-daemon and localhost web-ui."`
	Unload  UnloadCmd  `cmd:"" group:"Lifecycle" help:"Empty cached GPU AI models to free up VRAM."`
	Daemon  DaemonCmd  `cmd:"" group:"Lifecycle" help:"Read the cozy-daemon's own log."`
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

type PackageCmd struct {
	UpdateAll PackageUpdateAllCmd `cmd:"" help:"Update installed published packages to newer releases without downloading model weights; local and development installs are skipped."`
	Search    PackageSearchCmd    `cmd:"" help:"Search for AI magic."`
	Install   PackageInstallCmd   `cmd:"" help:"Install a published package or explicit local directory."`
	Recover   PackageRecoverCmd   `cmd:"" help:"Repair package inventory from an explicit Creator database backup." hidden:""`
	Remove    PackageRemoveCmd    `cmd:"" help:"Delete source-code."`
	List      PackageListCmd      `cmd:"" help:"List installed packages."`
	Publish   PackagePublishCmd   `cmd:"" help:"Publish a package release."`
	Yank      PackageYankCmd      `cmd:"" help:"Permanently yank a package release."`

	Bind     PackageBindCmd     `cmd:"" help:"Set an owner model override for one package slot."`
	Unbind   PackageUnbindCmd   `cmd:"" help:"Remove an owner override and use the package default."`
	Bindings PackageBindingsCmd `cmd:"" help:"Show the package owner's model overrides."`
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
	Ref string `arg:"" name:"package" help:"Published package name (org/name)."`
}

func (c *PackageBindingsCmd) Run(r *Runtime) error {
	return r.call(handlePackageBindings, []string{c.Ref}, nil, nil, false)
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
	Ref      string `arg:"" name:"package-or-directory" help:"Published org/name or explicit directory such as . or ./project."`
	Version  string `help:"Install this release instead of the newest, e.g. 1.2.3."`
	Editable bool   `help:"Keep an explicit local directory live for development."`
}

type PackageUpdateAllCmd struct{}

func (c *PackageUpdateAllCmd) Run(r *Runtime) error {
	return r.call(handlePackageUpdateAll, nil, nil, nil, false)
}

type PackageRecoverCmd struct {
	Database string `arg:"" name:"database" help:"Explicit prior Creator records database." type:"path"`
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
		values("--version", c.Version), false)
}

func (c *PackageRecoverCmd) Run(r *Runtime) error {
	return r.call(handlePackageRecover, []string{c.Database}, nil, nil, false)
}

type PackageRemoveCmd struct {
	Refs []string `arg:"" name:"package" help:"Installed package ref."`
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

type PackagePublishCmd struct{}

func (c *PackagePublishCmd) Run(r *Runtime) error {
	return r.call(handlePackagePublish, nil, nil, nil, false)
}

type ModelCmd struct {
	Info     ModelInfoCmd     `cmd:"" help:"Show all releases, lane sizes, and exact checkpoint refs."`
	Search   ModelSearchCmd   `cmd:"" help:"Search models, showing each model's latest available release."`
	Family   ModelFamilyCmd   `cmd:"" help:"Set a model repository's discovery family."`
	Download ModelDownloadCmd `cmd:"" help:"Acquire a source, optionally run one producer job, and retain it locally."`
	Remove   ModelRemoveCmd   `cmd:"" help:"Remove local model repositories and reclaim their bytes."`
	GC       ModelGCCmd       `cmd:"" name:"gc" help:"Reclaim the bytes no local model references."`
	List     ModelListCmd     `cmd:"" help:"List local model releases."`
	Upload   ModelUploadCmd   `cmd:"" help:"Acquire a source, optionally run one producer job, and retain owner-only checkpoints."`
	Publish  ModelPublishCmd  `cmd:"" help:"Update a release's mutable lane pointers."`
	Retarget ModelRetargetCmd `cmd:"" help:"Move one existing release lane to another retained checkpoint."`
	Yank     ModelYankCmd     `cmd:"" help:"Yank a model release."`
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
	Source         string `arg:"" name:"source" help:"Pinned provider source, Tensorhub release, local alias, or explicit local file."`
	Ref            string `arg:"" name:"model" help:"Local destination (local/name)."`
	Lane           string `help:"Select an input lane when source is a Tensorhub model release."`
	Rental         bool   `help:"Permit a managed rental when compatible local capacity is unavailable."`
	RentalOnly     bool   `help:"Require a remote rental instead of local capacity."`
	DryRun         bool   `help:"Resolve the exact transfer plan without moving bodies or spending."`
	Await          bool   `help:"Watch the accepted run until it settles."`
	IdempotencyKey string `help:"Stable request identity for exact replay; otherwise start a new run."`
}

func (c *ModelDownloadCmd) Run(r *Runtime) error {
	return r.call(handleModelDownload, []string{c.Source, c.Ref}, bools(
		"--rental", c.Rental, "--rental-only", c.RentalOnly,
		"--dry-run", c.DryRun, "--await", c.Await),
		values("--lane", c.Lane, "--idempotency-key", c.IdempotencyKey), false)
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
	Source         string `arg:"" name:"source" help:"Pinned provider source, Tensorhub release, local alias, or explicit local file."`
	Ref            string `arg:"" name:"model" help:"Tensorhub destination (org/name)."`
	Lane           string `help:"Select an input lane when source is a Tensorhub model release."`
	Rental         bool   `help:"Permit a managed rental when compatible local capacity is unavailable."`
	RentalOnly     bool   `help:"Require a remote rental instead of local capacity."`
	DryRun         bool   `help:"Resolve the exact transfer plan without moving bodies or spending."`
	Await          bool   `help:"Watch the accepted run until it settles."`
	IdempotencyKey string `help:"Stable request identity for exact replay; otherwise start a new run."`
}

func (c *ModelUploadCmd) Run(r *Runtime) error {
	return r.call(handleModelUpload, []string{c.Source, c.Ref}, bools(
		"--rental", c.Rental, "--rental-only", c.RentalOnly,
		"--dry-run", c.DryRun, "--await", c.Await),
		values("--lane", c.Lane, "--idempotency-key", c.IdempotencyKey), false)
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

type RunCmd struct {
	RetryPublication RunRetryPublicationCmd `cmd:"" help:"Retry a blocked model publication without rerunning its producer."`
	Execute          RunExecuteCmd          `cmd:"" default:"withargs" hidden:""`
	Cancel           RunCancelCmd           `cmd:"" help:"Cancel a queued or running run."`
	Pause            RunPauseCmd            `cmd:"" help:"Stop a private transaction while retaining its work and rental."`
	Resume           RunResumeCmd           `cmd:"" help:"Resume a paused transaction from its captured code and retained work."`
	List             RunListCmd             `cmd:"" help:"List current and past runs."`
	Watch            RunWatchCmd            `cmd:"" help:"Watch one recorded run until it settles."`
}

type RunExecuteCmd struct {
	Target          string   `arg:"" name:"target" help:"Package callable org/package[/function], or a single-entrypoint Python script."`
	Input           []string `arg:"" optional:"" name:"input" help:"Primary value, field=value payload, model.<param>=reference overrides (Tensorhub, hf://, or civitai://), and kernel.attention=[component=]backend for a request-scoped development override."`
	Out             string   `help:"Output directory." type:"path"`
	Timeout         string   `help:"Request deadline."`
	PayloadFile     string   `name:"input" aliases:"in" help:"Read the whole payload from a JSON file, e.g. --input=request.json; inline fields override file values." type:"path"`
	Assets          []string `name:"asset" help:"Attach a file or label=file to a declared Assets input; field-path=file binds a named payload asset."`
	AssetFidelity   []string `name:"asset-fidelity" help:"Set a declared asset hint as label-or-index=auto|low|medium|high (repeatable)."`
	LoRAs           []string `name:"lora" sep:"none" help:"Apply an ordered LoRA as model-parameter:component=reference[,strength] (repeatable)."`
	AttentionKernel string   `name:"attention-kernel" help:"Development override for this request: backend (all sites) or [model/]component=backend. Example: model/fl2va_dit=kitchen-int8. No fallback; Runtime validates hardware, compiled mode and parallelism."`
	Rental          *string  `help:"Run only on this existing rental name or id; never buy a replacement."`
	RentalOnly      bool     `help:"Require a remote rental even when local capacity is ready."`
	IdempotencyKey  string   `help:"Stable request identity for safe retries."`
	Retry           string   `help:"Retry with current code while retaining compatible work from this prior run."`
	Trees           []string `name:"input-tree" help:"Bind a job input tree as ref=directory."`
	Org             string   `help:"Job publication organization (defaults to local)."`
	PublishTo       string   `help:"Store the job's declared weight outputs as checkpoints in org/model; no release is created."`
	AllowPublish    []string `help:"Allow this rented transaction to publish only to org/model (repeatable)."`
	SourceProfiles  []string `name:"source-profile" help:"Map a foreign model input to a reviewed TensorFS source profile as slot=profile (repeatable)."`
	DryRun          bool     `help:"Resolve exact job inputs and conversion headers without queueing or renting."`
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
		"--rental-only", c.RentalOnly, "--describe", c.Describe, "--dry-run", c.DryRun), values(
		"--rental", rentalName, "--out", c.Out, "--timeout", c.Timeout,
		"--attention-kernel", c.AttentionKernel, "--lora", c.LoRAs,
		"--in", c.PayloadFile, "--asset", c.Assets, "--asset-fidelity", c.AssetFidelity,
		"--idempotency-key", c.IdempotencyKey, "--retry", c.Retry, "--input", c.Trees, "--org", c.Org,
		"--publish-to", c.PublishTo, "--allow-publish", c.AllowPublish, "--source-profile", c.SourceProfiles), !c.DryRun && !c.Describe)
}

type RunRetryPublicationCmd struct {
	ID string `arg:"" name:"run" help:"Run id with a retained failed model publication."`
}

func (c *RunRetryPublicationCmd) Run(r *Runtime) error {
	return r.call(handleRunRetryPublication, []string{c.ID}, nil, nil, true)
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
	State   string `help:"Filter by lifecycle state."`
	Package string `help:"Filter by package."`
	Limit   int    `help:"Maximum rows." default:"50"`
	Watch   bool   `help:"Refresh continuously (requires a terminal)."`
	NoWatch bool   `help:"Print one snapshot even in a terminal."`
}

func (c *RunListCmd) Run(r *Runtime) error {
	return r.call(handleRunList, nil, bools("--watch", c.Watch, "--no-watch", c.NoWatch), values(
		"--state", c.State, "--package", c.Package, "--limit", intText(c.Limit)), true)
}

type RunWatchCmd struct {
	ID string `arg:"" name:"run" help:"Run id."`
}

func (c *RunWatchCmd) Run(r *Runtime) error {
	return r.call(handleRunWatch, []string{c.ID}, nil, nil, true)
}

// RentalCmd has no default subcommand: bare `cozy rental` prints its verbs, the way
// bare `cozy package` and `cozy model` do.
type RentalCmd struct {
	Update  RentalUpdateCmd  `cmd:"" help:"Update this private rental's Runtime while retaining its files; active work is never interrupted."`
	Prepare RentalPrepareCmd `cmd:"" help:"Install one exact package release and prepare its model inputs on a rental."`
	SSHInfo RentalSSHInfoCmd `cmd:"" name:"ssh-info" help:"Read the current SSH endpoint of an attached development rental."`
	List    RentalListCmd    `cmd:"" help:"List rented machines, live on a terminal."`
	New     RentalNewCmd     `cmd:"" help:"Start a private rental, or list available machine types."`
	End     RentalEndCmd     `cmd:"" help:"End a private rental and stop billing."`
	Prune   RentalPruneCmd   `cmd:"" help:"Free unused cached operation results on a private rental."`
}

type RentalPrepareCmd struct {
	Rental  string   `arg:"" name:"rental" help:"Existing rental machine name or id."`
	Package string   `arg:"" name:"package" help:"Published package org/name."`
	Version string   `help:"Exact package release, e.g. 1.2.3." required:""`
	Models  []string `name:"model" help:"Exact model binding SLOT=org/model@release/lane; repeat for each slot."`
}

func (c *RentalPrepareCmd) Run(r *Runtime) error {
	return r.call(handleRentalPrepare, []string{c.Rental, c.Package}, nil,
		values("--version", c.Version, "--model", c.Models), false)
}

type RentalUpdateCmd struct {
	Rental string `arg:"" help:"Existing rental name or id."`
}

func (c *RentalUpdateCmd) Run(r *Runtime) error {
	return r.call(handleRentalUpdate, []string{c.Rental}, nil, nil, false)
}

type RentalNewCmd struct {
	Development    *bool    `help:"Enable SSH maintenance access on the selected worker image (default); false disables it."`
	SSHPublicKey   string   `name:"ssh-public-key" help:"SSH public-key file for this development rental."`
	SKU            string   `arg:"" optional:"" name:"machine-slug" help:"Machine type from the rental catalog, such as h100-sxm5-80gb."`
	Models         []string `name:"model" help:"Size disk for org/model@release/lane; repeat for several models. Hub measures their shared checkpoint closure."`
	IdempotencyKey string   `help:"Stable paid-operation identity."`
	Timeout        string   `help:"Caller wait deadline; does not release the rental."`
}

type RentalPruneCmd struct {
	ID string `arg:"" name:"rental" help:"Rental machine name or id."`
}

func (c *RentalPruneCmd) Run(r *Runtime) error {
	return r.call(handleRentalPrune, []string{c.ID}, nil, nil, true)
}

func (c *RentalNewCmd) Run(r *Runtime) error {
	flags := map[string]bool{}
	if c.Development != nil {
		flags["--development"] = *c.Development
	}
	return r.call(handleRent, []string{c.SKU}, flags, values(
		"--idempotency-key", c.IdempotencyKey, "--timeout", c.Timeout, "--model", c.Models, "--ssh-public-key", c.SSHPublicKey), false)
}

type RentalEndCmd struct {
	ID string `arg:"" name:"rental" help:"Private rental id."`
}

func (c *RentalEndCmd) Run(r *Runtime) error {
	return r.call(handleRentRelease, []string{c.ID}, nil, nil, true)
}

type RentalListCmd struct {
	Watch   bool `help:"Refresh continuously (requires a terminal)."`
	NoWatch bool `help:"Print one snapshot even in a terminal."`
}

func (c *RentalListCmd) Run(r *Runtime) error {
	return r.call(handleRentalList, nil, bools("--watch", c.Watch, "--no-watch", c.NoWatch), nil, false)
}

type VolumeCmd struct {
	Current VolumeListCmd `cmd:"" default:"1" hidden:""`
	Warm    VolumeWarmCmd `cmd:"" help:"Create an optional cache volume in a datacenter before renting there."`
	Drop    VolumeDropCmd `cmd:"" help:"Delete a disposable cache volume and its cached copies."`
}

type VolumeListCmd struct{}

func (c *VolumeListCmd) Run(r *Runtime) error {
	return r.call(handleVolumeLs, nil, nil, nil, false)
}

type VolumeWarmCmd struct {
	Datacenter string `arg:"" name:"datacenter" help:"Provider datacenter id, such as EU-RO-1."`
	Provider   string `help:"Provider name; the hub's primary when omitted."`
}

func (c *VolumeWarmCmd) Run(r *Runtime) error {
	return r.call(handleVolumeWarm, []string{c.Datacenter}, nil, values("--provider", c.Provider), false)
}

type VolumeDropCmd struct {
	Target string `arg:"" name:"volume" help:"Volume id (pvl-…) or its datacenter."`
}

func (c *VolumeDropCmd) Run(r *Runtime) error {
	return r.call(handleVolumeDrop, []string{c.Target}, nil, nil, false)
}

type UnloadCmd struct{}

func (c *UnloadCmd) Run(r *Runtime) error {
	return r.call(handleUnload, nil, nil, nil, false)
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
	Rental string `arg:"" name:"rental" help:"Attached machine name or rental id."`
}

func (c *RentalSSHInfoCmd) Run(r *Runtime) error {
	return r.call(handleRentalSSHInfo, []string{c.Rental}, nil, nil, false)
}
