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
	Up      UpCmd      `cmd:"" group:"Lifecycle" help:"Start the cozy-daemon and localhost web-ui."`
	Down    DownCmd    `cmd:"" group:"Lifecycle" help:"Stop cozy-daemon and localhost web-ui."`
	Unload  UnloadCmd  `cmd:"" group:"Lifecycle" help:"Empty cached GPU AI models to free up VRAM."`
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
	Search  PackageSearchCmd  `cmd:"" help:"Search for AI magic."`
	Install PackageInstallCmd `cmd:"" help:"Install a published package or explicit local directory."`
	Recover PackageRecoverCmd `cmd:"" help:"Repair package inventory from an explicit Creator database backup." hidden:""`
	Remove  PackageRemoveCmd  `cmd:"" help:"Delete source-code."`
	List    PackageListCmd    `cmd:"" help:"List installed packages."`
	Publish PackagePublishCmd `cmd:"" help:"Publish a package release."`
	Yank    PackageYankCmd    `cmd:"" help:"Permanently yank a package release."`
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
	Ref             string `arg:"" name:"package-or-directory" help:"Published org/name or explicit directory such as . or ./project."`
	Version         string `help:"Install this release instead of the newest, e.g. 1.2.3."`
	Editable        bool   `help:"Keep an explicit local directory live for development."`
	NoModelDownload bool   `help:"Install package code without prefetching its configured default model."`
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
	return r.call(handleInstall, []string{c.Ref}, bools("--editable", c.Editable,
		"--no-model-download", c.NoModelDownload),
		values("--version", c.Version), false)
}

func (c *PackageRecoverCmd) Run(r *Runtime) error {
	return r.call(handlePackageRecover, []string{c.Database}, nil, nil, false)
}

type PackageRemoveCmd struct {
	Refs []string `arg:"" name:"package" help:"Installed package ref."`
}

func (c *PackageRemoveCmd) Run(r *Runtime) error {
	return r.call(handleRm, c.Refs, nil, nil, true)
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
	Search   ModelSearchCmd   `cmd:"" help:"Search the model catalog."`
	Family   ModelFamilyCmd   `cmd:"" help:"Set a model repository's discovery family."`
	Import   ModelImportCmd   `cmd:"" help:"Import a foreign model into local TensorFS."`
	Download ModelDownloadCmd `cmd:"" help:"Download a model into the local TensorFS store."`
	Remove   ModelRemoveCmd   `cmd:"" help:"Remove local model repositories."`
	List     ModelListCmd     `cmd:"" help:"List local model releases."`
	Upload   ModelUploadCmd   `cmd:"" help:"Upload owner-only checkpoints from a source or reviewed producer."`
	Publish  ModelPublishCmd  `cmd:"" help:"Update a release's mutable lane pointers."`
	Yank     ModelYankCmd     `cmd:"" help:"Yank a model release."`
}

type ModelImportCmd struct {
	Source     string `arg:"" name:"source" help:"Pinned Hugging Face/Civitai source, allowlisted provider URL, or explicit local file."`
	Name       string `help:"Local model name under local/." required:""`
	DryRun     bool   `help:"Resolve and validate the import plan without moving model bodies."`
	TokenStdin bool   `help:"Read this local import's provider token from stdin."`
}

func (c *ModelImportCmd) Run(r *Runtime) error {
	return r.call(handleModelImport, []string{c.Source}, bools(
		"--dry-run", c.DryRun, "--token-stdin", c.TokenStdin), values("--name", c.Name), false)
}

type ModelSearchCmd struct {
	Query  []string `arg:"" optional:"" name:"query" help:"Search text or an exact org/name."`
	Limit  int      `help:"Maximum results." default:"20"`
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
	Ref        string `arg:"" name:"model" help:"Model release ref."`
	Lane       string `help:"Resolve one release lane."`
	DryRun     bool   `help:"Show the transfer plan without moving bytes."`
	TokenStdin bool   `help:"Read this invocation's hub token from stdin."`
}

func (c *ModelDownloadCmd) Run(r *Runtime) error {
	return r.call(handleModelDownload, []string{c.Ref}, bools(
		"--dry-run", c.DryRun, "--token-stdin", c.TokenStdin),
		values("--lane", c.Lane), false)
}

type ModelRemoveCmd struct {
	Refs []string `arg:"" name:"model" help:"Local model repository name."`
}

func (c *ModelRemoveCmd) Run(r *Runtime) error {
	return r.call(handleModelRemove, c.Refs, nil, nil, false)
}

type ModelListCmd struct{}

func (c *ModelListCmd) Run(r *Runtime) error {
	return r.call(handleModelList, nil, nil, nil, false)
}

type ModelUploadCmd struct {
	Ref      string `arg:"" name:"model" help:"Tensorhub destination (org/name)."`
	Source   string `arg:"" name:"source" help:"Manifest, pinned foreign source, Tensorhub model release, local alias, or explicit local file."`
	Producer string `help:"Reviewed org/package@vN/production declaration to execute."`
	Lane     string `help:"Select an input lane when source is a Tensorhub model release."`
	Rental   bool   `help:"Authorize one Creator-managed rental for this operation."`
	DryRun   bool   `help:"Resolve the exact production plan without moving model bodies or spending."`
	Detach   bool   `help:"Return after durable acceptance instead of following."`
}

func (c *ModelUploadCmd) Run(r *Runtime) error {
	return r.call(handleModelUpload, []string{c.Ref, c.Source}, bools(
		"--rental", c.Rental, "--dry-run", c.DryRun, "--detach", c.Detach),
		values("--producer", c.Producer, "--lane", c.Lane), false)
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

type ModelYankCmd struct {
	Ref     string `arg:"" name:"model" help:"Tensorhub model repository (org/name)."`
	Release string `help:"Release label to yank." required:""`
}

func (c *ModelYankCmd) Run(r *Runtime) error {
	return r.call(handleModelYank, []string{c.Ref}, nil, values("--release", c.Release), false)
}

type RunCmd struct {
	Execute RunExecuteCmd `cmd:"" default:"withargs" hidden:""`
	Cancel  RunCancelCmd  `cmd:"" help:"Cancel a queued or running run."`
	List    RunListCmd    `cmd:"" help:"List current and past runs."`
	Watch   RunWatchCmd   `cmd:"" help:"Watch one recorded run until it settles."`
}

type RunExecuteCmd struct {
	Target         string   `arg:"" name:"target" help:"Package or callable as org/package[/function]."`
	Input          []string `arg:"" optional:"" name:"input" help:"Primary value and field=value payload."`
	Out            string   `help:"Output directory." type:"path"`
	Timeout        string   `help:"Request deadline."`
	Stream         bool     `help:"Emit typed progress deltas."`
	PayloadFile    string   `name:"in" help:"Read the whole payload from a JSON file, e.g. --in request.json." type:"path"`
	Assets         []string `name:"asset" help:"Bind a local asset as field-path=file."`
	Models         []string `name:"model" help:"Bind a model: --model org/model@release for one slot; --model slot=org/model@release for named slots."`
	Rental         bool     `help:"Run on a Creator-managed rental."`
	RentalOnly     bool     `help:"Require a remote rental even when local capacity is ready."`
	IdempotencyKey string   `help:"Stable request identity for safe retries."`
	Trees          []string `name:"input-tree" help:"Bind a job input tree as ref=directory."`
	Org            string   `help:"Job publication organization (defaults to local)."`
	Await          bool     `help:"Wait for the terminal result instead of returning after the short optimistic observation."`
}

func (c *RunExecuteCmd) Run(r *Runtime) error {
	args := append([]string{c.Target}, c.Input...)
	return r.call(handleRunExecute, args, bools(
		"--stream", c.Stream, "--await", c.Await, "--rental", c.Rental,
		"--rental-only", c.RentalOnly), values(
		"--out", c.Out, "--timeout", c.Timeout,
		"--in", c.PayloadFile, "--asset", c.Assets,
		"--model", c.Models,
		"--idempotency-key", c.IdempotencyKey, "--input", c.Trees, "--org", c.Org), true)
}

type RunCancelCmd struct {
	ID string `arg:"" name:"run" help:"Run id."`
}

func (c *RunCancelCmd) Run(r *Runtime) error {
	return r.call(handleRunCancel, []string{c.ID}, nil, nil, true)
}

type RunListCmd struct {
	State   string `help:"Filter by lifecycle state."`
	Package string `help:"Filter by package."`
	Limit   int    `help:"Maximum rows." default:"50"`
}

func (c *RunListCmd) Run(r *Runtime) error {
	return r.call(handleRunList, nil, nil, values(
		"--state", c.State, "--package", c.Package, "--limit", intText(c.Limit)), true)
}

type RunWatchCmd struct {
	ID string `arg:"" name:"run" help:"Run id."`
}

func (c *RunWatchCmd) Run(r *Runtime) error {
	return r.call(handleRunWatch, []string{c.ID}, nil, nil, true)
}

type RentalCmd struct {
	Current RentalListCmd `cmd:"" default:"1" hidden:""`
	New     RentalNewCmd  `cmd:"" help:"Start a private rental."`
	End     RentalEndCmd  `cmd:"" help:"End a private rental and stop billing."`
}

type RentalNewCmd struct {
	SKU            string `arg:"" optional:"" name:"gpu" help:"Cozy GPU SKU, such as h200."`
	Name           string `help:"Memorable name for this rented machine."`
	IdempotencyKey string `help:"Stable paid-operation identity."`
	Timeout        string `help:"Caller wait deadline; does not release the rental."`
}

func (c *RentalNewCmd) Run(r *Runtime) error {
	return r.call(handleRent, []string{c.SKU}, nil, values(
		"--idempotency-key", c.IdempotencyKey, "--timeout", c.Timeout, "--name", c.Name), false)
}

type RentalEndCmd struct {
	ID string `arg:"" name:"rental" help:"Private rental id."`
}

func (c *RentalEndCmd) Run(r *Runtime) error {
	return r.call(handleRentRelease, []string{c.ID}, nil, nil, true)
}

type RentalListCmd struct{}

func (c *RentalListCmd) Run(r *Runtime) error {
	return r.call(handleRentLs, nil, nil, nil, false)
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
	All bool `help:"Cancel all work, end all rentals, then stop Cozy."`
}

func (c *DownCmd) Run(r *Runtime) error {
	return r.call(handleDown, nil, bools("--all", c.All), nil, false)
}
