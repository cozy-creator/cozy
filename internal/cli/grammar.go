package cli

// CLI is the complete public command grammar. Kong derives parsing and help from
// this tree; there is no parallel command manifest or string handler registry.
type CLI struct {
	JSON   bool     `help:"Emit JSON instead of human-readable output."`
	Full   bool     `help:"Include complete values and all available fields."`
	Fields []string `help:"Select result fields." sep:","`

	Package PackageCmd `cmd:"" group:"Resources" help:"Install the source-code that generates media."`
	Model   ModelCmd   `cmd:"" group:"Resources" help:"Download the tensors that are the AI's mind."`
	Auth    AuthCmd    `cmd:"" group:"Resources" help:"Authenticate this machine to Tensorhub."`
	Invoke  InvokeCmd  `cmd:"" group:"Work" help:"Generate media using your installed packages."`
	Rental  RentalCmd  `cmd:"" group:"Work" help:"Rent a more powerful GPU in the cloud."`
	Up      UpCmd      `cmd:"" group:"Lifecycle" help:"Start the cozy-daemon and localhost web-ui."`
	Down    DownCmd    `cmd:"" group:"Lifecycle" help:"Stop cozy-daemon and localhost web-ui."`
	Unload  UnloadCmd  `cmd:"" group:"Lifecycle" help:"Empty cached GPU AI models to free up VRAM."`
}

// Auth intentionally starts with one operation. Registration and recovery are the
// same email-root enrollment; a valid stored key authenticates automatically.
type AuthCmd struct {
	Login AuthLoginCmd `cmd:"" help:"Register or authenticate this machine by email."`
}

type AuthLoginCmd struct {
	Email string `arg:"" name:"email" help:"Tensorhub account email."`
}

func (c *AuthLoginCmd) Run(r *Runtime) error {
	return r.call(handleAuthLogin, []string{c.Email}, nil, nil, false)
}

type PackageCmd struct {
	Search  PackageSearchCmd  `cmd:"" help:"Search for AI magic."`
	Install PackageInstallCmd `cmd:"" help:"Downloads source-code and installs dependencies."`
	Remove  PackageRemoveCmd  `cmd:"" help:"Delete source-code."`
	List    PackageListCmd    `cmd:"" help:"List installed packages."`
	Publish PackagePublishCmd `cmd:"" help:"Publish a package release."`
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
	Ref           string `arg:"" name:"package" help:"Package ref (org/name[@release])."`
	From          string `help:"Install from a local release archive." type:"path"`
	Dir           string `help:"Install an editable local source tree." type:"path"`
	Digest        string `help:"Expected source digest."`
	Force         bool   `help:"Build and atomically replace an existing pin."`
	AllowUnsigned bool   `help:"Allow an unverified local development source."`
}

func (c *PackageInstallCmd) Run(r *Runtime) error {
	return r.call(handleInstall, []string{c.Ref}, bools(
		"--force", c.Force, "--allow-unsigned", c.AllowUnsigned), values(
		"--from", c.From, "--dir", c.Dir, "--digest", c.Digest), false)
}

type PackageRemoveCmd struct {
	Refs []string `arg:"" name:"package" help:"Installed package ref."`
}

func (c *PackageRemoveCmd) Run(r *Runtime) error {
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
	Search   ModelSearchCmd   `cmd:"" help:"Search the model catalog."`
	Download ModelDownloadCmd `cmd:"" help:"Download a model into the local TensorFS store."`
	Remove   ModelRemoveCmd   `cmd:"" help:"Remove local model roots."`
	List     ModelListCmd     `cmd:"" help:"List local model roots."`
	Publish  ModelPublishCmd  `cmd:"" help:"Publish a local TensorFS snapshot."`
}

type ModelSearchCmd struct {
	Query []string `arg:"" optional:"" name:"query" help:"Search text or an exact org/name."`
	Limit int      `help:"Maximum results." default:"20"`
}

func (c *ModelSearchCmd) Run(r *Runtime) error {
	return r.call(handleModelSearch, c.Query, nil,
		values("--limit", intText(c.Limit)), false)
}

type ModelDownloadCmd struct {
	Ref        string `arg:"" name:"model" help:"Model release or snapshot ref."`
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
	Refs []string `arg:"" name:"model" help:"Local model root name."`
}

func (c *ModelRemoveCmd) Run(r *Runtime) error {
	return r.call(handleModelRemove, c.Refs, nil, nil, false)
}

type ModelListCmd struct{}

func (c *ModelListCmd) Run(r *Runtime) error {
	return r.call(handleModelList, nil, nil, nil, false)
}

type ModelPublishCmd struct {
	Ref        string `arg:"" name:"model" help:"Model name (org/name)."`
	Snapshot   string `arg:"" name:"snapshot" help:"Local sha256 snapshot id."`
	DryRun     bool   `help:"Show the transfer plan without moving bytes."`
	TokenStdin bool   `help:"Read this invocation's hub token from stdin."`
}

func (c *ModelPublishCmd) Run(r *Runtime) error {
	return r.call(handleModelPublish, []string{c.Ref, c.Snapshot}, bools(
		"--dry-run", c.DryRun, "--token-stdin", c.TokenStdin),
		nil, false)
}

type InvokeCmd struct {
	Run    InvokeRunCmd    `cmd:"" help:"Run a package callable."`
	Cancel InvokeCancelCmd `cmd:"" help:"Cancel an invocation or job."`
	List   InvokeListCmd   `cmd:"" help:"List invocations and jobs."`
}

type InvokeRunCmd struct {
	Target         string   `arg:"" name:"target" help:"Callable as org/package/vN/function."`
	Input          []string `arg:"" optional:"" name:"input" help:"Primary value and field=value payload."`
	Out            string   `help:"Output directory." type:"path"`
	Timeout        string   `help:"Request deadline."`
	Stream         bool     `help:"Emit typed progress deltas."`
	PayloadFile    string   `name:"in" help:"Read the whole payload from JSON." type:"path"`
	Assets         []string `name:"asset" help:"Bind a local asset as field-path=file."`
	Worker         string   `help:"Run on an attached private rental."`
	IdempotencyKey string   `help:"Stable request identity for safe retries."`
	Trees          []string `name:"input-tree" help:"Bind a job input tree as ref=directory."`
	Org            string   `help:"Job publication organization (defaults to local)."`
	Detach         bool     `help:"Return after durable acceptance instead of following."`
}

func (c *InvokeRunCmd) Run(r *Runtime) error {
	args := append([]string{c.Target}, c.Input...)
	return r.call(handleInvokeRun, args, bools(
		"--stream", c.Stream, "--detach", c.Detach), values(
		"--out", c.Out, "--timeout", c.Timeout,
		"--in", c.PayloadFile, "--asset", c.Assets, "--worker", c.Worker,
		"--idempotency-key", c.IdempotencyKey, "--input", c.Trees, "--org", c.Org), true)
}

type InvokeCancelCmd struct {
	ID string `arg:"" name:"invocation" help:"Request or job id."`
}

func (c *InvokeCancelCmd) Run(r *Runtime) error {
	return r.call(handleInvokeCancel, []string{c.ID}, nil, nil, true)
}

type InvokeListCmd struct {
	State   string `help:"Filter by lifecycle state."`
	Package string `help:"Filter by package."`
	Limit   int    `help:"Maximum rows." default:"50"`
}

func (c *InvokeListCmd) Run(r *Runtime) error {
	return r.call(handleInvokeList, nil, nil, values(
		"--state", c.State, "--package", c.Package, "--limit", intText(c.Limit)), true)
}

type RentalCmd struct {
	New  RentalNewCmd  `cmd:"" help:"Start a private rental."`
	End  RentalEndCmd  `cmd:"" help:"End a private rental and stop billing."`
	List RentalListCmd `cmd:"" help:"List private rentals."`
}

type RentalNewCmd struct {
	SKU            string `arg:"" optional:"" name:"gpu" help:"Cozy GPU SKU, such as h200."`
	Package        string `arg:"" optional:"" name:"package" help:"Exact package ref."`
	IdempotencyKey string `help:"Stable paid-operation identity."`
	Timeout        string `help:"Caller wait deadline; does not release the rental."`
}

func (c *RentalNewCmd) Run(r *Runtime) error {
	return r.call(handleRent, []string{c.SKU, c.Package}, nil, values(
		"--idempotency-key", c.IdempotencyKey, "--timeout", c.Timeout), false)
}

type RentalEndCmd struct {
	ID string `arg:"" name:"rental" help:"Private rental id."`
}

func (c *RentalEndCmd) Run(r *Runtime) error {
	return r.call(handleRentRelease, []string{c.ID}, nil, nil, true)
}

type RentalListCmd struct{}

func (c *RentalListCmd) Run(r *Runtime) error {
	return r.call(handleRentLs, nil, nil, nil, true)
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
