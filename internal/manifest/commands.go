package manifest

import "github.com/cozy-creator/cozy-creator/internal/exit"

// FoundationTokens are the capability tokens this binary's CLI foundation carries,
// independent of any one command. Scripts gate on tokens, never version strings.
var FoundationTokens = []string{
	"cli.manifest",           // one declarative surface, startup self-checked
	"cli.noninteractive",     // no interactive prompt anywhere; --yes gates destructive acts
	"errors.typed",           // error(<name>) + remedy + next:
	"exit.matrix.v1",         // the shared cozy-runtime exit matrix, 0-14
	"install.generations",    // installs are immutable generations + a pin per (endpoint, major)
	"install.transaction",    // generation insert and pin activation commit together
	"output.fields",          // --fields
	"output.full",            // --full
	"output.json",            // --json
	"records.sqlite",         // ONE local SQLite lifecycle database, pure-Go driver
	"service.lock",           // liveness is an OS advisory lock the owner holds
	"worker.protocol.v1",     // the cozy.worker.v1 server over a unix socket
	"orchestrator.local",     // local dispatch: request -> attempt -> terminal -> visible output
	"terminal.transaction",   // terminal accepted + output visible commit together
	"catalog.public_reads",   // hub catalog reads carry no credential
	"hub.static_token",       // first-party writes carry ONE static admin token; no login act exists
	"transfer.declare_first", // publish declares the whole object set before a byte moves
	"transfer.resumable",     // an interrupted transfer resumes from verified state, not a local journal
	"transfer.verified_cas",  // every fetched byte enters the local store under its declared identity
	"cli.api_client",         // every lifecycle/request verb speaks the local client API, never a direct path
	"cli.credential.file",    // the CLI reads a 0600 handoff file; never argv, never an env value
	"run.one_path",           // one execution path: request -> attempt -> terminal -> accepted output
	"run.idempotency",        // `cozy run` carries a key; the same key returns the same request
	"run.payload.schema",     // payload typed against the recorded surface, before a request exists
	"run.media.save",         // `--out` writes accepted outputs by opaque id and declared field path
	"endpoint.generations",   // an installed generation is the ONLY source of launch facts
	"worker.remote.attach",   // a rented pod's worker is DIALED over TLS with its cert pinned
	"rental.pinned_triple",   // the rental's address/cert/owner token are pinned locally, 0600
}

// APITokens are the local client API's own tokens (cl-006). They live in internal/api's
// route table — ONE registry — and are folded in here so `cozy capabilities` and
// `GET /v1/capabilities` cannot disagree about what this binary carries.
var APITokens []string

// Tokens is the whole capability surface: the CLI foundation plus the API's own.
func Tokens() []string {
	return append(append([]string{}, FoundationTokens...), APITokens...)
}

var yesFlag = Flag{Name: "--yes", Summary: "confirm and execute; there is no prompt anywhere"}

// Commands is THE command surface. Implemented rows name a handler; planned rows
// name the issue that lands them. Nothing registers a command anywhere else.
var Commands = []Command{
	// ---- meta (cl-002) ----
	{
		Path: []string{"status"}, Group: "meta",
		Summary: "live status: LocalService, installed endpoints, running serves, active jobs",
		Args:    "", MaxArgs: 0,
		Exits:      []exit.Code{exit.OK},
		Capability: "cmd.status", Status: Implemented, Handler: "status",
		Next:     []string{"cozy run <org/endpoint/vN/function>", "cozy job ls"},
		Examples: []string{"cozy", "cozy status --json", "cozy status --fields service,endpoints"},
	},
	{
		Path: []string{"version"}, Group: "meta",
		Summary:    "tag + commit + protocol/contract versions",
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK},
		Capability: "cmd.version", Status: Implemented, Handler: "version",
		Next:     []string{"cozy capabilities"},
		Examples: []string{"cozy --version", "cozy version --json"},
	},
	{
		Path: []string{"capabilities"}, Group: "meta",
		Summary:    "feature tokens this binary supports — the one gate surface for scripts",
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK},
		Capability: "cmd.capabilities", Status: Implemented, Handler: "capabilities",
		Next:     []string{"cozy commands"},
		Examples: []string{"cozy capabilities", "cozy capabilities --json"},
	},
	{
		Path: []string{"commands"}, Group: "meta",
		Summary: "the command manifest itself: every command, its status and capability token",
		Args:    "[<prefix>]", MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.Usage},
		Capability: "cmd.commands", Status: Implemented, Handler: "commands",
		Next:     []string{"cozy help <command>"},
		Examples: []string{"cozy commands", "cozy commands job", "cozy commands --full"},
	},
	{
		Path: []string{"help"}, Group: "meta",
		Summary: "concise help for a command; never mutates, never touches the network",
		Args:    "[<command>]", MaxArgs: 2,
		Exits:      []exit.Code{exit.OK, exit.NotFound},
		Capability: "cmd.help", Status: Implemented, Handler: "help",
		Next:     []string{"cozy commands"},
		Examples: []string{"cozy help run", "cozy help job submit"},
	},

	// ---- service (cl-001: the LocalService's own lifecycle) ----
	{
		Path: []string{"up"}, Group: "service",
		Summary: "start the one LocalService (foreground; -d detaches)",
		Flags: []Flag{
			{Name: "--detach", Short: "-d", Summary: "run in the background"},
			{Name: "--port", Arg: "<n>", Summary: "local client API port, loopback only", Default: "2699"},
			{Name: "--open", Summary: "open the stub UI page with the per-launch token"},
			{Name: "--yield", Arg: "<smart|always|never>", Summary: "GPU yield policy", Default: "smart"},
		},
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK, exit.Conflict},
		Capability: "cmd.up", Status: Implemented, Handler: "up",
		Next:     []string{"cozy status", "cozy ls"},
		Examples: []string{"cozy up", "cozy up -d", "cozy up -d --port 2699 --yield never"},
	},
	{
		Path: []string{"down"}, Group: "service",
		Summary:    "stop the LocalService, draining endpoint processes",
		Flags:      []Flag{{Name: "--timeout", Arg: "<dur>", Summary: "drain bound; default follows the service cancellation budget"}},
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK},
		Capability: "cmd.down", Status: Implemented, Handler: "down",
		Next:     []string{"cozy up"},
		Examples: []string{"cozy down", "cozy down --timeout 60s"},
	},

	// ---- endpoints (cl-009) ----
	{
		Path: []string{"pack"}, Group: "endpoints",
		Summary: "pack an endpoint source tree into ONE deterministic py3-none-any wheel (th-039)",
		Args:    "<tree>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{
			{Name: "--name", Arg: "<dist>", Summary: "distribution name, when the tree declares no `[project] name` (the env lane passes the release name)"},
			{Name: "--version", Arg: "<x.y.z>", Summary: "version, when the tree declares no `[project] version`"},
			{Name: "--out", Arg: "<dir>", Summary: "where to write the wheel", Default: "the working directory"},
		},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound, exit.Structural},
		Capability: "cmd.pack", Status: Implemented, Handler: "pack",
		Next:     []string{"cozy install <org/endpoint> --from <archive.tar.gz>"},
		Examples: []string{"cozy pack ./my-endpoint", "cozy pack ./my-endpoint --out ./dist", "cozy pack ./my-endpoint --name my-endpoint --version 1.2.3"},
	},
	{
		Path: []string{"install"}, Group: "endpoints",
		Summary: "resolve, verify, build and pin an endpoint as an immutable generation",
		Args:    "<org/endpoint[@vN|@release]>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{
			{Name: "--from", Arg: "<archive.tar.gz>", Summary: "install one local release archive"},
			{Name: "--digest", Arg: "<sha256:…>", Summary: "the source digest the release declares — verified before anything executes"},
			{Name: "--force", Summary: "build a new generation and swap the pin"},
			{Name: "--allow-unsigned", Summary: "development-only door past source verification"},
			{Name: "--profile", Arg: "<profile>", Summary: "independently qualify and install one published local profile"},
			{Name: "--major", Arg: "<vN>", Summary: "required local pin major with --profile"},
			{Name: "--grant-ttl", Arg: "<duration>", Summary: "local wheel/lease grant lifetime", Default: "10m"},
			{Name: "--device", Arg: "<index>", Summary: "concrete local GPU index for native qualification", Default: "0"},
			{Name: "--reason", Arg: "<why>", Summary: "required with --profile; recorded before the lease"},
		},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound, exit.Credential, exit.Structural, exit.Conflict, exit.Capacity},
		Capability: "cmd.install", Status: Implemented, Handler: "install",
		Next:     []string{"cozy run <org/endpoint/vN/function>", "cozy describe <org/endpoint>"},
		Examples: []string{"cozy install org/endpoint@1.0.0 --profile torch2.13.0-cu130-cp312-linux-x86 --major v1 --reason <why>", "cozy install org/endpoint --from ./release.tar.gz --digest sha256:<hex>", "cozy install org/endpoint --from ./release.tar.gz --force"},
	},
	{
		Path: []string{"ls"}, Group: "endpoints",
		Summary:    "installed endpoints: pin, version, disk (read from the install record)",
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK, exit.Usage},
		Capability: "cmd.ls", Status: Implemented, Handler: "ls",
		Next:     []string{"cozy describe <org/endpoint>", "cozy run <org/endpoint/vN/function>"},
		Examples: []string{"cozy ls", "cozy ls --fields endpoint,version,disk", "cozy ls --full --json"},
	},
	{
		Path: []string{"rm"}, Group: "endpoints",
		Summary: "remove an install (venv + pin); the generation record stays until gc",
		Args:    "<org/endpoint[@vN]> …", MinArgs: 1, MaxArgs: -1,
		Flags:      []Flag{yesFlag},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Confirm, exit.Conflict},
		Capability: "cmd.rm", Destructive: true, Status: Implemented, Handler: "rm",
		Next:     []string{"cozy ls", "cozy gc"},
		Examples: []string{"cozy rm org/endpoint --yes", "cozy rm org/endpoint@v2 org/other --yes"},
	},
	{
		Path: []string{"gc"}, Group: "endpoints",
		Summary: "without --yes the reclaim plan (a read); with --yes it executes",
		Flags: []Flag{yesFlag,
			{Name: "--keep-media", Arg: "<dur>", Summary: "local output retention horizon; 0 keeps nothing", Default: "30d"}},
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Conflict},
		Capability: "cmd.gc", PlanFirst: true, Status: Implemented, Handler: "gc",
		Next:     []string{"cozy ls"},
		Examples: []string{"cozy gc", "cozy gc --yes"},
	},

	// ---- retained local outputs (cl-033) ----
	{
		Path: []string{"media", "ls"}, Group: "media",
		Summary: "retained local outputs: what this host stores, and what gc would reclaim",
		Flags: []Flag{
			{Name: "--keep-media", Arg: "<dur>", Summary: "horizon to mark rows against", Default: "30d"}},
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK, exit.Usage},
		Capability: "cmd.media.ls", Status: Implemented, Handler: "media.ls",
		Next:     []string{"cozy gc", "cozy gc --yes --keep-media <dur>"},
		Examples: []string{"cozy media ls", "cozy media ls --keep-media 7d", "cozy media ls --json"},
	},
	{
		Path: []string{"start"}, Group: "endpoints",
		Summary: "lifecycle/prewarm: make an endpoint worker resident (never a second invoke path)",
		Args:    "<org/endpoint[@vN]>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{
			{Name: "--detach", Short: "-d", Summary: "leave it warm in the background"},
			{Name: "--no-warm", Summary: "skip the boot warm pass (an entrypoint whose warm shape does not fit degrades its binding for nothing)"},
		},
		Exits: []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound, exit.Structural,
			exit.Unavailable, exit.Deadline, exit.Failed, exit.Conflict},
		Capability: "cmd.start", NeedsServer: true, Status: Implemented, Handler: "start",
		Next:     []string{"cozy run <org/endpoint/vN/function>", "cozy stop <org/endpoint>"},
		Examples: []string{"cozy start org/endpoint", "cozy start org/endpoint@v2 -d", "cozy start org/endpoint --no-warm"},
	},
	{
		Path: []string{"stop"}, Group: "endpoints",
		Summary: "stop endpoint process(es); blocks until the process group exits",
		Args:    "<org/endpoint> | --all", MaxArgs: 1,
		Flags: []Flag{
			{Name: "--all", Summary: "every endpoint process"},
		},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Unavailable},
		Capability: "cmd.stop", NeedsServer: true, Status: Implemented, Handler: "stop",
		Next:     []string{"cozy status", "cozy start <org/endpoint>"},
		Examples: []string{"cozy stop org/endpoint", "cozy stop --all"},
	},
	{
		Path: []string{"logs"}, Group: "endpoints",
		Summary: "endpoint process log, or the retained triage bundle for an attempt id",
		Args:    "<org/endpoint | attempt-id>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{
			{Name: "--lines", Short: "-n", Arg: "<count>", Summary: "tail count", Default: "100"},
		},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.NotFound, exit.Unavailable, exit.Conflict},
		Capability: "cmd.logs", NeedsServer: true, Status: Implemented, Handler: "logs",
		Next:     []string{"cozy status"},
		Examples: []string{"cozy logs org/endpoint", "cozy logs org/endpoint -n 200", "cozy logs <attempt-id> --json"},
	},

	// ---- invocation (cl-010) ----
	{
		Path: []string{"run"}, Group: "invocation",
		Summary: "invoke one function through the orchestrator — the one invocation path",
		Args:    "<org/endpoint/vN/function> [<primary>] [key=value …]", MinArgs: 1, MaxArgs: -1,
		Flags: []Flag{
			{Name: "--model", Arg: "<[binding-path=]ref|path>", Summary: "override a model binding (repeatable)"},
			{Name: "--lane", Arg: "<contract-expr|auto>", Summary: "pin a release lane"},
			{Name: "--adapter", Arg: "<ref[:scale]>", Summary: "stack an adapter (repeatable; order = stack order)"},
			{Name: "--seed", Arg: "<n>", Summary: "deterministic RNG"},
			{Name: "--out", Arg: "<dir>", Summary: "output directory"},
			{Name: "--offline", Summary: "CAS-only; a miss is exit 8"},
			{Name: "--timeout", Arg: "<dur>", Summary: "request deadline"},
			{Name: "--stream", Summary: "typed deltas as NDJSON"},
			{Name: "--in", Arg: "<file>", Summary: "whole payload as JSON"},
			{Name: "--asset", Arg: "<field-path>=<file>", Summary: "bind a local asset file to a request field (repeatable)"},
			{Name: "--local", Summary: "run on this host", Default: "on"},
			{Name: "--cloud", Summary: "submit to tensorhub under the account"},
			{Name: "--worker", Arg: "<rental-id>", Summary: "pin this run to an attached rented pod (`cozy rent ls`)"},
			// One key names one request forever. A bare `run` mints its own, so retry
			// safety across a process restart is the caller's explicit act.
			{Name: "--idempotency-key", Arg: "<k>", Summary: "reuse a key: the same key returns the same request"},
		},
		Exits: []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound, exit.Credential,
			exit.Structural, exit.OfflineMiss, exit.Unavailable, exit.Deadline, exit.Failed,
			exit.Canceled, exit.Conflict, exit.Capacity},
		Capability: "cmd.run", NeedsServer: true, Terminals: true, Status: Implemented, Handler: "run",
		Next:     []string{"cozy job submit <org/endpoint/vN/function>", "cozy logs <org/endpoint>"},
		Examples: []string{"cozy run org/endpoint/v1/generate \"a red bicycle\"", "cozy run org/endpoint/v1/generate --in ./payload.json --out ./outputs", "cozy run org/endpoint/v1/generate prompt=\"a red bicycle\" --seed 7 --stream"},
	},
	{
		Path: []string{"describe"}, Group: "invocation",
		Summary: "the endpoint descriptor: the surface this release's own runtime derived and the install verified",
		Args:    "<org/endpoint[@vN][/function]>", MinArgs: 1, MaxArgs: 1,
		Exits: []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound, exit.Structural, exit.Conflict},
		// A records-plane read (cl-009's D3 rule): the descriptor is a verified document
		// in the generation, so the LocalService is not on the path to reading it.
		Capability: "cmd.describe", Status: Implemented, Handler: "describe",
		SelfContained: true,
		Examples:      []string{"cozy describe org/endpoint", "cozy describe org/endpoint@v2/generate"},
	},
	{
		Path: []string{"doctor"}, Group: "invocation",
		Summary:    "host facts plus per-installed-endpoint fit verdicts",
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK, exit.Unavailable},
		Capability: "cmd.doctor", NeedsServer: true, Status: Implemented, Handler: "doctor",
		Next:     []string{"cozy fit <org/endpoint>"},
		Examples: []string{"cozy doctor", "cozy doctor --json"},
	},
	{
		Path: []string{"fit"}, Group: "invocation",
		Summary: "fit verdicts for one endpoint/binding: fits / degraded / structural / capacity",
		Args:    "<org/endpoint[@vN][/function]>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{
			{Name: "--model", Arg: "<ref>", Summary: "binding override"},
			{Name: "--lane", Arg: "<l>", Summary: "lane"},
		},
		// ADVISORY above the physical floor (a degradable shortfall degrades and exits 0);
		// 14 below it with the runtime's quantified shortfall; 6 structural.
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.NotFound, exit.Structural, exit.Capacity},
		Capability: "cmd.fit", Status: Implemented, Handler: "fit",
		SelfContained: true,
		Examples:      []string{"cozy fit org/endpoint", "cozy fit org/endpoint@v2/generate --lane auto"},
	},

	// ---- durable workflows (cl-018) ----
	{
		Path: []string{"workflow", "submit"}, Group: "workflows",
		Summary: "submit one canonical ordered workflow of ordinary endpoint requests",
		MaxArgs: 0,
		Flags: []Flag{
			{Name: "--in", Arg: "<file>", Summary: "canonical cozy.workflow.Plan/1 JSON"},
			{Name: "--worker", Arg: "<step>=<rental-id>", Summary: "run one step on an attached exact rental (repeatable)"},
			{Name: "--idempotency-key", Arg: "<key>", Summary: "name this workflow forever; a repeat answers the same one"},
		},
		Exits: []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound,
			exit.Unavailable, exit.Conflict},
		Capability: "cmd.workflow.submit", NeedsServer: true, Status: Implemented,
		Handler:  "workflow.submit",
		Next:     []string{"cozy workflow status <workflow-id>", "cozy workflow follow <workflow-id>"},
		Examples: []string{"cozy workflow submit --in ./plan.json --idempotency-key <key>", "cozy workflow submit --in ./plan.json --worker shot-1=<rental-id> --idempotency-key <key>"},
	},
	{
		Path: []string{"workflow", "status"}, Group: "workflows",
		Summary: "workflow state projected from its ordinary child requests and outputs",
		Args:    "<workflow-id>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Unavailable},
		Capability: "cmd.workflow.status", NeedsServer: true, Status: Implemented,
		Handler:       "workflow.status",
		SelfContained: true,
		Examples:      []string{"cozy workflow status <workflow-id>", "cozy workflow status <workflow-id> --json"},
	},
	{
		Path: []string{"workflow", "follow"}, Group: "workflows",
		Summary: "follow durable child progress to the workflow terminal; no elapsed-time verdict",
		Args:    "<workflow-id>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Unavailable, exit.Failed, exit.Canceled},
		Capability: "cmd.workflow.follow", NeedsServer: true, Terminals: true,
		Status: Implemented, Handler: "workflow.follow",
		Next:     []string{"cozy workflow download <workflow-id> --out <new-dir>"},
		Examples: []string{"cozy workflow follow <workflow-id>"},
	},
	{
		Path: []string{"workflow", "download"}, Group: "workflows",
		Summary: "download every child output and exact receipt into one fresh local directory",
		Args:    "<workflow-id>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{{Name: "--out", Arg: "<new-dir>", Summary: "required fresh local acceptance directory"}},
		Exits: []exit.Code{exit.OK, exit.Usage, exit.NotFound, exit.Unavailable,
			exit.Validation, exit.Conflict},
		Capability: "cmd.workflow.download", NeedsServer: true, Status: Implemented,
		Handler:  "workflow.download",
		Next:     []string{"cozy workflow status <workflow-id>"},
		Examples: []string{"cozy workflow download <workflow-id> --out ./accepted"},
	},
	{
		Path: []string{"workflow", "cancel"}, Group: "workflows",
		Summary: "persist cancellation, cancel the active child, and prevent later children",
		Args:    "<workflow-id>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Unavailable},
		Capability: "cmd.workflow.cancel", NeedsServer: true, Status: Implemented,
		Handler:  "workflow.cancel",
		Next:     []string{"cozy workflow status <workflow-id>", "cozy workflow ls"},
		Examples: []string{"cozy workflow cancel <workflow-id>"},
	},

	// ---- editable Cozy Video sources (cl-024) ----
	{
		Path: []string{"video", "compose"}, Group: "videos",
		Summary: "compose an editable cozy.video/1 source into an exact ordinary workflow",
		Args:    "<source.cozy-video.yaml>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{
			{Name: "--h3", Arg: "<endpoint-ref>", Summary: "local-only H3 endpoint; mutually exclusive with --rental"},
			{Name: "--rental", Arg: "<rental-id>", Summary: "remote H3 target; its endpoint is already exact"},
			{Name: "--out", Arg: "<file>", Summary: "write the path-free composition document"},
		},
		Exits: []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound,
			exit.Structural, exit.Unavailable, exit.Conflict},
		Capability: "cmd.video.compose", NeedsServer: true, Status: Implemented,
		Handler:  "video.compose",
		Next:     []string{"cozy video submit <creative-plan-digest> --idempotency-key <key>"},
		Examples: []string{"cozy video compose ./film.cozy-video.yaml --h3 org/endpoint", "cozy video compose ./film.cozy-video.yaml --out ./film.composition.json"},
	},
	{
		Path: []string{"video", "submit"}, Group: "videos",
		Summary: "submit a source or retained creative plan through the durable workflow owner",
		Args:    "<source.cozy-video.yaml | creative-plan-digest>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{
			{Name: "--h3", Arg: "<endpoint-ref>", Summary: "local-only H3 endpoint; mutually exclusive with --rental"},
			{Name: "--rental", Arg: "<rental-id>", Summary: "run every H3 shot on this Tensorhub-selected rental; assembly stays local"},
			{Name: "--idempotency-key", Arg: "<key>", Summary: "required: name this workflow so a lost response can be retried"},
		},
		Exits: []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound,
			exit.Structural, exit.Unavailable, exit.Conflict},
		Capability: "cmd.video.submit", NeedsServer: true, Status: Implemented,
		Handler:  "video.submit",
		Next:     []string{"cozy workflow status <workflow-id>", "cozy workflow follow <workflow-id>"},
		Examples: []string{"cozy video submit ./film.cozy-video.yaml --h3 org/endpoint --idempotency-key <key>", "cozy video submit <creative-plan-digest> --h3 org/endpoint --idempotency-key <key>"},
	},

	// ---- jobs (cl-004) ----
	{
		Path: []string{"job", "submit"}, Group: "jobs",
		Summary: "submit a @job; prints the job id and its publication repo",
		Args:    "<org/endpoint/vN/function> [key=value …]", MinArgs: 1, MaxArgs: -1,
		Flags: []Flag{
			{Name: "--local", Summary: "run on this host"},
			{Name: "--cloud", Summary: "submit to tensorhub"},
			{Name: "--model", Arg: "<ref>", Summary: "override a model binding (repeatable)"},
			{Name: "--input", Arg: "<ref>=<dir>", Summary: "a typed input TREE, granted as a read capability (repeatable)"},
			{Name: "--org", Arg: "<name>", Summary: "the org whose scratch repo this job publishes into", Default: "local"},
			{Name: "--in", Arg: "<file>", Summary: "whole payload as JSON"},
			{Name: "--idempotency-key", Arg: "<key>", Summary: "name this job forever; a repeat answers the same one"},
			{Name: "--follow", Summary: "attach immediately and exit with the terminal mapping"},
		},
		Exits:      []exit.Code{exit.OK, exit.Validation, exit.NotFound, exit.Unavailable, exit.Deadline, exit.Failed, exit.Canceled},
		Capability: "cmd.job.submit", NeedsServer: true, Terminals: true, Status: Implemented, Handler: "job.submit",
		Next:     []string{"cozy job follow <job-id>", "cozy job ls"},
		Examples: []string{"cozy job submit org/endpoint/v1/train epochs=3", "cozy job submit org/endpoint/v1/train --in ./payload.json --follow", "cozy job submit org/endpoint/v1/train --input dataset=./data --idempotency-key <key>"},
	},
	{
		Path: []string{"job", "status"}, Group: "jobs",
		Summary: "state, progress, elapsed, the running bill where a rate exists, terminal",
		Args:    "<job-id>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Unavailable},
		Capability: "cmd.job.status", NeedsServer: true, Status: Implemented, Handler: "job.status",
		SelfContained: true,
		Examples:      []string{"cozy job status <job-id>", "cozy job status <job-id> --json"},
	},
	{
		Path: []string{"job", "ls"}, Group: "jobs",
		Summary: "jobs with per-state counts in the footer",
		Flags: []Flag{
			{Name: "--state", Arg: "<s>", Summary: "filter by state"},
			{Name: "--endpoint", Arg: "<e>", Summary: "filter by endpoint"},
		},
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK, exit.Unavailable},
		Capability: "cmd.job.ls", NeedsServer: true, Status: Implemented, Handler: "job.ls",
		Next:     []string{"cozy job status <job-id>", "cozy job follow <job-id>"},
		Examples: []string{"cozy job ls", "cozy job ls --state running", "cozy job ls --endpoint org/endpoint --full"},
	},
	{
		Path: []string{"job", "follow"}, Group: "jobs",
		Summary: "attach to the progress/metric stream; exits with the terminal mapping",
		Args:    "<job-id>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Unavailable, exit.Deadline, exit.Failed, exit.Canceled},
		Capability: "cmd.job.follow", NeedsServer: true, Terminals: true, Status: Implemented, Handler: "job.follow",
		Next:     []string{"cozy job ls", "cozy logs <attempt-id>"},
		Examples: []string{"cozy job follow <job-id>"},
	},
	{
		Path: []string{"job", "cancel"}, Group: "jobs",
		Summary: "request cancel and block until the canceled terminal",
		Args:    "<job-id>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Unavailable},
		Capability: "cmd.job.cancel", NeedsServer: true, Status: Implemented, Handler: "job.cancel",
		Next:     []string{"cozy job ls", "cozy job status <job-id>"},
		Examples: []string{"cozy job cancel <job-id>"},
	},

	// ---- rentals (cl-015) ----
	// A rental is a provider-neutral pod lifetime. This slice persists exact desired
	// request bytes; cl-019 moves their side effects under the LocalService reconciler.
	{
		Path: []string{"rent"}, Group: "rentals",
		Summary: "rent a pod through the hub and pin its dial triple for `run --worker`",
		Args:    "<endpoint-ref>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{
			{Name: "--accelerator", Arg: "<model>", Summary: "required; provider-neutral accelerator model"},
			{Name: "--idempotency-key", Arg: "<key>", Summary: "resume one paid ask; the same key never buys twice"},
			{Name: "--timeout", Arg: "<dur>", Summary: "give up waiting for ready; the pod is NOT released"},
			{Name: "--reason", Arg: "<why>", Summary: "required; the hub records it durably before it acts"},
		},
		Exits: []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound, exit.Credential,
			exit.Unavailable, exit.Deadline, exit.Failed},
		Capability: "cmd.rent", Status: Implemented, Handler: "rent",
		Next:     []string{"cozy rent ls", "cozy rent probe <rental-id>"},
		Examples: []string{"cozy rent org/endpoint --accelerator <model> --reason <why>", "cozy rent org/endpoint --accelerator <model> --reason <why> --idempotency-key <key>"},
	},
	{
		Path: []string{"rent", "ls"}, Group: "rentals",
		Summary:    "pods this host holds: state, address, and the owner token's DIGEST",
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK, exit.Usage},
		Capability: "cmd.rent.ls", Status: Implemented, Handler: "rent.ls",
		Next:     []string{"cozy rent show <rental-id>", "cozy rent release <rental-id>"},
		Examples: []string{"cozy rent ls", "cozy rent ls --json"},
	},
	{
		Path: []string{"rent", "show"}, Group: "rentals",
		Summary: "the exact Tensorhub-selected execution and model-root closure for one rental",
		Args:    "<rental-id>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.NotFound, exit.Conflict},
		Capability: "cmd.rent.show", Status: Implemented, Handler: "rent.show",
		Next:     []string{"cozy rent probe <rental-id>", "cozy run <org/endpoint/vN/function> --worker <rental-id>"},
		Examples: []string{"cozy rent show <rental-id>"},
	},
	{
		Path: []string{"rent", "probe"}, Group: "rentals",
		Summary: "claim one ready rental and persist actual GPU readback without invoking a model",
		Args:    "<rental-id>", MinArgs: 1, MaxArgs: 1,
		Exits: []exit.Code{exit.OK, exit.Usage, exit.NotFound, exit.Unavailable,
			exit.Failed, exit.Conflict},
		Capability: "cmd.rent.probe", NeedsServer: true, Status: Implemented, Handler: "rent.probe",
		Next:     []string{"cozy run <org/endpoint/vN/function> --worker <rental-id>", "cozy rent show <rental-id>"},
		Examples: []string{"cozy rent probe <rental-id>"},
	},
	{
		Path: []string{"rent", "revise"}, Group: "rentals",
		Summary: "converge one new endpoint revision on the same claimed rental",
		Args:    "<rental-id>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{
			{Name: "--endpoint-ref", Arg: "<org/endpoint/vN/function>", Summary: "required new exact endpoint ref"},
			{Name: "--idempotency-key", Arg: "<key>", Summary: "required durable revision identity"},
			{Name: "--reason", Arg: "<why>", Summary: "required operator audit reason"},
		},
		Exits: []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound, exit.Credential,
			exit.Unavailable, exit.Deadline, exit.Conflict},
		Capability: "cmd.rent.revise", NeedsServer: true, Status: Implemented, Handler: "rent.revise",
		Next:     []string{"cozy rent show <rental-id>", "cozy status"},
		Examples: []string{"cozy rent revise <rental-id> --endpoint-ref org/endpoint/v2/generate --idempotency-key <key> --reason <why>"},
	},
	{
		Path: []string{"rent", "release"}, Group: "rentals",
		Summary: "without --yes what releasing costs (a read); with --yes the pod is destroyed",
		Args:    "<rental-id>", MinArgs: 1, MaxArgs: 1,
		Flags:      []Flag{yesFlag},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.NotFound, exit.Credential, exit.Unavailable, exit.Deadline},
		Capability: "cmd.rent.release", PlanFirst: true, Status: Implemented, Handler: "rent.release",
		Next:     []string{"cozy rent ls"},
		Examples: []string{"cozy rent release <rental-id>", "cozy rent release <rental-id> --yes"},
	},

	// ---- catalog (cl-011) ----
	// Launch-1 tensorhub is a headless distribution hub with NO identity plane
	// (decisions #229): reads are public and carry no credential, first-party writes
	// carry the ONE static admin token from TENSORHUB_TOKEN.
	{
		Path: []string{"endpoint", "search"}, Group: "catalog",
		Summary: "search endpoints — public, no credential",
		Args:    "[<query>]", MaxArgs: -1,
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Unavailable, exit.Deadline},
		Capability: "cmd.endpoint.search", Status: Implemented, Handler: "endpoint.search",
		Next:     []string{"cozy endpoint show <org/name>"},
		Examples: []string{"cozy endpoint search", "cozy endpoint search flux", "cozy endpoint search --json"},
	},
	{
		Path: []string{"model", "search"}, Group: "catalog",
		Summary: "search models — public, no credential",
		Args:    "[<query>]", MaxArgs: -1,
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Unavailable, exit.Deadline},
		Capability: "cmd.model.search", Status: Implemented, Handler: "model.search",
		Next:     []string{"cozy model show <org/name>"},
		Examples: []string{"cozy model search", "cozy model search flux", "cozy model search --json"},
	},
	{
		Path: []string{"endpoint", "show"}, Group: "catalog",
		Summary: "show one endpoint — public, no credential",
		Args:    "<org/name>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.NotFound, exit.Unavailable, exit.Deadline},
		Capability: "cmd.endpoint.show", Status: Implemented, Handler: "endpoint.show",
		Next:     []string{"cozy endpoint search <query>"},
		Examples: []string{"cozy endpoint show org/name"},
	},
	{
		Path: []string{"endpoint", "create"}, Group: "catalog",
		Summary: "create an endpoint — first-party, admin token",
		Args:    "<org/name>", MinArgs: 1, MaxArgs: 1,
		Flags:      []Flag{{Name: "--reason", Arg: "<why>", Summary: "required; the hub records it durably before it acts"}},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.Credential, exit.Unavailable, exit.Deadline, exit.Conflict},
		Capability: "cmd.endpoint.create", Status: Implemented, Handler: "endpoint.create",
		Next:     []string{"cozy endpoint show <org/name>", "cozy endpoint publish <org/name> --release <id> --dir <tree>"},
		Examples: []string{"cozy endpoint create org/name --reason <why>"},
	},
	{
		Path: []string{"model", "show"}, Group: "catalog",
		Summary: "show one model — public, no credential",
		Args:    "<org/name>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.NotFound, exit.Unavailable, exit.Deadline},
		Capability: "cmd.model.show", Status: Implemented, Handler: "model.show",
		Next:     []string{"cozy model search <query>"},
		Examples: []string{"cozy model show org/name"},
	},
	{
		Path: []string{"model", "create"}, Group: "catalog",
		Summary: "create a model — first-party, admin token",
		Args:    "<org/name>", MinArgs: 1, MaxArgs: 1,
		Flags:      []Flag{{Name: "--reason", Arg: "<why>", Summary: "required; the hub records it durably before it acts"}},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.Credential, exit.Unavailable, exit.Deadline, exit.Conflict},
		Capability: "cmd.model.create", Status: Implemented, Handler: "model.create",
		Next:     []string{"cozy model show <org/name>", "cozy model publish <org/model> <sha256:snapshot> --reason <why>"},
		Examples: []string{"cozy model create org/name --reason <why>"},
	},
	{
		Path: []string{"hub", "status"}, Group: "catalog",
		Summary: "the configured hub: URL, credential digest, reachability, catalog size",
		MaxArgs: 0,
		// Content-first: an unreachable hub is a STATE this verb
		// reports, not a refusal it raises.
		Exits:      []exit.Code{exit.OK},
		Capability: "cmd.hub.status", Status: Implemented, Handler: "hub.status",
		Next:     []string{"cozy endpoint search", "cozy model search", "cozy hub config"},
		Examples: []string{"cozy hub status", "cozy hub status --json"},
	},
	{
		Path: []string{"hub", "config"}, Group: "catalog",
		Summary:    "the hub's running config with per-key provenance — admin token; secrets digested",
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Credential, exit.Unavailable, exit.Deadline},
		Capability: "cmd.hub.config", Status: Implemented, Handler: "hub.config",
		Next:     []string{"cozy hub status"},
		Examples: []string{"cozy hub config", "cozy hub config --fields key,value,source"},
	},

	// ---- transfer (cl-012, cl-008) ----
	// cl-012's two transfer verbs are records-plane acts, not process-plane ones:
	// they move bytes between the local canonical store and the hub and never touch
	// an endpoint process, so neither needs the LocalService (cl-009's D3 rule).
	{
		Path: []string{"model", "download"}, Group: "transfer",
		Summary: "fetch a hub checkpoint into the shared local store — verified, resumable",
		Args:    "<org/model[@release|@sha256:…]>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{
			{Name: "--lane", Arg: "<selector>", Summary: "resolve one release lane; no default is invented"},
			{Name: "--dry-run", Summary: "print the plan without moving bytes"},
			// cl-011's law, enforced by the `secret` fence family: a credential never
			// rides argv. cozy-creator.md still spells this `--token <t>`; the doc is
			// the drift, not this row.
			{Name: "--token-stdin", Summary: "read this invocation's hub credential from stdin (never argv)"},
			{Name: "--crash-after", Arg: "<n>", Summary: "development: stop after n objects (resume verification)"},
		},
		Exits: []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound, exit.Credential,
			exit.Structural, exit.Unavailable, exit.Deadline},
		Capability: "cmd.model.download", Status: Implemented, Handler: "model.download",
		Next:     []string{"cozy install <org/endpoint>", "cozy model search <query>"},
		Examples: []string{"cozy model download org/model@release --lane task=text-to-video", "cozy model download org/model@sha256:<hex>", "cozy model download org/model@release --dry-run"},
	},
	{
		Path: []string{"model", "publish"}, Group: "transfer",
		Summary: "incrementally publish a canonical snapshot; only unsettled transfers move",
		Args:    "<org/model> <sha256:snapshot>", MinArgs: 2, MaxArgs: 2,
		Flags: []Flag{
			// No `--family`: th-003's hub CLASSIFIES the family from the artifact's
			// topology digest and refuses an unknown field, so a declared family is
			// both unnecessary and fatal (cl-006's real-hub side-check, closed here).
			{Name: "--reason", Arg: "<why>", Summary: "required; the hub records it durably before it acts"},
			{Name: "--dry-run", Summary: "open and claim transfers, then stop before grants or bytes"},
			{Name: "--token-stdin", Summary: "read this invocation's hub credential from stdin (never argv)"},
			{Name: "--crash-after", Arg: "<n>", Summary: "development: stop after n objects (resume verification)"},
		},
		Exits: []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound, exit.Credential,
			exit.Structural, exit.Unavailable, exit.Deadline, exit.Failed, exit.Conflict},
		Capability: "cmd.model.publish", Status: Implemented, Handler: "model.publish",
		Next:     []string{"cozy model show <org/name>", "cozy model download <org/model>"},
		Examples: []string{"cozy model publish org/model sha256:<snapshot> --reason <why>", "cozy model publish org/model sha256:<snapshot> --reason <why> --dry-run"},
	},
	{
		Path: []string{"datasets", "push"}, Group: "transfer",
		Summary: "the dataset dialect of push — deferred: there is no datasets plane to push into",
		Args:    "<path> <org/dataset>", MinArgs: 2, MaxArgs: 2,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Credential},
		Capability: "cmd.datasets.push", Status: Planned, Issue: "th-035",
	},
	{
		Path: []string{"datasets", "pull"}, Group: "transfer",
		Summary: "the dataset dialect of pull — deferred with the datasets plane",
		Args:    "<org/dataset[@release|@digest]>", MinArgs: 1, MaxArgs: 1,
		Flags:      []Flag{{Name: "--out", Arg: "<dir>", Summary: "materialize here"}},
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Credential},
		Capability: "cmd.datasets.pull", Status: Planned, Issue: "th-035",
	},
	{
		Path: []string{"model", "export"}, Group: "transfer",
		Summary: "authorized local checkpoint export through the orchestrator surface",
		Args:    "<ref>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Unavailable},
		Capability: "cmd.model.export", NeedsServer: true, Status: Planned, Issue: "cl-008",
	},
	// Endpoint publication, paid profile qualification, and atomic serving promotion
	// are three explicit acts. None invokes either of the other two.
	{
		Path: []string{"endpoint", "publish"}, Group: "transfer",
		Summary: "publish one source release and frozen profile candidate set",
		Args:    "<org/endpoint>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{
			{Name: "--release", Arg: "<id>", Summary: "required immutable endpoint release id"},
			{Name: "--dir", Arg: "<tree>", Summary: "endpoint source tree", Default: "the working directory"},
			{Name: "--profile", Arg: "<profile>", Summary: "required approved compatibility profile (repeatable)"},
			{Name: "--custom-wheel", Arg: "<profile>=<path>", Summary: "exact prebuilt custom wheel (repeatable)"},
			{Name: "--create", Summary: "required for the first publish of a new name"},
			{Name: "--reason", Arg: "<why>", Summary: "required; recorded before each mutation"},
		},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound, exit.Credential, exit.Structural, exit.Unavailable, exit.Deadline, exit.Failed, exit.Conflict},
		Capability: "cmd.endpoint.publish", Status: Implemented, Handler: "endpoint.publish",
		Next:     []string{"cozy endpoint qualify <org/endpoint>@<release> --profile <profile> --gpu <model> --max-cost <usd> --reason <why>", "cozy endpoint promote <org/endpoint> <release> --serve <vN/function> --reason <why>"},
		Examples: []string{"cozy endpoint publish org/endpoint --release 1.0.0 --profile torch2.13.0-cu130-cp312-linux-x86 --reason <why>", "cozy endpoint publish org/endpoint --release 1.0.0 --profile torch2.13.0-cu126-cp312-linux-x86 --custom-wheel torch2.13.0-cu126-cp312-linux-x86=./dist/custom.whl --reason <why>"},
	},
	{
		Path: []string{"endpoint", "qualify"}, Group: "transfer",
		Summary: "explicitly run one endpoint profile on paid compatible hardware",
		Args:    "<org/endpoint>@<release>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{
			{Name: "--profile", Arg: "<profile>", Summary: "required exact candidate profile"},
			{Name: "--gpu", Arg: "<model>", Summary: "required provider-neutral accelerator model"},
			{Name: "--max-cost", Arg: "<usd>", Summary: "required provider exposure ceiling in USD"},
			{Name: "--duration", Arg: "<duration>", Summary: "qualification lease cap", Default: "15m"},
			{Name: "--timeout", Arg: "<duration>", Summary: "optional caller wait deadline; does not alter the paid lease"},
			{Name: "--reason", Arg: "<why>", Summary: "required; recorded before paid acquisition"},
		},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound, exit.Credential, exit.Unavailable, exit.Deadline, exit.Failed, exit.Conflict, exit.Capacity},
		Capability: "cmd.endpoint.qualify", Status: Implemented, Handler: "endpoint.qualify",
		Next:     []string{"cozy endpoint promote <org/endpoint> <release> --serve <vN/function> --reason <why>"},
		Examples: []string{"cozy endpoint qualify org/endpoint@1.0.0 --profile torch2.13.0-cu130-cp312-linux-x86 --gpu 'NVIDIA GeForce RTX 4090' --max-cost 0.25 --reason <why>"},
	},
	{
		Path: []string{"endpoint", "promote"}, Group: "transfer",
		Summary: "atomically move explicit qualified release serving pointers",
		Args:    "<org/endpoint> <release-id>", MinArgs: 2, MaxArgs: 2,
		Flags: []Flag{
			{Name: "--serve", Arg: "<vN/function>", Summary: "required serving target (repeatable; no bare major)"},
			{Name: "--reason", Arg: "<why>", Summary: "required; recorded before the pointer transaction"},
		},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound, exit.Credential, exit.Unavailable, exit.Deadline, exit.Conflict},
		Capability: "cmd.endpoint.promote", Status: Implemented, Handler: "endpoint.promote",
		Next:     []string{"cozy endpoint show <org/endpoint>"},
		Examples: []string{"cozy endpoint promote org/endpoint 1.0.0 --serve v1/generate --reason <why>", "cozy endpoint promote org/endpoint 1.0.0 --serve v1/generate --serve v1/edit --reason <why>"},
	},

	// ---- account (th-031, Wave 2) ----
	// DEFERRED, not unbuilt-by-accident: Launch-1 tensorhub has NO identity plane
	// (owner ruling, decisions #229) — no accounts, no sessions, no orgs, so there is
	// nothing for a PKCE flow to authenticate against. Catalog reads are public and
	// first-party writes carry TENSORHUB_TOKEN (a value, never a login act).
	// These rows keep the surface honest: `cozy login` says what is true rather than
	// the verb quietly not existing.
	{
		Path: []string{"login"}, Group: "account",
		Summary: "hub login — deferred: Launch 1 has no identity plane; writes use TENSORHUB_TOKEN",
		MaxArgs: 0,
		Flags: []Flag{
			{Name: "--no-browser", Summary: "print the URL + user code and poll"},
			{Name: "--source", Arg: "<hf|civitai|comfy>", Summary: "store a per-source row"},
			{Name: "--token-stdin", Summary: "read the token value from stdin (never argv)"},
		},
		Exits:      []exit.Code{exit.OK, exit.Credential, exit.Unavailable},
		Capability: "cmd.login", Status: Planned, Issue: "th-031",
	},
	{
		Path: []string{"login", "ls"}, Group: "account",
		Summary:    "credential rows — deferred with login; `cozy hub status` shows the configured token",
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK},
		Capability: "cmd.login.ls", Status: Planned, Issue: "th-031",
	},
	{
		Path: []string{"logout"}, Group: "account",
		Summary: "remove a stored credential — deferred with login; nothing is stored yet",
		MaxArgs: 0,
		Flags: []Flag{
			{Name: "--source", Arg: "<name>", Summary: "one source row"},
			{Name: "--all", Summary: "every stored credential"},
		},
		Exits:      []exit.Code{exit.OK},
		Capability: "cmd.logout", Status: Planned, Issue: "th-031",
	},
}
