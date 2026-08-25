package manifest

import "github.com/cozy-creator/cozy-creator-v2/internal/exit"

// FoundationTokens are the capability tokens this binary's CLI foundation carries,
// independent of any one command. Scripts gate on tokens, never version strings.
var FoundationTokens = []string{
	"cli.manifest",         // one declarative surface, startup self-checked
	"cli.noninteractive",   // no interactive prompt anywhere; --yes gates destructive acts
	"errors.typed",         // error(<name>) + remedy + next:
	"exit.matrix.v1",       // the shared cozy-runtime exit matrix, 0-14
	"install.generations",  // installs are immutable generations + a pin per (endpoint, major)
	"install.transaction",  // generation insert and pin activation commit together
	"output.fields",        // --fields
	"output.full",          // --full
	"output.json",          // --json
	"records.libsql",       // ONE local Turso/libSQL lifecycle database
	"service.lock",         // liveness is an OS advisory lock the owner holds
	"worker.protocol.v1",   // the cozy.worker.v1 server over a unix socket
	"coordinator.local",    // local dispatch: request -> attempt -> terminal -> visible output
	"terminal.transaction", // terminal accepted + output visible commit together
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
	},
	{
		Path: []string{"version"}, Group: "meta",
		Summary:    "tag + commit + protocol/contract versions",
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK},
		Capability: "cmd.version", Status: Implemented, Handler: "version",
	},
	{
		Path: []string{"capabilities"}, Group: "meta",
		Summary:    "feature tokens this binary supports — the one gate surface for scripts",
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK},
		Capability: "cmd.capabilities", Status: Implemented, Handler: "capabilities",
	},
	{
		Path: []string{"commands"}, Group: "meta",
		Summary: "the command manifest itself: every command, its status and capability token",
		Args:    "[<prefix>]", MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.Usage},
		Capability: "cmd.commands", Status: Implemented, Handler: "commands",
	},
	{
		Path: []string{"help"}, Group: "meta",
		Summary: "concise help for a command; never mutates, never touches the network",
		Args:    "[<command>]", MaxArgs: 2,
		Exits:      []exit.Code{exit.OK, exit.NotFound},
		Capability: "cmd.help", Status: Implemented, Handler: "help",
	},

	// ---- service (cl-001: the LocalService's own lifecycle) ----
	{
		Path: []string{"up"}, Group: "service",
		Summary: "start the one LocalService (foreground; -d detaches)",
		Flags: []Flag{
			{Name: "--detach", Short: "-d", Summary: "run in the background"},
			{Name: "--port", Arg: "<n>", Summary: "local client API port (default 2699, loopback only)"},
			{Name: "--open", Summary: "open the stub UI page with the per-launch token"},
			{Name: "--yield", Arg: "<smart|always|never>", Summary: "GPU yield policy (default smart)"},
		},
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK, exit.Conflict},
		Capability: "cmd.up", Status: Implemented, Handler: "up",
	},
	{
		Path: []string{"down"}, Group: "service",
		Summary:    "stop the LocalService, draining endpoint processes",
		Flags:      []Flag{{Name: "--timeout", Arg: "<dur>", Summary: "drain bound (default 30s)"}},
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK},
		Capability: "cmd.down", Status: Implemented, Handler: "down",
	},

	// ---- endpoints (cl-009) ----
	{
		Path: []string{"install"}, Group: "endpoints",
		Summary: "resolve, verify, build and pin an endpoint as an immutable generation",
		Args:    "<org/endpoint[@vN]>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{
			{Name: "--from", Arg: "<archive.tar.gz>", Summary: "a local release archive (the pre-hub source door; cl-011 resolves releases instead)"},
			{Name: "--digest", Arg: "<sha256:…>", Summary: "the source digest the release declares — verified before anything executes"},
			{Name: "--prefetch", Summary: "eagerly pull weights"},
			{Name: "--all-variants", Summary: "with --prefetch, pull every admissible lane"},
			{Name: "--force", Summary: "build a new generation and swap the pin"},
			{Name: "--allow-unsigned", Summary: "development-only door past source verification"},
			{Name: "--crash-after", Arg: "<stage>", Summary: "development: SIGKILL after stage|verify|venv|descriptor|activate (crash-matrix verification)"},
		},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound, exit.Credential, exit.Structural, exit.Conflict, exit.Capacity},
		Capability: "cmd.install", Status: Implemented, Handler: "install",
	},
	{
		Path: []string{"ls"}, Group: "endpoints",
		Summary:    "installed endpoints: pin, version, disk (read from the install record)",
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK, exit.Usage},
		Capability: "cmd.ls", Status: Implemented, Handler: "ls",
	},
	{
		Path: []string{"rm"}, Group: "endpoints",
		Summary: "remove an install (venv + pin); the generation record stays until gc",
		Args:    "<org/endpoint[@vN]> …", MinArgs: 1, MaxArgs: -1,
		Flags:      []Flag{yesFlag},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Confirm, exit.Conflict},
		Capability: "cmd.rm", Destructive: true, Status: Implemented, Handler: "rm",
	},
	{
		Path: []string{"gc"}, Group: "endpoints",
		Summary:    "without --yes the reclaim plan (a read); with --yes it executes",
		Flags:      []Flag{yesFlag},
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK, exit.Conflict},
		Capability: "cmd.gc", PlanFirst: true, Status: Implemented, Handler: "gc",
	},
	{
		Path: []string{"start"}, Group: "endpoints",
		Summary: "lifecycle/prewarm: make an endpoint worker resident (never a second invoke path)",
		Args:    "<org/endpoint[@vN]>", MinArgs: 1, MaxArgs: 1,
		Flags:      []Flag{{Name: "--detach", Short: "-d", Summary: "leave it warm in the background"}},
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Unavailable},
		Capability: "cmd.start", NeedsServer: true, Status: Planned, Issue: "cl-010",
	},
	{
		Path: []string{"stop"}, Group: "endpoints",
		Summary: "stop endpoint process(es); blocks until the process group exits",
		Args:    "<org/endpoint> | --all", MaxArgs: 1,
		Flags: []Flag{
			{Name: "--all", Summary: "every endpoint process"},
			{Name: "--timeout", Arg: "<dur>", Summary: "cooperative phase bound"},
		},
		Exits:      []exit.Code{exit.OK, exit.Unavailable},
		Capability: "cmd.stop", NeedsServer: true, Status: Planned, Issue: "cl-010",
	},
	{
		Path: []string{"logs"}, Group: "endpoints",
		Summary: "endpoint process log, or the retained triage bundle for an attempt id",
		Args:    "<org/endpoint | attempt-id>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{
			{Name: "--follow", Short: "-f", Summary: "follow"},
			{Name: "--lines", Short: "-n", Arg: "<count>", Summary: "tail count (default 100)"},
		},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.NotFound, exit.Unavailable},
		Capability: "cmd.logs", NeedsServer: true, Status: Planned, Issue: "cl-010",
	},

	// ---- invocation (cl-010) ----
	{
		Path: []string{"run"}, Group: "invocation",
		Summary: "invoke one function through the coordinator — the one invocation path",
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
			{Name: "--local", Summary: "run on this host (default)"},
			{Name: "--cloud", Summary: "submit to tensorhub under the account"},
		},
		Exits:      []exit.Code{exit.OK, exit.Validation, exit.NotFound, exit.Credential, exit.Structural, exit.OfflineMiss, exit.Unavailable, exit.Deadline, exit.Failed, exit.Canceled, exit.Capacity},
		Capability: "cmd.run", NeedsServer: true, Status: Planned, Issue: "cl-010",
	},
	{
		Path: []string{"describe"}, Group: "invocation",
		Summary: "the endpoint descriptor: installed delegates to the runtime, else the catalog",
		Args:    "<org/endpoint[@vN][/function]>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Unavailable},
		Capability: "cmd.describe", NeedsServer: true, Status: Planned, Issue: "cl-010",
	},
	{
		Path: []string{"doctor"}, Group: "invocation",
		Summary:    "host facts plus per-installed-endpoint fit verdicts",
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK, exit.Unavailable},
		Capability: "cmd.doctor", NeedsServer: true, Status: Planned, Issue: "cl-010",
	},
	{
		Path: []string{"fit"}, Group: "invocation",
		Summary: "fit verdicts for one endpoint/binding: fits / degraded / structural / capacity",
		Args:    "<org/endpoint[@vN]>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{
			{Name: "--model", Arg: "<ref>", Summary: "binding override"},
			{Name: "--lane", Arg: "<l>", Summary: "lane"},
		},
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Unavailable},
		Capability: "cmd.fit", NeedsServer: true, Status: Planned, Issue: "cl-010",
	},
	{
		Path: []string{"search"}, Group: "invocation",
		Summary: "hub catalog discovery (public reads need no login)",
		Args:    "<query>", MinArgs: 1, MaxArgs: -1,
		Flags:      []Flag{{Name: "--kind", Arg: "<endpoint|model>", Summary: "restrict the kind"}},
		Exits:      []exit.Code{exit.OK, exit.Unavailable},
		Capability: "cmd.search", Status: Planned, Issue: "cl-011",
	},

	// ---- jobs (cl-004) ----
	{
		Path: []string{"job", "submit"}, Group: "jobs",
		Summary: "submit a @job; prints the job id",
		Args:    "<org/endpoint/vN/function> [key=value …]", MinArgs: 1, MaxArgs: -1,
		Flags: []Flag{
			{Name: "--local", Summary: "run on this host"},
			{Name: "--cloud", Summary: "submit to tensorhub"},
			{Name: "--model", Arg: "<ref>", Summary: "override a model binding (repeatable)"},
			{Name: "--out", Arg: "<dir>", Summary: "output directory"},
			{Name: "--in", Arg: "<file>", Summary: "whole payload as JSON"},
			{Name: "--follow", Summary: "attach immediately and exit with the terminal mapping"},
		},
		Exits:      []exit.Code{exit.OK, exit.Validation, exit.NotFound, exit.Unavailable, exit.Deadline, exit.Failed, exit.Canceled},
		Capability: "cmd.job.submit", NeedsServer: true, Terminals: true, Status: Planned, Issue: "cl-004",
	},
	{
		Path: []string{"job", "status"}, Group: "jobs",
		Summary: "state, progress, elapsed, the running bill where a rate exists, terminal",
		Args:    "<job-id>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Unavailable},
		Capability: "cmd.job.status", NeedsServer: true, Status: Planned, Issue: "cl-004",
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
		Capability: "cmd.job.ls", NeedsServer: true, Status: Planned, Issue: "cl-004",
	},
	{
		Path: []string{"job", "follow"}, Group: "jobs",
		Summary: "attach to the progress/metric stream; exits with the terminal mapping",
		Args:    "<job-id>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Unavailable, exit.Deadline, exit.Failed, exit.Canceled},
		Capability: "cmd.job.follow", NeedsServer: true, Terminals: true, Status: Planned, Issue: "cl-004",
	},
	{
		Path: []string{"job", "cancel"}, Group: "jobs",
		Summary: "request cancel and block until the canceled terminal",
		Args:    "<job-id>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Unavailable},
		Capability: "cmd.job.cancel", NeedsServer: true, Status: Planned, Issue: "cl-004",
	},

	// ---- transfer (cl-012, cl-008) ----
	{
		Path: []string{"pull"}, Group: "transfer",
		Summary: "fetch checkpoints/adapters into the shared local CAS",
		Args:    "<ref> …", MinArgs: 1, MaxArgs: -1,
		Flags: []Flag{
			{Name: "--lane", Arg: "<label>", Summary: "lane"},
			{Name: "--all-variants", Summary: "every admissible lane"},
			{Name: "--dry-run", Summary: "print the plan without moving bytes"},
			{Name: "--token", Arg: "<t>", Summary: "this invocation's source credential"},
		},
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Credential},
		Capability: "cmd.pull", NeedsServer: true, Status: Planned, Issue: "cl-012",
	},
	{
		Path: []string{"push"}, Group: "transfer",
		Summary: "declare-first upload of models, LoRAs and datasets",
		Args:    "<org/repo> <path|local-ref> …", MinArgs: 2, MaxArgs: -1,
		Flags:      []Flag{{Name: "--dry-run", Summary: "print the transfer plan"}},
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Credential},
		Capability: "cmd.push", NeedsServer: true, Status: Planned, Issue: "cl-012",
	},
	{
		Path: []string{"datasets", "push"}, Group: "transfer",
		Summary: "the dataset dialect of push",
		Args:    "<path> <org/repo>", MinArgs: 2, MaxArgs: 2,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Credential},
		Capability: "cmd.datasets.push", NeedsServer: true, Status: Planned, Issue: "cl-012",
	},
	{
		Path: []string{"datasets", "pull"}, Group: "transfer",
		Summary: "the dataset dialect of pull; --out materializes a tree from the CAS",
		Args:    "<org/repo[@release|@digest]>", MinArgs: 1, MaxArgs: 1,
		Flags:      []Flag{{Name: "--out", Arg: "<dir>", Summary: "materialize here"}},
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Credential},
		Capability: "cmd.datasets.pull", NeedsServer: true, Status: Planned, Issue: "cl-012",
	},
	{
		Path: []string{"export"}, Group: "transfer",
		Summary: "authorized local checkpoint export through the coordinator surface",
		Args:    "<ref>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Unavailable},
		Capability: "cmd.export", NeedsServer: true, Status: Planned, Issue: "cl-008",
	},
	{
		Path: []string{"deploy"}, Group: "transfer",
		Summary: "publish a release: git manifest, declare digests, then follow the build",
		MaxArgs: 0,
		Flags: []Flag{
			{Name: "--create", Summary: "required for the first publish of a new name"},
			{Name: "--detach", Summary: "return after upload"},
		},
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Credential, exit.Failed},
		Capability: "cmd.deploy", Status: Planned, Issue: "cl-012",
	},
	{
		Path: []string{"promote"}, Group: "transfer",
		Summary: "move the serving pointer; rollback is promote with an older release",
		Args:    "<org/endpoint> <release-id>", MinArgs: 2, MaxArgs: 2,
		Flags:      []Flag{{Name: "--serve", Arg: "<major>", Summary: "required; no default"}},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.NotFound, exit.Credential},
		Capability: "cmd.promote", Status: Planned, Issue: "cl-012",
	},

	// ---- account (cl-011) ----
	{
		Path: []string{"login"}, Group: "account",
		Summary: "browser/device PKCE against the hub; nothing reads stdin as a prompt",
		MaxArgs: 0,
		Flags: []Flag{
			{Name: "--no-browser", Summary: "print the URL + user code and poll"},
			{Name: "--source", Arg: "<hf|civitai|comfy>", Summary: "store a per-source row"},
			{Name: "--token-stdin", Summary: "read the token value from stdin (never argv)"},
		},
		Exits:      []exit.Code{exit.OK, exit.Credential, exit.Unavailable},
		Capability: "cmd.login", Status: Planned, Issue: "cl-011",
	},
	{
		Path: []string{"login", "ls"}, Group: "account",
		Summary:    "credential rows: source, kind, state",
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK},
		Capability: "cmd.login.ls", Status: Planned, Issue: "cl-011",
	},
	{
		Path: []string{"logout"}, Group: "account",
		Summary: "remove the hub session, a per-source row, or everything",
		MaxArgs: 0,
		Flags: []Flag{
			{Name: "--source", Arg: "<name>", Summary: "one source row"},
			{Name: "--all", Summary: "every stored credential"},
		},
		Exits:      []exit.Code{exit.OK},
		Capability: "cmd.logout", Status: Planned, Issue: "cl-011",
	},
}
