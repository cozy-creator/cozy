package manifest

import "github.com/cozy-creator/cozy-creator-v2/internal/exit"

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
	"coordinator.local",      // local dispatch: request -> attempt -> terminal -> visible output
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
			{Name: "--from", Arg: "<archive.tar.gz>", Summary: "a local release archive (the pre-hub source door; hub release resolve lands with th-003)"},
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
		Flags: []Flag{
			{Name: "--detach", Short: "-d", Summary: "leave it warm in the background"},
			{Name: "--no-warm", Summary: "skip the boot warm pass (an entrypoint whose warm shape does not fit degrades its binding for nothing)"},
		},
		Exits: []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound, exit.Structural,
			exit.Unavailable, exit.Deadline, exit.Failed, exit.Conflict},
		Capability: "cmd.start", NeedsServer: true, Status: Implemented, Handler: "start",
	},
	{
		Path: []string{"stop"}, Group: "endpoints",
		Summary: "stop endpoint process(es); blocks until the process group exits",
		Args:    "<org/endpoint> | --all", MaxArgs: 1,
		Flags: []Flag{
			{Name: "--all", Summary: "every endpoint process"},
			{Name: "--timeout", Arg: "<dur>", Summary: "cooperative phase bound"},
		},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Unavailable},
		Capability: "cmd.stop", NeedsServer: true, Status: Implemented, Handler: "stop",
	},
	{
		Path: []string{"logs"}, Group: "endpoints",
		Summary: "endpoint process log, or the retained triage bundle for an attempt id",
		Args:    "<org/endpoint | attempt-id>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{
			{Name: "--follow", Short: "-f", Summary: "follow"},
			{Name: "--lines", Short: "-n", Arg: "<count>", Summary: "tail count (default 100)"},
		},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.NotFound, exit.Unavailable, exit.Conflict},
		Capability: "cmd.logs", NeedsServer: true, Status: Implemented, Handler: "logs",
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
			// One key names one request forever. A bare `run` mints its own, so retry
			// safety across a process restart is the caller's explicit act.
			{Name: "--idempotency-key", Arg: "<k>", Summary: "reuse a key: the same key returns the same request"},
		},
		Exits: []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound, exit.Credential,
			exit.Structural, exit.OfflineMiss, exit.Unavailable, exit.Deadline, exit.Failed,
			exit.Canceled, exit.Conflict, exit.Capacity},
		Capability: "cmd.run", NeedsServer: true, Terminals: true, Status: Implemented, Handler: "run",
	},
	{
		Path: []string{"describe"}, Group: "invocation",
		Summary: "the endpoint descriptor: the surface this release's own runtime derived and the install verified",
		Args:    "<org/endpoint[@vN][/function]>", MinArgs: 1, MaxArgs: 1,
		Exits: []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound, exit.Structural, exit.Conflict},
		// A records-plane read (cl-009's D3 rule): the descriptor is a verified document
		// in the generation, so the LocalService is not on the path to reading it.
		Capability: "cmd.describe", Status: Implemented, Handler: "describe",
	},
	{
		Path: []string{"doctor"}, Group: "invocation",
		Summary:    "host facts plus per-installed-endpoint fit verdicts",
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK, exit.Unavailable},
		Capability: "cmd.doctor", NeedsServer: true, Status: Implemented, Handler: "doctor",
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
			{Name: "--org", Arg: "<name>", Summary: "the org whose scratch repo this job publishes into (default local)"},
			{Name: "--in", Arg: "<file>", Summary: "whole payload as JSON"},
			{Name: "--idempotency-key", Arg: "<key>", Summary: "name this job forever; a repeat answers the same one"},
			{Name: "--follow", Summary: "attach immediately and exit with the terminal mapping"},
		},
		Exits:      []exit.Code{exit.OK, exit.Validation, exit.NotFound, exit.Unavailable, exit.Deadline, exit.Failed, exit.Canceled},
		Capability: "cmd.job.submit", NeedsServer: true, Terminals: true, Status: Implemented, Handler: "job.submit",
	},
	{
		Path: []string{"job", "status"}, Group: "jobs",
		Summary: "state, progress, elapsed, the running bill where a rate exists, terminal",
		Args:    "<job-id>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Unavailable},
		Capability: "cmd.job.status", NeedsServer: true, Status: Implemented, Handler: "job.status",
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
	},
	{
		Path: []string{"job", "follow"}, Group: "jobs",
		Summary: "attach to the progress/metric stream; exits with the terminal mapping",
		Args:    "<job-id>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Unavailable, exit.Deadline, exit.Failed, exit.Canceled},
		Capability: "cmd.job.follow", NeedsServer: true, Terminals: true, Status: Implemented, Handler: "job.follow",
	},
	{
		Path: []string{"job", "cancel"}, Group: "jobs",
		Summary: "request cancel and block until the canceled terminal",
		Args:    "<job-id>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Unavailable},
		Capability: "cmd.job.cancel", NeedsServer: true, Status: Implemented, Handler: "job.cancel",
	},

	// ---- catalog (cl-011) ----
	// Launch-1 tensorhub is a headless distribution hub with NO identity plane
	// (decisions #229): reads are public and carry no credential, first-party writes
	// carry the ONE static admin token from TENSORHUB_TOKEN.
	{
		Path: []string{"search"}, Group: "catalog",
		Summary: "hub catalog discovery — public, no credential",
		Args:    "[<query>]", MaxArgs: -1,
		Flags:      []Flag{{Name: "--kind", Arg: "<endpoint|model>", Summary: "restrict the kind"}},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Unavailable, exit.Deadline},
		Capability: "cmd.search", Status: Implemented, Handler: "search",
	},
	{
		Path: []string{"repo", "show"}, Group: "catalog",
		Summary: "resolve one repo ref against the catalog — public, no credential",
		Args:    "<org/name>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.NotFound, exit.Unavailable, exit.Deadline},
		Capability: "cmd.repo.show", Status: Implemented, Handler: "repo.show",
	},
	{
		Path: []string{"repo", "create"}, Group: "catalog",
		Summary: "create a model or endpoint repo — first-party, admin token",
		Args:    "<org/name>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{
			{Name: "--kind", Arg: "<model|endpoint>", Summary: "required; the hub locks a repo's kind at creation"},
			{Name: "--reason", Arg: "<why>", Summary: "required; the hub records it durably before it acts"},
		},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.Credential, exit.Unavailable, exit.Deadline, exit.Conflict},
		Capability: "cmd.repo.create", Status: Implemented, Handler: "repo.create",
	},
	{
		Path: []string{"hub", "status"}, Group: "catalog",
		Summary: "the configured hub: URL, credential digest, reachability, catalog size",
		MaxArgs: 0,
		// Content-first like bare `cozy`: an unreachable hub is a STATE this verb
		// reports, not a refusal it raises.
		Exits:      []exit.Code{exit.OK},
		Capability: "cmd.hub.status", Status: Implemented, Handler: "hub.status",
	},
	{
		Path: []string{"hub", "config"}, Group: "catalog",
		Summary:    "the hub's running config with per-key provenance — admin token; secrets digested",
		MaxArgs:    0,
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.Credential, exit.Unavailable, exit.Deadline},
		Capability: "cmd.hub.config", Status: Implemented, Handler: "hub.config",
	},

	// ---- transfer (cl-012, cl-008) ----
	// cl-012's two transfer verbs are records-plane acts, not process-plane ones:
	// they move bytes between the local canonical store and the hub and never touch
	// an endpoint process, so neither needs the LocalService (cl-009's D3 rule).
	{
		Path: []string{"pull"}, Group: "transfer",
		Summary: "fetch a hub checkpoint into the shared local store — verified, resumable",
		Args:    "<org/repo[@sha256:…]>", MinArgs: 1, MaxArgs: 1,
		Flags: []Flag{
			{Name: "--dry-run", Summary: "print the plan without moving bytes"},
			// cl-011's law, enforced by the `secret` fence family: a credential never
			// rides argv. cozy-creator.md still spells this `--token <t>`; the doc is
			// the drift, not this row.
			{Name: "--token-stdin", Summary: "read this invocation's hub credential from stdin (never argv)"},
			{Name: "--crash-after", Arg: "<n>", Summary: "development: stop after n objects (resume verification)"},
		},
		Exits: []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound, exit.Credential,
			exit.Structural, exit.Unavailable, exit.Deadline},
		Capability: "cmd.pull", Status: Implemented, Handler: "pull",
	},
	{
		Path: []string{"push"}, Group: "transfer",
		Summary: "declare-first upload of a local canonical snapshot; only missing bytes move",
		Args:    "<org/repo> <sha256:snapshot>", MinArgs: 2, MaxArgs: 2,
		Flags: []Flag{
			// No `--family`: th-003's hub CLASSIFIES the family from the artifact's
			// topology digest and refuses an unknown field, so a declared family is
			// both unnecessary and fatal (cl-006's real-hub side-check, closed here).
			{Name: "--reason", Arg: "<why>", Summary: "required; the hub records it durably before it acts"},
			{Name: "--session", Arg: "<name>", Summary: "publish session name (default derived from the snapshot)"},
			{Name: "--dry-run", Summary: "declare and stop: the hub's own transfer plan"},
			{Name: "--token-stdin", Summary: "read this invocation's hub credential from stdin (never argv)"},
			{Name: "--crash-after", Arg: "<n>", Summary: "development: stop after n objects (resume verification)"},
		},
		Exits: []exit.Code{exit.OK, exit.Usage, exit.Validation, exit.NotFound, exit.Credential,
			exit.Structural, exit.Unavailable, exit.Deadline, exit.Failed, exit.Conflict},
		Capability: "cmd.push", Status: Implemented, Handler: "push",
	},
	{
		Path: []string{"datasets", "push"}, Group: "transfer",
		Summary: "the dataset dialect of push — deferred: there is no datasets plane to push into",
		Args:    "<path> <org/repo>", MinArgs: 2, MaxArgs: 2,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Credential},
		Capability: "cmd.datasets.push", Status: Planned, Issue: "th-035",
	},
	{
		Path: []string{"datasets", "pull"}, Group: "transfer",
		Summary: "the dataset dialect of pull — deferred with the datasets plane",
		Args:    "<org/repo[@release|@digest]>", MinArgs: 1, MaxArgs: 1,
		Flags:      []Flag{{Name: "--out", Arg: "<dir>", Summary: "materialize here"}},
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Credential},
		Capability: "cmd.datasets.pull", Status: Planned, Issue: "th-035",
	},
	{
		Path: []string{"export"}, Group: "transfer",
		Summary: "authorized local checkpoint export through the coordinator surface",
		Args:    "<ref>", MinArgs: 1, MaxArgs: 1,
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Unavailable},
		Capability: "cmd.export", NeedsServer: true, Status: Planned, Issue: "cl-008",
	},
	// `deploy` and `promote` publish and point at an ENDPOINT RELEASE. Neither
	// referent exists yet: releases arrive with th-003 (the hub's own promote route
	// refuses `promote.not_armed` by name today) and the leased build that turns a
	// source snapshot into a release arrives with th-004. Advertised, not built —
	// a clean-commit snapshot uploaded into a plane with no release to cut would be
	// machinery with no consumer (law 13).
	{
		Path: []string{"deploy"}, Group: "transfer",
		Summary: "publish an endpoint release — deferred: there is no release plane to cut into",
		MaxArgs: 0,
		Flags: []Flag{
			{Name: "--create", Summary: "required for the first publish of a new name"},
			{Name: "--detach", Summary: "return after upload"},
		},
		Exits:      []exit.Code{exit.OK, exit.NotFound, exit.Credential, exit.Failed},
		Capability: "cmd.deploy", Status: Planned, Issue: "th-003",
	},
	{
		Path: []string{"promote"}, Group: "transfer",
		Summary: "move the serving pointer — deferred: the hub refuses promote.not_armed until releases exist",
		Args:    "<org/endpoint> <release-id>", MinArgs: 2, MaxArgs: 2,
		Flags:      []Flag{{Name: "--serve", Arg: "<major>", Summary: "required; no default"}},
		Exits:      []exit.Code{exit.OK, exit.Usage, exit.NotFound, exit.Credential},
		Capability: "cmd.promote", Status: Planned, Issue: "th-003",
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
