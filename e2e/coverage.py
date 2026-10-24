"""Every cozy command and flag has a test, and every test ran.

    coverage.py tree    <cozy>                                  the command tree, one id per line
    coverage.py check   <cozy> <coverage.toml> --tiers T[,T] [--log ARGV.LOG]...

The tree is read from the binary under test (`--help` of every command), so a new command or flag
is seen the moment it exists. An id is `cozy <command path>` or `cozy <command path> --flag`; a flag
every command takes (`--json`, `--tensorhub`, `--yes`) is the root's: `cozy --json`.

`check` fails when the tree holds an id coverage.toml does not map, when coverage.toml maps an id the
tree no longer has, or (with --log) when an id mapped to one of --tiers was not exercised: no logged
invocation named that command with that flag. A log holds one record per call, its words separated by
\x1f and ended by \x1e (e2e/lib.sh's `cozy`, e2e/bin/cozy), so a word may hold any text.
"""

import subprocess
import sys
import tomllib
from dataclasses import dataclass, field

TIERS = {"pr", "release", "gate", "manual", "none"}
# Shown by every --help, exercised by the walk itself.
WALKED = {"--help", "--version"}


@dataclass
class Command:
    path: tuple[str, ...]
    runnable: bool = True  # a group whose usage reads `[COMMAND]` runs on its own too (`cozy credits`)
    flags: dict[str, bool] = field(default_factory=dict)  # long flag -> takes a value
    shorts: dict[str, str] = field(default_factory=dict)  # -f -> --follow
    children: dict[str, "Command"] = field(default_factory=dict)


def _help(cozy: str, path: tuple[str, ...]) -> str:
    p = subprocess.run([cozy, *path, "--help"], capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"cozy {' '.join(path)} --help exited {p.returncode}: {p.stderr.strip()}")
    return p.stdout


def _parse(text: str) -> tuple[list[str], dict[str, bool], dict[str, str], bool]:
    commands, flags, shorts, section, runnable = [], {}, {}, "", True
    for line in text.splitlines():
        if line.startswith("Usage:"):
            runnable = "<COMMAND>" not in line
        if line and not line[0].isspace():
            section = line.rstrip(":")
            continue
        words = line.split()
        if not words:
            continue
        if section == "Commands" and line.startswith("  ") and not line.startswith("   "):
            if words[0] != "help":
                commands.append(words[0])
        elif section == "Options" and words[0].startswith("-"):
            short = words[0].rstrip(",") if not words[0].startswith("--") else ""
            long = next((w for w in words if w.startswith("--")), "")
            if not long:
                continue
            after = words[words.index(long) + 1] if words.index(long) + 1 < len(words) else ""
            flags[long] = after.startswith("<")
            if short:
                shorts[short] = long
    return commands, flags, shorts, runnable


def tree(cozy: str) -> Command:
    def walk(path: tuple[str, ...]) -> Command:
        commands, flags, shorts, runnable = _parse(_help(cozy, path))
        node = Command(path, runnable, flags, shorts)
        for name in commands:
            node.children[name] = walk((*path, name))
        return node

    return walk(())


def ids(root: Command) -> set[str]:
    out, global_flags = set(), set(root.flags) - WALKED

    def add(node: Command) -> None:
        name = " ".join(("cozy", *node.path))
        if node.path and node.runnable:
            out.add(name)
        for flag in node.flags:
            if flag in WALKED:
                continue
            out.add(f"cozy {flag}" if flag in global_flags else f"{name} {flag}")
        for child in node.children.values():
            add(child)

    add(root)
    return out


def calls(log: str) -> list[list[str]]:
    with open(log, encoding="utf-8", errors="replace") as f:
        return [record.split("\x1f")[:-1] for record in f.read().split("\x1e") if record.strip("\x1f\n")]


def exercised(root: Command, words: list[str]) -> set[str]:
    """The ids one logged invocation exercised: its command, and each flag it passed."""
    node, out, i, descend = root, set(), 0, True
    while i < len(words):
        word = words[i]
        i += 1
        if word.startswith("-") and word != "-":
            flag, _, inline = word.partition("=")
            flag = node.shorts.get(flag, root.shorts.get(flag, flag))
            takes = node.flags.get(flag, root.flags.get(flag))
            if flag in root.flags and flag not in WALKED:
                out.add(f"cozy {flag}")
            elif flag in node.flags and flag not in WALKED:
                out.add(" ".join(("cozy", *node.path, flag)))
            if takes and not inline:
                i += 1
            continue
        if descend and word in node.children:
            node = node.children[word]
            continue
        descend = False
    if node.path and node.runnable:
        out.add(" ".join(("cozy", *node.path)))
    return out


def manifest(path: str) -> dict[str, tuple[str, str]]:
    """coverage.toml: `[command]` tables (`"run list"`), `test` for the command itself and one key per
    flag, each `"tier: the step that tests it"`; `[global]` for the root's flags."""
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    out = {}
    for command, entries in doc.items():
        prefix = "cozy" if command == "global" else f"cozy {command}"
        for key, value in entries.items():
            tier, _, step = value.partition(":")
            tier, step = tier.strip(), step.strip()
            if tier not in TIERS or not step:
                sys.exit(f"coverage.toml [{command}] {key} = {value!r}: write \"<{'|'.join(sorted(TIERS))}>: <step>\"")
            out[prefix if key == "test" else f"{prefix} {key}"] = (tier, step)
    return out


def check(cozy: str, toml: str, tiers: set[str], logs: list[str]) -> int:
    root, mapped = tree(cozy), manifest(toml)
    have = ids(root)
    problems = [f"no test: {i} (map it in e2e/coverage.toml)" for i in sorted(have - set(mapped))]
    problems += [f"stale: {i} is in e2e/coverage.toml but not in this cozy" for i in sorted(set(mapped) - have)]
    seen: set[str] = set()
    for log in logs:
        for words in calls(log):
            seen |= exercised(root, words)
    due = {i for i, (tier, _) in mapped.items() if tier in tiers and i in have}
    if logs:
        problems += [f"did not run: {i} ({mapped[i][0]}: {mapped[i][1]})" for i in sorted(due - seen)]
    untested = sorted((i, t, s) for i, (t, s) in mapped.items() if t in ("manual", "none") and i in have)
    print(f"{len(have)} commands and flags; {len(due)} due in tier(s) {','.join(sorted(tiers))}; "
          f"{len(due & seen)} exercised; {len(untested)} not automated")
    for i, tier, step in untested:
        print(f"  {tier}: {i}: {step}")
    for p in problems:
        print(f"::error::coverage: {p}")
    return 1 if problems else 0


def main() -> int:
    args = sys.argv[1:]
    if args[:1] == ["tree"] and len(args) == 2:
        print("\n".join(sorted(ids(tree(args[1])))))
        return 0
    if args[:1] == ["check"] and len(args) >= 5 and args[3] == "--tiers":
        logs = [args[i + 1] for i in range(5, len(args) - 1) if args[i] == "--log"]
        return check(args[1], args[2], set(args[4].split(",")), logs)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
