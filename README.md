# Cozy

Cozy runs generative-media packages on your own machines and on rented GPUs. One `cozy` binary
carries the command line, Calcifer (the local controller), TensorD (the machine) and TensorFS
(the weight store).

## Install

Linux and macOS:

```sh
curl -LsSf https://tensorhub.com/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://tensorhub.com/install.ps1 | iex
```

Both put `cozy` in `~/.cozy/bin` and on your PATH for new terminals: open a new one, or run the line
the installer prints. `cozy update` moves to the newest release.

## Use

Write the request in a YAML file, `request.yaml`:

```yaml
prompt: a lighthouse on a basalt cliff at dusk, oil painting
aspect_ratio: "16:9"
steps: 30
seed: 7
```

```sh
cozy login you@example.com
cozy run sdxl/generate --input request.yaml
cozy run sdxl/generate --input request.yaml seed=8      # an inline input overrides the file
cozy run birefnet/remove_background --asset image=./character.png
cozy run sdxl/generate --describe                       # the fields a function takes
```

Inline `key=value` inputs are the quick form: `cozy run sdxl/generate prompt='a fox in fresh snow'`.
Files go in with `--asset` (or by path inside the YAML, relative to it). A run is followed to its end;
Ctrl-C stops following, never the run, and `cozy run show <N>` picks it up again.

```sh
cozy machine rent            # what is for rent; then cozy machine rent <sku>
cozy model search sdxl
```

First-party packages and models have no owner (`sdxl/generate`, `flux-3-klein@0.1.0`); everyone
else's are `owner/name` (`you/my-pack/generate`). cozy 0.11.13 and older need the old spelling,
`fidika/sdxl/generate`; `cozy update` moves to a release that takes both.

`cozy login` signs you up on first use, with a code it emails. Runs go to this computer's machine,
which starts when used and stops after 15 minutes idle. Without an NVIDIA GPU here (a Mac, say),
add `--on new` to run on a rented one. Outputs land in `~/.cozy/outputs`.

`cozy --help` and `cozy <verb> --help` are the reference.

`cozy doctor` diagnoses this computer's cozy (Calcifer, its service, the machine, tools, GPUs) and
restarts a Calcifer that is stuck or runs an older cozy.

## Remove cozy

End your rentals first (`cozy machine list`, `cozy machine end <name>`). Then:

Linux:

```sh
systemctl --user disable --now cozy-calcifer cozy-machine
pkill -f ~/.cozy/bin/cozy      # what they started (a machine, its executors)
rm -rf ~/.cozy ~/.config/cozy ~/.local/share/bash-completion/completions/cozy ~/.config/fish/completions/cozy.fish
```

macOS:

```sh
launchctl bootout gui/$(id -u)/com.cozy-creator.calcifer; launchctl bootout gui/$(id -u)/com.cozy-creator.machine
pkill -f ~/.cozy/bin/cozy
rm -rf ~/.cozy ~/.config/cozy ~/Library/LaunchAgents/com.cozy-creator.*.plist
```

Windows (PowerShell):

```powershell
Remove-ItemProperty HKCU:\Software\Microsoft\Windows\CurrentVersion\Run cozy-calcifer, cozy-machine -ErrorAction SilentlyContinue
Get-Process cozy -ErrorAction SilentlyContinue | Stop-Process
Remove-Item -Recurse -Force ~\.cozy, $env:LOCALAPPDATA\cozy
$path = ([Environment]::GetEnvironmentVariable('Path', 'User') -split ';') -ne "$HOME\.cozy\bin"
[Environment]::SetEnvironmentVariable('Path', $path -join ';', 'User')
```

On Linux and macOS, delete the line that sources `~/.config/cozy/env.sh` (or `env.fish`) from `.profile`,
`.bashrc`, `.zshrc` or fish's `conf.d`. `~/.cozy` holds the machine's model store, which can be large;
a store set elsewhere in `~/.cozy/config.toml` stays until you delete it. Homebrew: `brew uninstall
cozy`, then the same steps.

## Build

```sh
cargo build --release -p cozy
```

The workspace's crates live in `crates/`: `cozy` (the binary), `cli`, `calcifer`, `tensord`,
`tensorfs`, `varena`, `tensorfs-py`, `hub-client`, `protocol` and `platform`.
`crates/tensorfs` is the Rust library and `tfs` CLI; `crates/tensorfs-py` contains its PyO3 bindings,
Python package and wheel configuration together. The browser player is `web/player`.
