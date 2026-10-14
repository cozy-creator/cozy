package producttest

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestAnUnlockedPackageSuggestsALock: a human who installs or runs a package directory with no
// uv.lock is told once a day per directory that `cozy package lock` pins it. A locked package
// and --json never are.
func TestAnUnlockedPackageSuggestsALock(t *testing.T) {
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv locks the fixture")
	}
	const tip = "tip: `cozy package lock` pins this package's dependencies"
	project := func(t *testing.T, locked bool) string {
		t.Helper()
		dir := filepath.Join(t.TempDir(), "lock-tip")
		must(t, os.MkdirAll(dir, 0o755))
		write := func(name, body string) { must(t, os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600)) }
		write("pyproject.toml", "[project]\nname=\"lock-tip\"\nversion=\"0.1.0\"\nrequires-python=\">=3.12,<3.13\"\n"+
			"dependencies=[]\n[project.entry-points.\"cozy.application\"]\ndefault=\"lock_tip:app\"\n"+
			"[build-system]\nrequires=[\"hatchling\"]\nbuild-backend=\"hatchling.build\"\n"+
			"[tool.hatch.build.targets.wheel]\nonly-include=[\"lock_tip.py\"]\n")
		write("package.toml", "[application]\nobject=\"lock_tip:app\"\n")
		write("lock_tip.py", "import msgspec\nfrom cozy_runtime.author import App, Context, invocable\n"+
			"class Result(msgspec.Struct):\n    count: int\n"+
			"@invocable\nasync def count(ctx: Context, *, count: int) -> Result:\n    return Result(count)\n"+
			"app=App()\napp.job(count, accelerator=False)\n")
		if locked {
			if data, err := exec.Command(uv, "lock", "--no-progress", "--project", dir).CombinedOutput(); err != nil {
				t.Fatalf("lock fixture: %v\n%s", err, data)
			}
		}
		return dir
	}
	read := func(t *testing.T, root string, args ...string) string {
		t.Helper()
		code, stdout, stderr := runCozyStreams(t, root, args...)
		if code != 0 {
			t.Fatalf("cozy %s [exit %d]\n%s%s", strings.Join(args, " "), code, stdout, stderr)
		}
		return stdout + stderr
	}
	tips := func(t *testing.T, want int, out string) {
		t.Helper()
		if got := strings.Count(out, tip); got != want {
			t.Fatalf("the lock tip appeared %d times, want %d\n%s", got, want, out)
		}
	}

	locked, root := project(t, true), t.TempDir()
	tips(t, 0, read(t, root, "package", "install", locked))
	tips(t, 0, read(t, root, "run", locked, "--describe"))

	unlocked, root := project(t, false), t.TempDir()
	tips(t, 0, read(t, root, "package", "install", unlocked, "--json"))
	tips(t, 0, read(t, root, "run", unlocked, "--describe", "--json"))

	root = t.TempDir()
	tips(t, 1, read(t, root, "package", "install", unlocked))
	tips(t, 0, read(t, root, "run", unlocked, "--describe"))
	module := filepath.Join(unlocked, "lock_tip.py")
	source, err := os.ReadFile(module)
	must(t, err)
	must(t, os.WriteFile(module, append(source, []byte("# an edit the next install reads\n")...), 0o600))
	out := read(t, root, "package", "install", unlocked)
	if !strings.Contains(out, "installed") || strings.Contains(out, "already installed") {
		t.Fatalf("the edited package was not read again\n%s", out)
	}
	tips(t, 0, out)
	tips(t, 1, read(t, root, "run", project(t, false), "--describe"))
}
