package producttest

import (
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// bashWordBreaks is bash's default COMP_WORDBREAKS.
const bashWordBreaks = " \t\n\"'><=;|&(:"

// The real binary answers the shell protocol from the Kong grammar and this root's
// records, read-only: no daemon starts and nothing is written.
func TestShellCompletion(t *testing.T) {
	root, work := t.TempDir(), t.TempDir()
	seedCompletionRoot(t, root)
	must(t, os.MkdirAll(filepath.Join(work, "sub", "deep"), 0o755))
	for _, name := range []string{"a.png", "b.json", "sub/c.png", "my file.png"} {
		must(t, os.WriteFile(filepath.Join(work, name), nil, 0o644))
	}
	complete := func(shell, line string) []string {
		t.Helper()
		args := []string{"-n", "19", cozyBin, "__complete", shell, line}
		if shell == "bash" {
			args = append(args, bashWordBreaks)
		}
		cmd := exec.Command("/usr/bin/nice", args...)
		cmd.Dir, cmd.Env = work, childEnv(t, root)
		var stderr strings.Builder
		cmd.Stderr = &stderr
		out, err := cmd.Output()
		if err != nil || stderr.Len() > 0 {
			t.Fatalf("%s %q: %v %s", shell, line, err, stderr.String())
		}
		return strings.Fields(strings.ReplaceAll(strings.TrimSpace(string(out)), `\ `, "\x00"))
	}
	for _, c := range []struct {
		shell, line string
		want        []string
	}{
		{"bash", "cozy ru", []string{"run"}},
		{"bash", "cozy rent", []string{"rental"}},
		{"bash", "cozy rental en", []string{"end"}},
		{"bash", "cozy completion ", []string{"bash", "zsh", "fish"}},
		{"bash", "cozy run proof/dryrun/prepare --as", []string{"--asset", "--asset-fidelity"}},
		{"bash", "cozy run --asset=", []string{"a.png", "b.json", "my\x00file.png", "sub/"}},
		{"zsh", "cozy run x --asset=s", []string{"--asset=sub/"}},
		{"bash", "cozy run x --asset ./sub/", []string{"./sub/c.png", "./sub/deep/"}},
		{"bash", "cozy run x --asset lbl=s", []string{"sub/"}},
		{"bash", "cozy run x --asset=lbl=", []string{"a.png", "b.json", "my\x00file.png", "sub/"}},
		{"zsh", "cozy run x --asset field.image=sub/c", []string{"field.image=sub/c.png"}},
		{"fish", "cozy run x --asset=lbl=b", []string{"--asset=lbl=b.json"}},
		{"bash", "cozy run x --out ", []string{"sub/"}},
		{"bash", "cozy run x --out=./sub/", []string{"./sub/deep/"}},
		{"bash", "cozy run x --input ", []string{"a.png", "b.json", "my\x00file.png", "sub/"}},
		{"bash", "cozy run proof/", []string{"proof/dryrun/prepare"}},
		{"bash", "cozy run x --rental ", []string{"loran"}},
		{"bash", "cozy rental end ", []string{"loran"}},
		{"bash", "cozy rent ssh-info l", []string{"loran"}},
		{"bash", "cozy package remove ", []string{"proof/dryrun"}},
		{"bash", "cozy run x --timeout ", nil},
	} {
		if got := complete(c.shell, c.line); !slices.Equal(got, c.want) {
			t.Errorf("%s %q = %q, want %q", c.shell, c.line, got, c.want)
		}
	}
	if got := complete("bash", "cozy run "); !slices.Contains(got, "cancel") || !slices.Contains(got, "proof/dryrun/prepare") {
		t.Errorf("cozy run <TAB> = %q, want its verbs and installed callables", got)
	}
	if _, err := os.Stat(filepath.Join(root, "daemon.lock")); !os.IsNotExist(err) {
		t.Fatalf("completion touched the daemon: %v", err)
	}

	// The printed bash script drives the same protocol from COMP_LINE/COMP_POINT.
	line := "cozy run x --asset=lbl=s"
	script := `source <(cozy completion bash) && COMP_LINE="$1" COMP_POINT=${#1} && _cozy 2>/dev/null; printf '%s\n' "${COMPREPLY[@]}"`
	cmd := exec.Command("/usr/bin/nice", "-n", "19", "bash", "--norc", "--noprofile", "-c", script, "bash", line)
	cmd.Dir = work
	cmd.Env = childEnv(t, root)
	for i, entry := range cmd.Env {
		if strings.HasPrefix(entry, "PATH=") {
			cmd.Env[i] = "PATH=" + filepath.Dir(cozyBin) + ":" + strings.TrimPrefix(entry, "PATH=")
		}
	}
	out, err := cmd.CombinedOutput()
	if err != nil || strings.TrimSpace(string(out)) != "sub/" {
		t.Fatalf("bash script completion = %q (%v), want sub/", out, err)
	}
}

func seedCompletionRoot(t *testing.T, root string) {
	t.Helper()
	iface := []byte(`{"application":"q:app","entrypoints":[],"format":"cozy.package.interface/1","jobs":[{"models":[],"name":"prepare","publishes":false,"request":{"fields":[]},"result":{"fields":[]},"weights_outputs":[]}]}`)
	parsed, problem := launch.DecodePackageInterface(iface)
	fatal(t, problem)
	if len(parsed.Raw) == 0 {
		t.Fatal("fixture lost its canonical interface")
	}
	dir := filepath.Join(root, "installs", "dryrun")
	must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(dir)), 0o700))
	must(t, os.WriteFile(launch.PackageInterfacePath(dir), iface, 0o600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	_, problem = store.Activate(records.PackageInstall{ID: "completion-install", Package: "proof/dryrun", Major: 1,
		Version: "1.0.0", SourceKind: "tensorhub", Dir: dir,
		Platform: "linux-x86"})
	fatal(t, problem)
	for _, rental := range []records.Rental{
		{ID: "rental-live", MachineName: "loran", State: "ready"},
		{ID: "rental-gone", MachineName: "lark", State: "released"},
	} {
		rental.AcceleratorModel, rental.AcceleratorCount, rental.HourlyRateUSDMicros, rental.Hub = "CPU", 1, 1, "https://hub.invalid"
		fatal(t, store.RecordRental(rental))
	}
}
