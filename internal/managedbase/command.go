package managedbase

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

type Command struct {
	Bin  string
	Args []string
}

// RuntimeCommand constructs the exact digest-pinned Runtime command. The
// container runs as the host user, sees only Cozy home, and forwards only the
// closed values its Runtime invocation needs. Nothing resolves cozy-runtime on
// the host PATH.
func RuntimeCommand(root, profile, manifestDigest, workdir string, worker bool) (Command, *exit.Error) {
	record, problem := Open(root, profile, manifestDigest)
	if problem != nil {
		return Command{}, problem
	}
	docker, err := exec.LookPath("docker")
	if err != nil {
		return Command{}, exit.Named(exit.Structural, "managed_base.docker_missing",
			"the managed local Runtime base cannot launch because Docker is absent")
	}
	rel, err := filepath.Rel(filepath.Dir(root), workdir)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return Command{}, exit.Named(exit.Structural, "managed_base.workdir_outside_home",
			"the managed local Runtime work directory %s is outside Cozy home", workdir)
	}
	if strings.Contains(root, ":") {
		return Command{}, exit.Named(exit.Structural, "managed_base.home_unmountable",
			"Cozy home %s cannot be expressed as one Docker bind mount", root)
	}
	args := []string{"run", "--rm", "--init", "--pull=never", "--network=none", "--ipc=host",
		"--user", strconv.Itoa(os.Getuid()) + ":" + strconv.Itoa(os.Getgid()),
		"--volume", root + ":" + root, "--workdir", workdir,
		"--env", "COZY_HOME"}
	groups, _ := os.Getgroups()
	for _, group := range groups {
		if group != os.Getgid() {
			args = append(args, "--group-add", strconv.Itoa(group))
		}
	}
	if worker {
		args = append(args, "--gpus", "all", "--env", "CUDA_VISIBLE_DEVICES",
			"--env", "COZY_BOOTSTRAP_CREDENTIAL")
	}
	args = append(args, "--entrypoint", "/usr/bin/python", record.Identity.Image,
		"-m", "cozy_runtime.cli.main")
	return Command{Bin: docker, Args: args}, nil
}
