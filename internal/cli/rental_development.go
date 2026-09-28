package cli

import (
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/flock"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

func rentalDevelopment(ctx *Context, existing *records.RentalOperation) (*hub.RentalDevelopment, *exit.Error) {
	var pinned *hub.RentalDevelopment
	if existing != nil {
		req, problem := hub.ParseRentalRequestBytes(existing.RequestBody)
		if problem != nil {
			return nil, problem
		}
		pinned = req.Development
	}
	flagValue, flagSet := ctx.Inv.Bools["--development"]
	explicit := flagSet || ctx.Inv.Value("--ssh-public-key") != ""
	// An acquisition is immutable. A changed default or deleted public-key file
	// cannot rewrite its mode, prevent reconciliation, or spend for another pod.
	if existing != nil && !explicit {
		return pinned, nil
	}
	enabled := ctx.Cfg.RentalsDevelopment
	if flagSet {
		enabled = flagValue
	}
	path := ctx.Inv.Value("--ssh-public-key")
	if path == "" && enabled {
		path = ctx.Cfg.RentalsSSHPublicKey
		if path != "" && !filepath.IsAbs(path) && !strings.HasPrefix(path, "~/") {
			path = filepath.Join(ctx.Cfg.Home, path)
		}
	}
	if !enabled && path == "" {
		if pinned != nil {
			return nil, exit.Named(exit.Conflict, "rental.idempotency_conflict", "rental operation already declares different development access").WithRemedy("resume without development flags or use a new operation key")
		}
		return nil, nil
	}
	if !enabled {
		return nil, exit.Usagef("--ssh-public-key requires --development")
	}
	if path == "" {
		if pinned != nil {
			return pinned, nil
		}
		var problem *exit.Error
		path, problem = managedRentalSSHKey(ctx)
		if problem != nil {
			return nil, problem
		}
	}
	if strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, exit.Usagef("cannot resolve the SSH public-key home directory")
		}
		path = filepath.Join(home, strings.TrimPrefix(path, "~/"))
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, exit.Usagef("cannot read the SSH public-key file")
	}
	defer file.Close()
	raw, err := io.ReadAll(io.LimitReader(file, 8193))
	if err != nil || len(raw) > 8192 {
		return nil, exit.Usagef("SSH public-key file exceeds 8192 bytes or is unreadable")
	}
	key := strings.TrimSpace(string(raw))
	fields := strings.Fields(key)
	if len(fields) < 2 || strings.ContainsAny(key, "\r\n\x00") || !(strings.HasPrefix(fields[0], "ssh-") || strings.HasPrefix(fields[0], "ecdsa-") || strings.HasPrefix(fields[0], "sk-")) {
		return nil, exit.Usagef("supply one public SSH key line, not private key material or authorized-key options")
	}
	selected := &hub.RentalDevelopment{SSHPublicKey: key}
	if existing != nil && (pinned == nil || *pinned != *selected) {
		return nil, exit.Named(exit.Conflict, "rental.idempotency_conflict", "rental operation already declares different development access").WithRemedy("resume without development flags or use a new operation key")
	}
	return selected, nil
}

// managedRentalSSHKey keeps the operator's maintenance identity in the existing
// protected credential directory. Explicit operator keys still take precedence.
func managedRentalSSHKey(ctx *Context) (string, *exit.Error) {
	directory := filepath.Join(ctx.Cfg.Home, "auth")
	if err := os.MkdirAll(directory, 0700); err != nil {
		return "", exit.Internalf("cannot create private rental credentials: %s", err)
	}
	lock, err := os.OpenFile(filepath.Join(directory, "rental-ssh.lock"), os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return "", exit.Internalf("cannot lock private rental credentials: %s", err)
	}
	defer lock.Close()
	if err := flock.Exclusive(lock); err != nil {
		return "", exit.Unavailablef("another rental command is creating maintenance credentials; retry shortly")
	}
	path := filepath.Join(directory, "rental-ssh")
	if info, err := os.Lstat(path); err == nil {
		if !info.Mode().IsRegular() || info.Mode().Perm()&0077 != 0 {
			return "", exit.New(exit.Credential, "rental SSH private key must be a private regular file")
		}
		if _, err := os.Stat(path + ".pub"); err != nil {
			return "", exit.New(exit.Credential, "rental SSH identity is incomplete; restore its public key")
		}
		return path + ".pub", nil
	} else if !os.IsNotExist(err) {
		return "", exit.New(exit.Credential, "rental SSH private key cannot be inspected")
	}
	if _, err := os.Lstat(path + ".pub"); !os.IsNotExist(err) {
		return "", exit.New(exit.Credential, "rental SSH public key already exists without its private key; restore that identity")
	}
	command := exec.Command("ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "cozy-private-rental", "-f", path)
	command.Env = ctx.Cfg.Tool()
	if err := command.Run(); err != nil {
		return "", exit.New(exit.Credential, "cannot create a private rental SSH key").WithRemedy("install OpenSSH client tools, or select --ssh-public-key FILE")
	}
	return path + ".pub", nil
}

func handleRentalSSHInfo(ctx *Context) *exit.Error {
	adoptRentalHub(ctx, ctx.Inv.Args[0])
	_, store, problem := rentalStores(ctx)
	if problem != nil {
		return problem
	}
	defer store.Close()
	row, problem := store.RentalByMachine(strings.TrimSpace(ctx.Inv.Args[0]))
	if problem != nil {
		return problem
	}
	if row == nil {
		return exit.New(exit.Conflict, "development rental is not attached on this Creator home")
	}
	c := client(ctx)
	if row.Hub != c.Base() {
		return exit.New(exit.Conflict, "rental belongs to another configured Hub")
	}
	call, cancel := hub.Context()
	defer cancel()
	remote, problem := c.Rental(call, row.ID)
	if problem != nil {
		return problem
	}
	if !remote.Development || !remote.Ready() || remote.WorkerID != row.ExpectedWorkerID || remote.WorkerBootID != row.ExpectedWorkerBootID {
		return exit.New(exit.Conflict, "development rental is not ready on its pinned worker boot")
	}
	host, port, err := net.SplitHostPort(remote.SSHAddress)
	number, parseErr := strconv.Atoi(port)
	if err != nil || net.ParseIP(host) == nil || parseErr != nil || number < 1 || number > 65535 {
		return exit.New(exit.Unavailable, "Hub has not supplied a valid mapped SSH endpoint")
	}
	return emit(ctx, compactRecord([]output.Field{{K: "rental", V: row.ID}, {K: "machine", V: row.MachineName}, {K: "ssh_address", V: remote.SSHAddress}, {K: "host", V: host}, {K: "port", V: number}, {K: "user", V: "root"}, {K: "worker_boot_id", V: remote.WorkerBootID}}, "machine", "ssh_address"))
}

// rentalImage is the registered worker image `--image` names in place of the
// machine's default. A recorded acquisition keeps the image it was asked with.
func rentalImage(ctx *Context, existing *records.RentalOperation) (string, *exit.Error) {
	image := strings.TrimSpace(ctx.Inv.Value("--image"))
	if existing == nil {
		return image, nil
	}
	req, problem := hub.ParseRentalRequestBytes(existing.RequestBody)
	if problem != nil {
		return "", problem
	}
	if image != "" && image != req.Image {
		return "", exit.Named(exit.Conflict, "rental.idempotency_conflict",
			"rental operation already names worker image %q", req.Image).
			WithRemedy("resume without --image or use a new operation key")
	}
	return req.Image, nil
}
