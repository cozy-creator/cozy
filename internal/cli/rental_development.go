package cli

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"io"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
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
		return nil, exit.Usagef("development rentals require --ssh-public-key FILE or rentals.ssh_public_key in config.yaml")
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
	if existing != nil && (pinned == nil || pinned.SSHPublicKey != selected.SSHPublicKey) {
		return nil, exit.Named(exit.Conflict, "rental.idempotency_conflict", "rental operation already declares different development access").WithRemedy("resume without development flags or use a new operation key")
	}
	return selected, nil
}

func handleRentalSSHInfo(ctx *Context) *exit.Error {
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
