package install

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/flock"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

// Lease holds one install against reclamation for the life of a submission: from the
// moment a caller resolves the install until the daemon has recorded the request that
// references it. It is a shared flock on the install's lease file, so the kernel ends it
// with its holder, SIGKILL included, and nothing about it is timed.
type Lease struct{ file *os.File }

func leasePath(l home.Layout, id string) string { return filepath.Join(l.Installs, id+".lease") }

// LeaseInstall takes the lease and then confirms the install still exists: a reclaim that
// won before the lease was taken answers not found.
func LeaseInstall(l home.Layout, st *records.Store, id string) (*Lease, *exit.Error) {
	if _, problem := installRemovalTarget(l, id); problem != nil {
		return nil, problem
	}
	file, err := os.OpenFile(leasePath(l, id), os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, exit.Internalf("cannot open the lease on install %s: %s", id, err)
	}
	if err := flock.BlockShared(file); err != nil {
		file.Close()
		return nil, exit.Internalf("cannot lease install %s: %s", id, err)
	}
	inst, problem := st.Install(id)
	if problem != nil {
		file.Close()
		return nil, problem
	}
	if inst == nil {
		file.Close()
		return nil, exit.New(exit.NotFound, "install %s is no longer present", id)
	}
	return &Lease{file: file}, nil
}

// Release ends the lease. A nil lease is a no-op, so callers may defer it unconditionally.
func (lease *Lease) Release() {
	if lease != nil && lease.file != nil {
		_ = lease.file.Close()
		lease.file = nil
	}
}

// claimForReclaim takes the install's lease exclusively without waiting. It answers nil
// while any submission holds the install; the reclaim then leaves the install for a
// later pass, exactly as it leaves one a request references.
func claimForReclaim(l home.Layout, id string) (*os.File, *exit.Error) {
	file, err := os.OpenFile(leasePath(l, id), os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, exit.Internalf("cannot open the lease on install %s: %s", id, err)
	}
	if flock.Exclusive(file) != nil {
		file.Close()
		return nil, nil
	}
	return file, nil
}

// sweepLease removes the lease file of an install that no longer exists, once nobody holds
// it. A lease on a live install is the install's own and stays.
func sweepLease(l home.Layout, st *records.Store, name string) {
	id, ok := strings.CutSuffix(name, ".lease")
	if !ok {
		return
	}
	if inst, problem := st.Install(id); problem != nil || inst != nil {
		return
	}
	lease, problem := claimForReclaim(l, id)
	if problem != nil || lease == nil {
		return
	}
	defer lease.Close()
	_ = os.Remove(leasePath(l, id))
}
