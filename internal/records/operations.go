package records

import (
	"cmp"
	"database/sql"
	"encoding/json"
	"errors"
	"path/filepath"
	"slices"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Operation is a command's work on one machine that runs no function: a package installed or
// removed, a model downloaded or uploaded, a function kept warm. It is a run like any other:
// one row of the requests outbox under its own run number (`cozy run list/show/watch/cancel`),
// sent to its machine as a warm or remove run. Its payload is this record.
type Operation struct {
	Package string     `json:"package,omitempty"`
	Release string     `json:"release,omitempty"`
	Models  []ModelRef `json:"models,omitempty"`
	// Destination is where the one model is put: the org/name it is uploaded to (`cozy model
	// upload`), or the local/name the machine holds it under (`cozy model download … local/`).
	Destination string `json:"destination,omitempty"`
	// Write is a local file this computer writes to the machine first: the file an upload
	// makes (its model's object:// source).
	Write string `json:"write,omitempty"`
	// Hub is the Tensorhub the operation is read at.
	Hub string `json:"hub,omitempty"`
	// Warm makes the package's Entrypoint a member of the machine's warm set, kept ready up to
	// this level (WarmLevels); "off" takes it out. Models are then its `model.<param>=` choices.
	Entrypoint string `json:"entrypoint,omitempty"`
	Warm       string `json:"warm,omitempty"`
	// Local is this computer's installation of local/<name> code, written to the machine
	// and installed there (`cozy package install ./dir --rental`).
	Local string `json:"local,omitempty"`
	// Remove takes the machine's installation of that name away (`cozy package remove
	// --rental`); Package and Release say what it was.
	Remove string `json:"remove,omitempty"`
	// Software is the Runtime/TensorFS pair a rental's software update installs (`cozy rental
	// update`, a new boot taking its Hub's target), and nothing else is named with it.
	Software *SoftwareUpdate `json:"software,omitempty"`
}

// SoftwareUpdate names published versions or local wheels, or Target: the pair the rental's
// Hub names when the update is sent (unpinned, never older than the machine runs).
type SoftwareUpdate struct {
	RuntimeVersion  string `json:"runtime_version,omitempty"`
	TensorFSVersion string `json:"tensorfs_version,omitempty"`
	RuntimeWheel    *Wheel `json:"runtime_wheel,omitempty"`
	TensorFSWheel   *Wheel `json:"tensorfs_wheel,omitempty"`
	Target          bool   `json:"target,omitempty"`
}

// Wheel is a local wheel as it was measured when the update was asked for; it is written to
// the machine as measured, or the machine refuses it.
type Wheel struct {
	Path   string `json:"path"`
	Digest string `json:"digest"`
	Length int64  `json:"length"`
}

// OperationKinds are the request kinds an Operation runs as.
var OperationKinds = []string{"install", "remove", "download", "upload", "warm", "update"}

// WarmLevels are how far a machine keeps a warm set member ready, each including the ones
// before it.
var WarmLevels = []string{"installed", "downloaded", "imported", "host", "gpu"}

// Kind is the run kind `cozy run list` shows for the operation.
func (o Operation) Kind() string {
	switch {
	case o.Software != nil:
		return "update"
	case o.Remove != "":
		return "remove"
	case o.Warm != "":
		return "warm"
	case o.Destination != "" && !strings.HasPrefix(o.Destination, "local/"):
		return "upload"
	case o.Package != "":
		return "install"
	}
	return "download"
}

// HoldsLocally is an operation only a machine that keeps local models takes: a file it is
// written, a local/ alias it reads, or a local/ alias it is put under.
func (o Operation) HoldsLocally() bool {
	local := o.Write != "" || strings.HasPrefix(o.Destination, "local/")
	for _, model := range o.Models {
		local = local || strings.HasPrefix(model.Model, "local/")
	}
	return local
}

// Target names what an operation puts on its machine.
func (o Operation) Target() string {
	if software := o.Software; software != nil {
		if software.Target {
			return "the Hub's target software"
		}
		var named []string
		for _, member := range []struct {
			name, version string
			wheel         *Wheel
		}{{"runtime", software.RuntimeVersion, software.RuntimeWheel}, {"tensorfs", software.TensorFSVersion, software.TensorFSWheel}} {
			if member.wheel != nil {
				named = append(named, filepath.Base(member.wheel.Path))
			} else if member.version != "" {
				named = append(named, member.name+" "+member.version)
			}
		}
		return strings.Join(named, ", ")
	}
	if o.Warm != "" {
		return o.Package + "/" + o.Entrypoint + " warm=" + o.Warm
	}
	if o.Package != "" && o.Release == "" {
		return o.Package // the hub's newest, until the machine names it
	}
	if o.Package != "" {
		return o.Package + "@" + o.Release
	}
	models := make([]string, 0, len(o.Models))
	for _, model := range o.Models {
		if o.Destination != "" {
			models = append(models, cmp.Or(model.Source, model.Model, model.Manifest)+" -> "+o.Destination)
			continue
		}
		name := model.Model
		if model.Release != "" {
			name += "@" + model.Release
		}
		if model.Lane != "" {
			name += "/" + model.Lane
		}
		if model.Release == "" {
			name += "#" + model.Manifest
		}
		models = append(models, name)
	}
	return strings.Join(models, ", ")
}

// IsOperation is a request that runs no function (Operation).
func (r Request) IsOperation() bool { return slices.Contains(OperationKinds, r.Kind) }

// Operation is the operation a request of an operation kind carries.
func (r Request) Operation() (Operation, *exit.Error) {
	var op Operation
	if err := json.Unmarshal(r.Payload, &op); err != nil {
		return op, exit.Internalf("run %s holds no readable operation: %s", r.ID, err)
	}
	return op, nil
}

// PendingOperation is the unsettled run of the same operation on the same machine, if any: a
// repeated command follows it instead of starting another.
func (s *Store) PendingOperation(machine, digest string) (*Request, *exit.Error) {
	var id string
	err := s.db.QueryRow(`SELECT r.id FROM requests r JOIN machine_executions e ON e.request_id=r.id
		WHERE r.body_digest=? AND (e.machine_id=? OR r.requested_rental=?)
		AND r.state NOT IN ('succeeded','failed','canceled','refused','abandoned')
		ORDER BY r.created_at LIMIT 1`, digest, machine, machine).Scan(&id)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read pending operations: %s", err)
	}
	return s.RequestByReference(id)
}

// PendingUpdate is the unsettled software update run on a machine, if any: the machine's other
// work waits for it, so it lands on the new software.
func (s *Store) PendingUpdate(machine string) (*Request, *exit.Error) {
	var id string
	err := s.db.QueryRow(`SELECT r.id FROM requests r JOIN machine_executions e ON e.request_id=r.id
		WHERE r.kind='update' AND (e.machine_id=? OR r.requested_rental=?)
		AND r.state NOT IN ('succeeded','failed','canceled','refused','abandoned')
		ORDER BY r.created_at LIMIT 1`, machine, machine).Scan(&id)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read pending software updates: %s", err)
	}
	return s.RequestByReference(id)
}
