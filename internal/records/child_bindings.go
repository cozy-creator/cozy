package records

import (
	"database/sql"
	"errors"

	"github.com/cozy-creator/cozy/internal/exit"
)

const childRequestIndex = `CREATE UNIQUE INDEX IF NOT EXISTS requests_parent_call ON requests(parent_request_id,parent_call_index) WHERE parent_request_id!=''`

const childBindingsDDL = `
CREATE TABLE IF NOT EXISTS private_child_bindings (
 parent_install_id TEXT NOT NULL REFERENCES installs(id) ON DELETE CASCADE,
 module TEXT NOT NULL,
 export TEXT NOT NULL,
 child_install_id TEXT NOT NULL REFERENCES installs(id),
 entrypoint TEXT NOT NULL,
 PRIMARY KEY(parent_install_id,module,export)
)`

// ChildBinding is a frozen dependency fact belonging to a parent install. It is
// written by intake, never supplied by executing package code or looked up by a
// mutable package pin. The child install stays owned while this binding exists.
type ChildBinding struct {
	ParentInstallID string
	Module          string
	Export          string
	ChildInstallID  string
	Entrypoint      string
}

func (s *Store) HasChildBindings(parentInstall string) (bool, *exit.Error) {
	var held bool
	if err := s.db.QueryRow(`SELECT EXISTS(SELECT 1 FROM private_child_bindings WHERE parent_install_id=?)`, parentInstall).Scan(&held); err != nil {
		return false, exit.Internalf("cannot read parent execution role: %s", err)
	}
	return held, nil
}

func (s *Store) ChildBindings(parentInstall string) ([]ChildBinding, *exit.Error) {
	rows, err := s.db.Query(`SELECT `+childBindingCols+` FROM private_child_bindings WHERE parent_install_id=? ORDER BY module,export`, parentInstall)
	if err != nil {
		return nil, exit.Internalf("cannot read frozen child dependencies: %s", err)
	}
	defer rows.Close()
	var out []ChildBinding
	for rows.Next() {
		binding, err := scanChildBinding(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read frozen child dependency: %s", err)
		}
		out = append(out, binding)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish frozen child dependency census: %s", err)
	}
	return out, nil
}

const childBindingCols = `parent_install_id,module,export,child_install_id,entrypoint`

// SelfCallableEntrypoints names the install's own exports it captured as its own
// children. Those entrypoints are the CALLEES of a composition, never its parent, so
// the CPU orchestration role and its resource fence must not be read onto them.
func (s *Store) SelfCallableEntrypoints(installID string) (map[string]bool, *exit.Error) {
	rows, err := s.db.Query(`SELECT entrypoint FROM private_child_bindings WHERE parent_install_id=? AND child_install_id=?`, installID, installID)
	if err != nil {
		return nil, exit.Internalf("cannot read self-callable exports: %s", err)
	}
	defer rows.Close()
	out := map[string]bool{}
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			return nil, exit.Internalf("cannot read a self-callable export: %s", err)
		}
		out[name] = true
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish the self-callable export census: %s", err)
	}
	return out, nil
}

// CompositionParent identifies an entrypoint with captured callees that is not a
// captured callee itself. This is availability, not an exclusive execution role:
// only a request without model, device, or weights requirements can take the
// spare CPU orchestration slot. Ordinary library consumers keep their resources.
func (s *Store) CompositionParent(installID, entrypoint string) (bool, *exit.Error) {
	bound, problem := s.HasChildBindings(installID)
	if problem != nil || !bound {
		return false, problem
	}
	callees, problem := s.SelfCallableEntrypoints(installID)
	if problem != nil {
		return false, problem
	}
	return !callees[entrypoint], nil
}

func scanChildBinding(row interface{ Scan(...any) error }) (ChildBinding, error) {
	var binding ChildBinding
	err := row.Scan(&binding.ParentInstallID, &binding.Module, &binding.Export, &binding.ChildInstallID, &binding.Entrypoint)
	return binding, err
}

func (s *Store) ChildBinding(parentInstall, module, export string) (*ChildBinding, *exit.Error) {
	binding, err := scanChildBinding(s.db.QueryRow(`SELECT `+childBindingCols+` FROM private_child_bindings WHERE parent_install_id=? AND module=? AND export=?`, parentInstall, module, export))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read frozen child binding: %s", err)
	}
	return &binding, nil
}

func (s *Store) RecordChildBindings(bindings []ChildBinding) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin child dependency capture: %s", err)
	}
	defer tx.Rollback()
	for _, binding := range bindings {
		if binding.ParentInstallID == "" || binding.ChildInstallID == "" || binding.Module == "" || binding.Export == "" || binding.Entrypoint == "" {
			return exit.New(exit.Validation, "child binding requires parent installation, export, child installation and entrypoint")
		}
		if _, err := tx.Exec(`INSERT INTO private_child_bindings(`+childBindingCols+`) VALUES(?,?,?,?,?) ON CONFLICT DO NOTHING`, binding.ParentInstallID, binding.Module, binding.Export, binding.ChildInstallID, binding.Entrypoint); err != nil {
			return exit.Internalf("cannot capture child dependency: %s", err)
		}
		held, err := scanChildBinding(tx.QueryRow(`SELECT `+childBindingCols+` FROM private_child_bindings WHERE parent_install_id=? AND module=? AND export=?`, binding.ParentInstallID, binding.Module, binding.Export))
		if err != nil {
			return exit.Internalf("cannot verify captured child binding: %s", err)
		}
		if held != binding {
			return exit.Named(exit.Conflict, "child.binding_changed", "immutable parent dependency already names another child implementation")
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit child dependency capture: %s", err)
	}
	return nil
}
