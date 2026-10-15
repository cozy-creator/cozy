package records

import (
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

func scanChildBinding(row interface{ Scan(...any) error }) (ChildBinding, error) {
	var binding ChildBinding
	err := row.Scan(&binding.ParentInstallID, &binding.Module, &binding.Export, &binding.ChildInstallID, &binding.Entrypoint)
	return binding, err
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
