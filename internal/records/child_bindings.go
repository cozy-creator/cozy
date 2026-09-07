package records

import (
	"database/sql"
	"errors"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

const childRequestIndex = `CREATE UNIQUE INDEX IF NOT EXISTS requests_parent_call ON requests(parent_request_id,parent_call_index) WHERE parent_request_id!=''`

const childBindingsDDL = `
CREATE TABLE IF NOT EXISTS private_child_bindings (
 parent_install_id TEXT NOT NULL REFERENCES installs(id) ON DELETE CASCADE,
 interface_digest TEXT NOT NULL,
 module TEXT NOT NULL,
 export TEXT NOT NULL,
 child_install_id TEXT NOT NULL REFERENCES installs(id),
 local_revision_digest TEXT NOT NULL,
 entrypoint TEXT NOT NULL,
 PRIMARY KEY(parent_install_id,interface_digest,module,export)
)`

// ChildBinding is a frozen dependency fact belonging to a parent install. It is
// written by intake, never supplied by executing package code or looked up by a
// mutable package pin. The child install stays owned while this binding exists.
type ChildBinding struct {
	ParentInstallID     string
	InterfaceDigest     string
	Module              string
	Export              string
	ChildInstallID      string
	LocalRevisionDigest string
	Entrypoint          string
}

func (s *Store) HasChildBindings(parentInstall string) (bool, *exit.Error) {
	var held bool
	if err := s.db.QueryRow(`SELECT EXISTS(SELECT 1 FROM private_child_bindings WHERE parent_install_id=?)`, parentInstall).Scan(&held); err != nil {
		return false, exit.Internalf("cannot read parent execution role: %s", err)
	}
	return held, nil
}

const childBindingCols = `parent_install_id,interface_digest,module,export,child_install_id,local_revision_digest,entrypoint`

func scanChildBinding(row interface{ Scan(...any) error }) (ChildBinding, error) {
	var binding ChildBinding
	err := row.Scan(&binding.ParentInstallID, &binding.InterfaceDigest, &binding.Module, &binding.Export, &binding.ChildInstallID, &binding.LocalRevisionDigest, &binding.Entrypoint)
	return binding, err
}

func (s *Store) ChildBinding(parentInstall, interfaceDigest, module, export string) (*ChildBinding, *exit.Error) {
	binding, err := scanChildBinding(s.db.QueryRow(`SELECT `+childBindingCols+` FROM private_child_bindings WHERE parent_install_id=? AND interface_digest=? AND module=? AND export=?`, parentInstall, interfaceDigest, module, export))
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
		_, interfaceError := canonical.Raw(binding.InterfaceDigest)
		_, revisionError := canonical.Raw(binding.LocalRevisionDigest)
		if binding.ParentInstallID == "" || binding.ChildInstallID == "" || binding.Module == "" || binding.Export == "" || binding.Entrypoint == "" || interfaceError != nil || revisionError != nil {
			return exit.New(exit.Validation, "child binding requires exact parent, interface, export, child revision and entrypoint")
		}
		if _, err := tx.Exec(`INSERT INTO private_child_bindings(`+childBindingCols+`) VALUES(?,?,?,?,?,?,?) ON CONFLICT DO NOTHING`, binding.ParentInstallID, binding.InterfaceDigest, binding.Module, binding.Export, binding.ChildInstallID, binding.LocalRevisionDigest, binding.Entrypoint); err != nil {
			return exit.Internalf("cannot capture child dependency: %s", err)
		}
		held, err := scanChildBinding(tx.QueryRow(`SELECT `+childBindingCols+` FROM private_child_bindings WHERE parent_install_id=? AND interface_digest=? AND module=? AND export=?`, binding.ParentInstallID, binding.InterfaceDigest, binding.Module, binding.Export))
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
