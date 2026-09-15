package records

import (
	"database/sql"
	"errors"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

const capturePinsDDL = `
CREATE TABLE IF NOT EXISTS capture_pins (
 caller TEXT PRIMARY KEY,
 input_digest TEXT NOT NULL,
 install_id TEXT NOT NULL REFERENCES installs(id),
 revision_digest TEXT NOT NULL
)`

// CapturePin retains one completed immutable installation per logical caller.
// Caller groups replacements; only InputDigest establishes reusable identity.
type CapturePin struct {
	Caller, InputDigest, InstallID, RevisionDigest string
}

func (s *Store) CapturePin(caller string) (*CapturePin, *exit.Error) {
	var pin CapturePin
	err := s.db.QueryRow(`SELECT caller,input_digest,install_id,revision_digest FROM capture_pins WHERE caller=?`, caller).
		Scan(&pin.Caller, &pin.InputDigest, &pin.InstallID, &pin.RevisionDigest)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read completed local capture: %s", err)
	}
	return &pin, nil
}

// ReplaceCapturePin publishes only a completed capture. The install writer lock
// covers byte validation and this transaction; existing request/binding owners
// continue to retain a superseded installation.
func (s *Store) ReplaceCapturePin(pin CapturePin) (string, *exit.Error) {
	for _, value := range []string{pin.Caller, pin.InputDigest, pin.RevisionDigest} {
		if _, err := canonical.Raw(value); err != nil {
			return "", exit.New(exit.Validation, "completed capture requires exact caller, input and revision identities")
		}
	}
	if pin.InstallID == "" {
		return "", exit.New(exit.Validation, "completed capture requires an install")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return "", exit.Internalf("cannot begin completed capture replacement: %s", err)
	}
	defer tx.Rollback()
	var prior string
	if err := tx.QueryRow(`SELECT install_id FROM capture_pins WHERE caller=?`, pin.Caller).Scan(&prior); err != nil && !errors.Is(err, sql.ErrNoRows) {
		return "", exit.Internalf("cannot read prior completed capture: %s", err)
	}
	if _, err := tx.Exec(`INSERT INTO capture_pins(caller,input_digest,install_id,revision_digest) VALUES(?,?,?,?)
		ON CONFLICT(caller) DO UPDATE SET input_digest=excluded.input_digest,install_id=excluded.install_id,revision_digest=excluded.revision_digest`,
		pin.Caller, pin.InputDigest, pin.InstallID, pin.RevisionDigest); err != nil {
		return "", exit.Internalf("cannot retain completed capture: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return "", exit.Internalf("cannot commit completed capture: %s", err)
	}
	return prior, nil
}

// ForgetCapturePin is the explicit reinstall boundary. Unpinning a capture does
// not remove its bytes while requests or another ordinary owner still use them.
func (s *Store) ForgetCapturePin(caller string) *exit.Error {
	if _, err := s.db.Exec(`DELETE FROM capture_pins WHERE caller=?`, caller); err != nil {
		return exit.Internalf("cannot release completed capture: %s", err)
	}
	return nil
}
