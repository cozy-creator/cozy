package records

import (
	"database/sql"
	"errors"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

// The account's credit as each Hub last stated it (th-242): run and rental commands warn
// from it without asking the Hub again. No row: the account never touches credit there,
// or nothing was read yet.
const hubCreditDDL = `CREATE TABLE IF NOT EXISTS hub_credit (
 hub TEXT PRIMARY KEY,
 available_usd_micros INTEGER NOT NULL,
 warning_usd_micros INTEGER NOT NULL,
 floor_usd_micros INTEGER NOT NULL,
 observed_ms INTEGER NOT NULL
)`

// HubCredit is one Hub's last statement of the account's credit.
type HubCredit struct {
	Available, Warning, Floor int64
	Observed                  time.Time
}

// StateHubCredit records what hub stated; nil forgets it.
func (s *Store) StateHubCredit(hub string, credit *HubCredit) *exit.Error {
	// An observation the Hub states again at every listing: it never waits on the disk, which
	// the daemon would otherwise sync every few seconds for as long as it holds a rental.
	err := s.unsynced(func(tx *sql.Tx) error {
		if credit == nil {
			_, err := tx.Exec(`DELETE FROM hub_credit WHERE hub=?`, hub)
			return err
		}
		_, err := tx.Exec(`INSERT INTO hub_credit(hub,available_usd_micros,warning_usd_micros,floor_usd_micros,observed_ms)
 VALUES(?,?,?,?,?) ON CONFLICT(hub) DO UPDATE SET available_usd_micros=excluded.available_usd_micros,
 warning_usd_micros=excluded.warning_usd_micros,floor_usd_micros=excluded.floor_usd_micros,observed_ms=excluded.observed_ms`,
			hub, credit.Available, credit.Warning, credit.Floor, credit.Observed.UnixMilli())
		return err
	})
	if err != nil {
		return exit.Internalf("cannot record the Hub's credit: %s", err)
	}
	return nil
}

// HubCredit is what hub last stated, or nil.
func (s *Store) HubCredit(hub string) (*HubCredit, *exit.Error) {
	var c HubCredit
	var observed int64
	err := s.db.QueryRow(`SELECT available_usd_micros,warning_usd_micros,floor_usd_micros,observed_ms
 FROM hub_credit WHERE hub=?`, hub).Scan(&c.Available, &c.Warning, &c.Floor, &observed)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read the Hub's credit: %s", err)
	}
	c.Observed = time.UnixMilli(observed)
	return &c, nil
}
