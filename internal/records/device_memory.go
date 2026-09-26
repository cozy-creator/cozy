package records

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"math"
	"slices"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// device_memory_measurements is the measured working-memory ledger (proto-061 B):
// one row per succeeded attempt whose outcome reported working_peak_device_bytes,
// the device memory the call used above what was resident when it entered the device.
// Rows outlive their request; nothing references them.
const deviceMemoryMeasurementsDDL = `CREATE TABLE IF NOT EXISTS device_memory_measurements (
 request_id TEXT NOT NULL,
 attempt INTEGER NOT NULL,
 package TEXT NOT NULL,
 release TEXT NOT NULL,
 entrypoint TEXT NOT NULL,
 models_digest TEXT NOT NULL,
 shape_cell TEXT NOT NULL DEFAULT '',
 sku TEXT NOT NULL DEFAULT '',
 working_peak_bytes INTEGER NOT NULL,
 measured_at TEXT NOT NULL,
 PRIMARY KEY(request_id, attempt)
)`

const deviceMemoryMeasurementsIndex = `CREATE INDEX IF NOT EXISTS device_memory_measurements_subject
 ON device_memory_measurements(package, release, entrypoint, models_digest)`

// WorkingPeak is the largest measured working memory for one subject and how many
// runs measured it. Runs == 0 means unmeasured, never "needs nothing".
type WorkingPeak struct {
	Bytes int64
	Runs  int
}

// WorkingPeaks is one (package, release, entrypoint)'s measurements keyed by
// ModelsDigest.
type WorkingPeaks map[string]WorkingPeak

// ModelsDigest names a pinned model selection: every slot's model, release and
// lane, and its adapters, independent of order. Sizing facts (bytes, ladders) are
// excluded, so the recorded and the sized selection agree whenever they execute
// the same weights.
func ModelsDigest(models []ModelRef) string {
	rows := make([]string, 0, len(models))
	for _, model := range models {
		slot := model.BindingPath
		if slot == "" {
			slot = model.Slot
		}
		row := []string{slot, model.Model, model.Release, model.Lane}
		for _, adapter := range model.Adapters {
			row = append(row, adapter.Component, adapter.Model, adapter.Release, adapter.Lane, adapter.Scale)
		}
		rows = append(rows, strings.Join(row, "\x00"))
	}
	slices.Sort(rows)
	sum := sha256.Sum256([]byte(strings.Join(rows, "\x01")))
	return "sha256:" + hex.EncodeToString(sum[:])
}

// measurementSubject is the (package, release, entrypoint) a request's runs are
// measured under. A local package has no release; its sealed digest stands in.
func measurementSubject(req Request) (string, string, string) {
	release := req.Release
	if release == "" {
		release = req.LocalInstallationID
	}
	return req.Package, release, req.Entrypoint
}

// WorkingPeaks reads every measurement for the request's package release and
// entrypoint, keyed by models digest.
func (s *Store) WorkingPeaks(req Request) (WorkingPeaks, *exit.Error) {
	pkg, release, entrypoint := measurementSubject(req)
	rows, err := s.db.Query(`SELECT models_digest, MAX(working_peak_bytes), COUNT(*)
 FROM device_memory_measurements WHERE package=? AND release=? AND entrypoint=?
 GROUP BY models_digest`, pkg, release, entrypoint)
	if err != nil {
		return nil, exit.Internalf("cannot read measured working memory for %s: %s", pkg, err)
	}
	defer rows.Close()
	out := WorkingPeaks{}
	for rows.Next() {
		var digest string
		var peak WorkingPeak
		if err := rows.Scan(&digest, &peak.Bytes, &peak.Runs); err != nil {
			return nil, exit.Internalf("cannot read measured working memory for %s: %s", pkg, err)
		}
		out[digest] = peak
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot read measured working memory for %s: %s", pkg, err)
	}
	return out, nil
}

// recordDeviceMemoryTx files a succeeded attempt's working-memory measurement in
// the terminal's own transaction. An outcome without field 14 (wire < 61, or a
// CPU run) records nothing.
func recordDeviceMemoryTx(tx *sql.Tx, t Terminal) *exit.Error {
	if t.Status != "SUCCEEDED" || len(t.Body) == 0 {
		return nil
	}
	// The orchestrator verified the worker's body before this transaction; a body
	// that is not AttemptOutcomeBody/1 (a synthesized close) carries no measurement.
	var body pb.AttemptOutcomeBody
	if canonical.Unmarshal(t.Body, &body) != nil {
		return nil
	}
	peak := body.GetMetrics().GetWorkingPeakDeviceBytes()
	if peak == 0 {
		return nil
	}
	if peak > math.MaxInt64 {
		return exit.New(exit.Structural, "%s#%d reported an impossible working peak %d", t.RequestID, t.Attempt, peak)
	}
	req, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE id=?`, t.RequestID))
	if err != nil {
		return exit.Internalf("cannot read request %s for its measurement: %s", t.RequestID, err)
	}
	sku := "local"
	if req.Worker != "" {
		err := tx.QueryRow(`SELECT sku FROM rentals WHERE id=?`, req.Worker).Scan(&sku)
		if errors.Is(err, sql.ErrNoRows) {
			sku = ""
		} else if err != nil {
			return exit.Internalf("cannot read rental %s for its measurement: %s", req.Worker, err)
		}
	}
	pkg, release, entrypoint := measurementSubject(req)
	if _, err := tx.Exec(`INSERT INTO device_memory_measurements(request_id,attempt,package,release,
 entrypoint,models_digest,shape_cell,sku,working_peak_bytes,measured_at) VALUES(?,?,?,?,?,?,?,?,?,?)
 ON CONFLICT(request_id,attempt) DO NOTHING`, t.RequestID, t.Attempt, pkg, release, entrypoint,
		ModelsDigest(req.Models), body.GetMetrics().GetShapeCell(), sku, int64(peak), now()); err != nil {
		return exit.Internalf("cannot record the working memory of %s#%d: %s", t.RequestID, t.Attempt, err)
	}
	return nil
}
