package records

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// device_memory_measurements is the measured working-memory ledger (proto-061 B):
// one row per succeeded attempt reporting working or total allocator memory.
// Working bytes are above the residency baseline; total bytes include that residency.
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
 total_peak_bytes INTEGER NOT NULL DEFAULT 0,
 request_digest TEXT NOT NULL DEFAULT '',
 exact_models_digest TEXT NOT NULL DEFAULT '',
 gpu_count INTEGER NOT NULL DEFAULT 0,
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
	// Total is eligible only after exact request, manifest, SKU and width matching.
	TotalBytes int64
	TotalRuns  int
}

// WorkingPeaks is one (package, release, entrypoint)'s measurements keyed by
// ModelsDigest, with separately prefixed exact model/SKU/width total observations.
type WorkingPeaks map[string]WorkingPeak

// For returns legacy working evidence plus any exactly matched total observation.
// A different lane manifest, adapter, SKU or width cannot lower the legacy estimate.
func (peaks WorkingPeaks) For(models []ModelRef, sku string, width int) WorkingPeak {
	peak := peaks[ModelsDigest(models)]
	if key := totalPeakKey(exactModelsDigest(models), sku, width); key != "" {
		total := peaks[key]
		peak.TotalBytes, peak.TotalRuns = total.TotalBytes, total.TotalRuns
	}
	return peak
}

func totalPeakKey(models, sku string, width int) string {
	if models == "" || sku == "" || sku == "local" || width <= 0 {
		return ""
	}
	return "total:" + models + ":" + sku + ":" + strconv.Itoa(width)
}

func measurementDigest(value any) string {
	raw, err := json.Marshal(value)
	if err != nil {
		return ""
	}
	digest, _ := canonical.Spell(canonical.Digest(raw))
	return digest
}

func exactMemoryDigest(digest string) bool {
	raw, err := canonical.Raw(digest)
	spelled, _ := canonical.Spell(raw)
	return err == nil && spelled == digest
}

// exactModelsDigest includes manifests and ordered adapter semantics. Catalog names
// alone are insufficient for private checkpoints. Sizing and ladder observations
// are excluded so pinning a candidate produces the same execution identity.
func exactModelsDigest(models []ModelRef) string {
	rows := make([]string, 0, len(models))
	for _, model := range models {
		if !exactMemoryDigest(model.Manifest) {
			return ""
		}
		adapters := append([]ModelAdapterRef(nil), model.Adapters...)
		for i := range adapters {
			if !exactMemoryDigest(adapters[i].Manifest) {
				return ""
			}
			adapters[i].Bytes = 0
		}
		slot := model.BindingPath
		if slot == "" {
			slot = model.Slot
		}
		rows = append(rows, measurementDigest(struct {
			Package, Slot, Model, Release, Lane, Manifest string
			Adapters                                      []ModelAdapterRef
		}{model.Package, slot, model.Model, model.Release, model.Lane, model.Manifest, adapters}))
	}
	slices.Sort(rows)
	return measurementDigest(rows)
}

// exactMemoryRequest names the workload independently of candidate model pinning.
// The sealed local delivery covers the complete transported package closure. A
// published request must already carry its immutable environment identity; a cold
// request without one retains the legacy estimate. Never replace either with a
// mutable install name, a package version, or a fingerprint that omits delivery.
func exactMemoryRequest(req Request) string {
	// Installation handles select a worker environment but are deliberately not
	// content or memoization identities. Without an operation-owned delivery
	// identity, retain the conservative legacy estimate.
	return ""
}

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
 AND working_peak_bytes > 0 GROUP BY models_digest`, pkg, release, entrypoint)
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
	// The store owns one connection: release the legacy query before reading totals.
	rows.Close()
	request := exactMemoryRequest(req)
	if request == "" {
		return out, nil
	}
	totals, err := s.db.Query(`SELECT exact_models_digest,sku,gpu_count,MAX(total_peak_bytes),COUNT(*)
 FROM device_memory_measurements WHERE package=? AND release=? AND entrypoint=?
 AND request_digest=? AND total_peak_bytes>0 AND total_peak_bytes>=working_peak_bytes
 GROUP BY exact_models_digest,sku,gpu_count`, pkg, release, entrypoint, request)
	if err != nil {
		return nil, exit.Internalf("cannot read measured total memory for %s: %s", pkg, err)
	}
	defer totals.Close()
	for totals.Next() {
		var models, sku string
		var width int
		var peak WorkingPeak
		if err := totals.Scan(&models, &sku, &width, &peak.TotalBytes, &peak.TotalRuns); err != nil {
			return nil, exit.Internalf("cannot read measured total memory for %s: %s", pkg, err)
		}
		if key := totalPeakKey(models, sku, width); key != "" {
			out[key] = peak
		}
	}
	if err := totals.Err(); err != nil {
		return nil, exit.Internalf("cannot read measured total memory for %s: %s", pkg, err)
	}
	return out, nil
}

// recordDeviceMemoryTx files a succeeded attempt's working-memory measurement in
// the terminal's own transaction, retaining the total allocator peak from that same
// attested outcome. Missing metrics never become measured zero memory.
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
	total := body.GetMetrics().GetPeakDeviceMemoryBytes()
	if peak == 0 && total == 0 {
		return nil
	}
	if peak > math.MaxInt64 || total > math.MaxInt64 {
		return exit.New(exit.Structural, "%s#%d reported impossible memory peaks %d/%d", t.RequestID, t.Attempt, peak, total)
	}
	req, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE id=?`, t.RequestID))
	if err != nil {
		return exit.Internalf("cannot read request %s for its measurement: %s", t.RequestID, err)
	}
	sku := "local"
	var width int
	if req.Worker != "" {
		err := tx.QueryRow(`SELECT sku,accelerator_count FROM rentals WHERE id=?`, req.Worker).Scan(&sku, &width)
		if errors.Is(err, sql.ErrNoRows) {
			sku = ""
		} else if err != nil {
			return exit.Internalf("cannot read rental %s for its measurement: %s", req.Worker, err)
		}
	}
	pkg, release, entrypoint := measurementSubject(req)
	if _, err := tx.Exec(`INSERT INTO device_memory_measurements(request_id,attempt,package,release,
 entrypoint,models_digest,shape_cell,sku,working_peak_bytes,measured_at,
 total_peak_bytes,request_digest,exact_models_digest,gpu_count) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
 ON CONFLICT(request_id,attempt) DO NOTHING`, t.RequestID, t.Attempt, pkg, release, entrypoint,
		ModelsDigest(req.Models), body.GetMetrics().GetShapeCell(), sku, int64(peak), now(),
		int64(total), exactMemoryRequest(req), exactModelsDigest(req.Models), width); err != nil {
		return exit.Internalf("cannot record the working memory of %s#%d: %s", t.RequestID, t.Attempt, err)
	}
	return nil
}
