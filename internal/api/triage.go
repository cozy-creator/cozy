package api

import (
	"encoding/json"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
)

// THE TRIAGE READ SURFACE (cr-011's seam, cl-006's half of it).
//
// cr-011 built the bundle and made it explainable from a directory; it deliberately did
// NOT build two things, and both are here: PERSISTENCE (a one-shot run deletes its worker
// root, so a bundle worth keeping is copied by the client — orchestrator/server.go's
// captureTriage) and RENDERING.
//
// The read path is a key and a digest, exactly as the runtime's own reader is:
//
//	records.AttemptByKey(attempt_key)  ->  the retained row
//	the row's triage_digest            ->  recomputed over the bytes on read
//
// The one deliberate DIVERGENCE from cr-011's `BundleStore.read`: it verifies against the
// RUNTIME'S JOURNALED RECEIPT, and this verifies against the TERMINAL DOCUMENT. The
// terminal is strictly stronger evidence here — this orchestrator recomputed its digest
// over the resident bytes, parsed it under unknown-field refusal, and committed it in the
// same transaction as the outputs. The worker's journal is a file the worker can still
// write; the accepted terminal is not.
//
// `explain` is a PROJECTION, never authority — the first lines a person reads, ported
// from cr-011's own `explain()` over the CLOSED section set that schema froze.

const bundleFormat = "cozy.runtime.WorkerTriageBundle/1"

func (s *Server) triage(w http.ResponseWriter, r *http.Request) {
	key := r.PathValue("attempt_key")
	row, e := s.store.AttemptByKey(key)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if row == nil {
		s.refuse(w, r, http.StatusNotFound, "not_found",
			"no attempt by that key on this host",
			"triage is addressed by the opaque attempt key; there is no path form and no request-id form")
		return
	}
	if row.TriageSubject == "" {
		s.refuse(w, r, http.StatusNotFound, "no_bundle",
			"attempt "+key+" named no triage bundle",
			"a bundle exists for a terminal attempt whose worker committed one")
		return
	}
	if row.TriagePath == "" {
		s.refuse(w, r, http.StatusGone, "bundle_not_kept",
			"attempt "+key+" named bundle "+row.TriageSubject+" and its bytes were not kept",
			"the orchestrator refused the bytes when the terminal was accepted; the terminal itself still stands")
		return
	}
	data, err := os.ReadFile(row.TriagePath)
	if err != nil {
		s.refuse(w, r, http.StatusGone, "bundle_absent",
			"the retained bundle for "+key+" is no longer readable", "")
		return
	}
	// VERIFIED ON READ, against what the terminal claimed. A bundle edited on disk is a
	// DETECTED CORRUPTION rather than a plausible story — which is the whole reason
	// this route recomputes instead of trusting the row it just read.
	spelled, _ := canonical.Spell(canonical.Digest(data))
	if int64(len(data)) != row.TriageLength || spelled != row.TriageDigest {
		s.refuse(w, r, http.StatusConflict, "bundle_corrupt",
			"the retained bundle does not hash to what its terminal declared",
			"a bundle whose bytes moved is not evidence; the terminal document is still authoritative")
		return
	}
	var document map[string]any
	if err := json.Unmarshal(data, &document); err != nil {
		s.refuse(w, r, http.StatusConflict, "bundle_malformed",
			"the retained bundle is not readable as a document", "")
		return
	}
	if document["format"] != bundleFormat {
		s.refuse(w, r, http.StatusConflict, "bundle_malformed",
			"the retained bundle is not a "+bundleFormat, "")
		return
	}
	s.ok(w, r, http.StatusOK, map[string]any{
		"attempt_key": key,
		"subject_id":  row.TriageSubject,
		"length":      row.TriageLength,
		"digest":      row.TriageDigest,
		"verified":    true,
		"explain":     explain(document),
		"bundle":      document,
	})
}

// explain is cr-011's projection, ported. It reads the first lines a person needs and
// nothing more; the whole document rides beside it for a client that wants everything.
func explain(document map[string]any) []string {
	attempt := object(document["attempt"])
	terminal := object(document["terminal"])
	measurements := object(document["measurements"])
	lines := []string{
		"attempt   " + str(attempt["request_id"]) + "#" + str(attempt["attempt"]) +
			" (" + str(document["subject_id"]) + ")",
		"spec      " + clip(str(attempt["exec_spec_digest"]), 30),
		// `outcome`, and the ADMISSION GENERATION beside the incarnation: the same two
		// lines cr-011's own renderer prints (cozy-runtime `triage.py::explain`).
		// `readiness_epoch` is DELETED with rev-2, not renamed, and this projection is
		// deliberately the SAME VIEW as the runtime's — a second reader that drifts is two
		// answers to one question.
		"outcome   " + str(terminal["status"]) + " / " + str(terminal["cause_code"]) +
			" from " + str(terminal["cause_origin"]),
		"because   " + clip(str(terminal["safe_message"]), 300),
		"executor  incarnation " + str(attempt["executor_incarnation"]) +
			" admission generation " + str(attempt["admission_generation"]),
	}
	for _, f := range list(document["faults"]) {
		row := object(f)
		lines = append(lines, "fault     "+str(row["reason"])+": "+clip(str(row["detail"]), 200))
	}
	for _, c := range list(document["confessions"]) {
		row := object(c)
		lines = append(lines, "confessed "+str(row["name"])+": "+str(row["value"]))
	}
	for _, l := range list(document["liveness"]) {
		row := object(l)
		if str(row["verdict"]) == "" {
			continue
		}
		lines = append(lines, "liveness  "+str(row["subject"])+" ["+str(row["step"])+"] "+
			str(row["verdict"])+" at position "+str(row["position"])+
			" after "+str(row["samples"])+" observations")
	}
	attribution := object(measurements["attribution"])
	stages := object(attribution["stages"])
	for _, name := range sorted(stages) {
		track := object(stages[name])
		lines = append(lines, "stage     "+name+": "+str(track["total_ms"])+
			" ms over "+str(track["count"]))
	}
	steps := object(attribution["steps"])
	for _, name := range sorted(steps) {
		track := object(steps[name])
		lines = append(lines, "steps     "+name+": "+str(track["count"])+
			" x mean "+str(track["mean_ms"])+" ms (min "+str(track["min_ms"])+
			", max "+str(track["max_ms"])+")")
	}
	caps := object(document["caps"])
	lines = append(lines, "tail      "+itoa(len(list(document["events"])))+" kept, "+
		str(caps["dropped"])+" shed by the ring, "+str(caps["shed_to_fit"])+
		" shed to fit the "+str(caps["max_bundle_bytes"])+" B cap, "+
		str(caps["refused"])+" refused at the emit boundary")
	return lines
}

func object(v any) map[string]any {
	if m, ok := v.(map[string]any); ok {
		return m
	}
	return map[string]any{}
}

func list(v any) []any {
	if l, ok := v.([]any); ok {
		return l
	}
	return nil
}

// str renders a JSON scalar the way the bundle's own reader would. A float that is a
// whole number prints without its ".0" — the bundle's numbers are integers, and JSON's
// float decoding is the only reason they would not look like it.
func str(v any) string {
	switch t := v.(type) {
	case nil:
		return "0"
	case string:
		return t
	case bool:
		if t {
			return "true"
		}
		return "false"
	case float64:
		if t == float64(int64(t)) {
			return itoa(int(t))
		}
		return strings.TrimRight(strings.TrimRight(strconv.FormatFloat(t, 'f', 3, 64), "0"), ".")
	}
	return ""
}

func sorted(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func clip(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
