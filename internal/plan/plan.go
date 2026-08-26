// Package plan owns the ENTRYPOINT BINDING PLAN RECORD: what its identity is, what is
// merely how this machine resolved it, and the one digest both ends compute.
//
// #506a — THE IDENTITY IS PATH-FREE. The record used to be digested whole, and it embeds
// three machine-local absolute paths: `project` (where this host staged the endpoint
// tree), `store` (this host's CAS root) and `config` (a file inside it). So a pod that
// installed the BYTE-IDENTICAL release archive computed a DIFFERENT plan id for the same
// plan, and every remote-serving lane was blocked by arithmetic rather than by design.
//
// The record is therefore partitioned, declared here and nowhere else:
//
//   - IDENTITY — content and reference identities only: digests, release ids, entrypoint
//     names, declared budgets. These are the digested bytes, and `ID` REFUSES any absolute
//     path that appears among them. The law is enforced, not intended: a future field that
//     smuggles a path in fails at the moment the id is minted, on the machine that minted
//     it, instead of silently making two hosts disagree.
//   - RESOLUTION — where the identity's referents actually landed on THIS disk. It travels
//     in the staged record because the worker needs it to open the files, and it is outside
//     the identity because it is a fact about a machine, not about a plan.
//
// The consequence that had to become true: two machines installing the same release
// compute the same id. `cozy-live rent` proves it against two independent roots.
package plan

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

// Format is the identity document's format tag. It moved from `/1` to `/2` with the
// partition: `/1` digested the whole record including its paths, so the two versions
// name genuinely different documents and an id minted under one never collides with the
// other's. Pre-launch there is nothing to migrate — a `/1` id exists only inside a
// running worker's staged directory, which dies with the worker.
const Format = "cozy.local.EntrypointBindingRecord/2"

// IDKey is the record's own id field. A document never contains its own digest, so it is
// excluded from the identity by name.
const IDKey = "entrypoint_binding_plan_id"

// ResolutionKeys are the record fields that say WHERE, on the machine that staged the
// record, the identity's referents are. They travel with the record and are NOT digested.
//
// Each one is a path today and could not be anything else: the runtime opens `config` as
// a file, hands `store` to TensorFS as a root, and imports the endpoint tree from
// `project`. What #506a changes is not that they exist but that they stopped being
// identity — the plan is the same plan wherever those three point.
var ResolutionKeys = []string{"project", "store", "config"}

var resolution = func() map[string]bool {
	m := map[string]bool{}
	for _, k := range ResolutionKeys {
		m[k] = true
	}
	return m
}()

// Identity is the subset of a record that its id is taken over: everything that is not
// resolution and is not the id itself.
func Identity(record map[string]any) map[string]any {
	out := make(map[string]any, len(record))
	for k, v := range record {
		if k == IDKey || resolution[k] {
			continue
		}
		out[k] = v
	}
	return out
}

// Resolution is the complement: this machine's answer to where the referents are.
func Resolution(record map[string]any) map[string]any {
	out := map[string]any{}
	for k, v := range record {
		if resolution[k] {
			out[k] = v
		}
	}
	return out
}

// absolutePath matches the two spellings of a rooted path this product runs on: a POSIX
// `/…` and a Windows `C:\…` / `C:/…`. It is deliberately a SHAPE test rather than a
// filesystem one — the interesting case is a path that exists nowhere near this process,
// which no stat can distinguish from a typo.
var absolutePath = regexp.MustCompile(`^(/|[A-Za-z]:[\\/]|\\\\)`)

// ID mints one record's plan id: sha256 over the canonical bytes of its IDENTITY.
//
// It refuses a rooted path anywhere in the identity. That refusal is the whole of #506a
// expressed as a law rather than a convention: whoever adds a field to the record has to
// decide, at the moment they add it, whether it names a plan or names a machine.
func ID(record map[string]any) (string, *exit.Error) {
	doc := map[string]canonical.Value{}
	for k, v := range Identity(record) {
		value, e := spell(k, v)
		if e != nil {
			return "", e
		}
		if e := pathFree(k, value); e != nil {
			return "", e
		}
		doc[k] = value
	}
	doc["format"] = Format
	data, err := canonical.Write(doc)
	if err != nil {
		return "", exit.Internalf("cannot canonicalize the binding record's identity: %s", err)
	}
	spelled, err := canonical.Spell(canonical.Digest(data))
	if err != nil {
		return "", exit.Internalf("cannot spell the binding plan id: %s", err)
	}
	return spelled, nil
}

// pathFree is the law. It walks the whole value, because a path hidden one level down in
// a list or a map is exactly as machine-local as one at the top.
func pathFree(key string, v canonical.Value) *exit.Error {
	switch t := v.(type) {
	case string:
		if absolutePath.MatchString(t) {
			return exit.Named(exit.Structural, "plan_identity_path",
				"the binding record's identity field %q is the absolute path %q, and a plan id "+
					"digested over a machine-local path cannot be the same on two machines "+
					"(#506a)", key, t).
				WithRemedy("name it by CONTENT or by REFERENCE (a digest, a release id, a " +
					"store-relative ref) and put the path in the record's RESOLUTION half — " +
					"internal/plan.ResolutionKeys is the declared list")
		}
	case []canonical.Value:
		for _, item := range t {
			if e := pathFree(key, item); e != nil {
				return e
			}
		}
	case map[string]canonical.Value:
		for name, item := range t {
			if e := pathFree(key+"."+name, item); e != nil {
				return e
			}
		}
	}
	return nil
}

// spell converts one Go record value into the canonical writer's value space. These
// documents are INTEGER-ONLY, exactly as the writer is: an integral float is the integer
// it spells (a record that crossed JSON arrives that way) and a fractional one refuses
// rather than being rounded into a different identity.
func spell(key string, v any) (canonical.Value, *exit.Error) {
	switch t := v.(type) {
	case string:
		return t, nil
	case bool:
		return t, nil
	case int:
		return int64(t), nil
	case int64:
		return t, nil
	case float64:
		if t != float64(int64(t)) {
			return nil, exit.Internalf(
				"binding record field %q is %v; these documents are integer-only", key, t)
		}
		return int64(t), nil
	case []string:
		items := make([]canonical.Value, 0, len(t))
		for _, item := range t {
			items = append(items, item)
		}
		return items, nil
	case []any:
		items := make([]canonical.Value, 0, len(t))
		for i, item := range t {
			value, e := spell(key+"["+itoa(i)+"]", item)
			if e != nil {
				return nil, e
			}
			items = append(items, value)
		}
		return items, nil
	case map[string]string:
		pairs := map[string]canonical.Value{}
		for name, value := range t {
			pairs[name] = value
		}
		return pairs, nil
	case map[string]any:
		pairs := map[string]canonical.Value{}
		for name, value := range t {
			spelled, e := spell(key+"."+name, value)
			if e != nil {
				return nil, e
			}
			pairs[name] = spelled
		}
		return pairs, nil
	default:
		return nil, exit.Internalf("binding record field %q has no canonical spelling (%T)", key, v)
	}
}

// FileName is the record's file name under a `binding-plans` directory: the id's hex,
// which is what the runtime resolves a wire plan id against.
func FileName(id string) string { return strings.TrimPrefix(id, "sha256:") + ".json" }

// Render produces the STAGED document: identity plus resolution plus the record's own id.
// It is what lands on disk at both ends, and it is byte-identical for one record however
// the map was ordered, because encoding/json sorts map keys.
func Render(record map[string]any, id string) ([]byte, *exit.Error) {
	staged := make(map[string]any, len(record)+1)
	for k, v := range record {
		staged[k] = v
	}
	staged[IDKey] = id
	data, err := json.MarshalIndent(staged, "", "  ")
	if err != nil {
		return nil, exit.Internalf("cannot render the binding record: %s", err)
	}
	return data, nil
}

// Stage writes one record into a `binding-plans` directory and answers its id. It is the
// LOCAL half of delivery; the remote half (internal/media) ships exactly these bytes to
// the pod's media server, which re-derives the id before it will keep them.
func Stage(dir string, record map[string]any) (string, *exit.Error) {
	id, e := ID(record)
	if e != nil {
		return "", e
	}
	data, e := Render(record, id)
	if e != nil {
		return "", e
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", exit.Internalf("cannot create the binding-plan directory %s: %s", dir, err)
	}
	if err := os.WriteFile(filepath.Join(dir, FileName(id)), data, 0o644); err != nil {
		return "", exit.Internalf("cannot stage the binding record: %s", err)
	}
	return id, nil
}

// Verify re-derives a staged record's id from its own bytes and checks it against the
// name it was delivered under. It is what makes the media server's plan route fail-closed:
// a record that does not hash to its own name is not the plan the owner dispatched
// against, whoever sent it, and the pod refuses it rather than staging a plan the
// coordinator will then name in a directive.
//
// This check exists only because the identity became path-free: under `/1` the pod could
// not have recomputed the id at all, because the digest covered the owner's own paths.
func Verify(data []byte, claimed string) (map[string]any, *exit.Error) {
	var record map[string]any
	if err := json.Unmarshal(data, &record); err != nil {
		return nil, exit.New(exit.Validation,
			"the binding record is not a JSON document: %s", err)
	}
	if stated, _ := record[IDKey].(string); stated != "" && stated != claimed {
		return nil, exit.Named(exit.Validation, "plan_id_disagreement",
			"the record names itself %s and was delivered as %s", short(stated), short(claimed))
	}
	id, e := ID(record)
	if e != nil {
		return nil, e
	}
	if id != claimed {
		return nil, exit.Named(exit.Validation, "plan_id_mismatch",
			"this record's identity hashes to %s and it was delivered as %s: a binding plan "+
				"is named by the digest of its own identity, so a record that does not hash "+
				"to its name is not the plan the owner dispatched against",
			short(id), short(claimed)).
			WithRemedy("identity fields: %s", strings.Join(sortedKeys(Identity(record)), ", "))
	}
	return record, nil
}

func sortedKeys(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func short(id string) string {
	bare := strings.TrimPrefix(id, "sha256:")
	if len(bare) > 12 {
		return "sha256:" + bare[:12] + "…"
	}
	return id
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	digits := ""
	for n > 0 {
		digits = string(rune('0'+n%10)) + digits
		n /= 10
	}
	return digits
}
