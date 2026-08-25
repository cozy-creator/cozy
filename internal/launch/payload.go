package launch

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

// The `run` payload grammar (cozy-runtime-cli.md), the same one cr-016 implements for a
// bare-venv run: a positional PRIMARY filling the first declared field, `key=value`
// scalars, `key=@file` file inputs, `key:=<json>` raw JSON for nested values, and
// `--in <file>` supplying the whole payload with `key=value` merged on top.
//
// TYPING IS THE DESCRIPTOR'S, not this module's. `steps=20` is an int because the field's
// rendered schema says int, never because the string looked numeric — the difference shows
// up the first time an endpoint declares a string field whose value is digits. The schema
// is the surface the release's own runtime vouched for at install, read back from the
// generation, so this costs microseconds and no subprocess: cozy-creator.md's
// "payload validation is client-side from the recorded schema, near-instant".
//
// An undeclared key refuses HERE, before a request is recorded and long before a model
// loads, because a typo should cost a millisecond.

// ParsePayload builds one request document from the argv terms.
func ParsePayload(ep *Entrypoint, terms []string, infile string) (json.RawMessage, *exit.Error) {
	document := map[string]json.RawMessage{}

	if infile != "" {
		data, err := os.ReadFile(infile)
		if err != nil {
			return nil, exit.New(exit.NotFound, "--in %s: %s", infile, err).
				WithRemedy("--in takes one JSON file holding the whole payload")
		}
		var loaded map[string]json.RawMessage
		if err := json.Unmarshal(data, &loaded); err != nil {
			return nil, exit.New(exit.Validation, "--in %s does not hold one JSON object: %s", infile, err)
		}
		for k, v := range loaded {
			document[k] = v
		}
	}

	var positional []string
	for _, term := range terms {
		colon := strings.Index(term, ":=")
		// `key:=json` only when the FIRST `=` is the one in `:=` — `a=b:=c` is a scalar.
		if colon >= 0 && strings.Index(term, "=") == colon+1 {
			key, raw := term[:colon], term[colon+2:]
			if !json.Valid([]byte(raw)) {
				return nil, exit.Usagef("%s:=… is not JSON: %s", key, raw).
					WithRemedy("`key:=<json>` carries a nested value verbatim; `key=value` is the scalar form")
			}
			if e := declared(ep, key); e != nil {
				return nil, e
			}
			document[key] = json.RawMessage(raw)
			continue
		}
		if key, raw, ok := strings.Cut(term, "="); ok {
			if e := declared(ep, key); e != nil {
				return nil, e
			}
			if after, isFile := strings.CutPrefix(raw, "@"); isFile {
				data, err := os.ReadFile(after)
				if err != nil {
					return nil, exit.New(exit.NotFound, "%s=@%s: %s", key, after, err)
				}
				encoded, err := json.Marshal(string(data))
				if err != nil {
					return nil, exit.Internalf("cannot carry %s: %s", after, err)
				}
				document[key] = encoded
				continue
			}
			value, e := typed(ep, key, raw)
			if e != nil {
				return nil, e
			}
			document[key] = value
			continue
		}
		positional = append(positional, term)
	}

	if len(positional) > 0 {
		primary := ""
		if len(ep.Request.Fields) > 0 {
			// The FIRST declared field, in the author's own declaration order (msgspec
			// preserves it), never a name this module blesses.
			primary = ep.Request.Fields[0].Name
		}
		if primary == "" {
			return nil, exit.Usagef("%s takes no positional value: it declares no request field", ep.Name)
		}
		if len(positional) > 1 {
			return nil, exit.Usagef("%s takes ONE positional value (%s); got %d",
				ep.Name, primary, len(positional)).
				WithRemedy("every other field is `key=value`, `key=@file` or `key:=<json>`")
		}
		value, e := typed(ep, primary, positional[0])
		if e != nil {
			return nil, e
		}
		document[primary] = value
	}

	data, err := json.Marshal(document)
	if err != nil {
		return nil, exit.Internalf("cannot render the payload: %s", err)
	}
	return data, nil
}

func declared(ep *Entrypoint, key string) *exit.Error {
	if _, ok := ep.TypeOfField(key); ok {
		return nil
	}
	return exit.New(exit.Validation, "%s declares no request field %q", ep.Name, key).
		WithRemedy("it declares: %s", strings.Join(ep.RequestFields(), ", ")).
		WithNext("cozy describe <org/endpoint>")
}

// typed spells one scalar the way the field's rendered schema declares it.
func typed(ep *Entrypoint, key, raw string) (json.RawMessage, *exit.Error) {
	rendered, ok := ep.TypeOfField(key)
	if !ok {
		return nil, declared(ep, key)
	}
	kind, _ := typeOf(rendered)
	switch kind {
	case "scalar:int":
		if _, err := strconv.ParseInt(raw, 10, 64); err != nil {
			return nil, wrongType(ep, key, raw, "int")
		}
		return json.RawMessage(raw), nil
	case "scalar:float":
		if _, err := strconv.ParseFloat(raw, 64); err != nil {
			return nil, wrongType(ep, key, raw, "float")
		}
		return json.RawMessage(raw), nil
	case "scalar:bool":
		switch raw {
		case "true", "false":
			return json.RawMessage(raw), nil
		}
		return nil, wrongType(ep, key, raw, "bool")
	case "scalar:str", "tree":
		// A TREE's wire value IS its ref (cr-009): the path rides the field VALUE at the
		// far end, hydrated from the grant, and a ref the grant does not cover never
		// reaches a filesystem. So the scalar spelling is the ref, and `--input
		// <ref>=<dir>` is what grants the read.
		encoded, err := json.Marshal(raw)
		if err != nil {
			return nil, exit.Internalf("cannot carry %s: %s", key, err)
		}
		return encoded, nil
	}
	// Anything structured — an asset, a list, a nested struct — has no scalar spelling.
	// `key:=<json>` is the form that carries it, and saying so beats guessing.
	return nil, exit.New(exit.Validation,
		"%s.%s is not a scalar and `key=value` cannot spell one", ep.Name, key).
		WithRemedy("carry it as `%s:=<json>`, or `%s=@<file>` for a file input", key, key)
}

func wrongType(ep *Entrypoint, key, raw, want string) *exit.Error {
	return exit.New(exit.Validation,
		"%s.%s is declared %s and %q is not one", ep.Name, key, want, raw).
		WithRemedy("the schema is the release's own — `cozy describe <org/endpoint>/%s` prints it", ep.Name)
}
