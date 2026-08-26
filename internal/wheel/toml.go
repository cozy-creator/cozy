package wheel

import (
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

// A deliberately SMALL TOML reader: strings and string arrays, in the three tables the
// packer acts on. It exists instead of a dependency because the packer's whole claim is
// that project metadata is inert data on a fixed path — a general TOML library would be
// a larger surface reading the same four keys.
//
// Scope is the safety property: a table the packer does not want is SKIPPED WHOLE, so an
// endpoint's `[bindings."Fl2VAModel"]` or a repo's `[tool.mypy]` never has to be
// expressible here. Inside a wanted table the reader is strict — a line it cannot read is
// a typed refusal, never a silently dropped field.

type table struct {
	strs  map[string]string
	lists map[string][]string
}

func (t table) str(key string) string { return t.strs[key] }

func (t table) list(key string) []string { return t.lists[key] }

// parseTOML returns the wanted tables. `want` holds full dotted table names.
func parseTOML(file string, body []byte, want map[string]bool) (map[string]table, *exit.Error) {
	out := map[string]table{}
	lines := strings.Split(strings.ReplaceAll(string(body), "\r\n", "\n"), "\n")
	current := ""

	for i := 0; i < len(lines); i++ {
		raw := strings.TrimSpace(stripComment(lines[i]))
		if raw == "" {
			continue
		}
		if strings.HasPrefix(raw, "[") {
			name, ok := tableName(raw)
			if !ok {
				// A header this reader cannot NAME (a quoted key, an array of tables) is
				// still unmistakably a header: it ends the previous table, and the one it
				// opens is skipped whole. Strictness applies to the keys of a table the
				// packer acts on, never to grammar it deliberately does not read.
				current = "\x00unread"
				continue
			}
			current = name
			if want[name] {
				if _, dup := out[name]; dup {
					return nil, malformed("%s line %d: `[%s]` is declared twice", file, i+1, name)
				}
				out[name] = table{strs: map[string]string{}, lists: map[string][]string{}}
			}
			continue
		}
		if !want[current] {
			continue
		}

		key, rest, ok := strings.Cut(raw, "=")
		if !ok {
			return nil, malformed("%s line %d: %q is not `key = value`", file, i+1, raw)
		}
		key = strings.TrimSpace(key)
		if !bareKey(key) {
			return nil, malformed("%s line %d: %q is not a bare key", file, i+1, key)
		}
		val := strings.TrimSpace(rest)

		if strings.HasPrefix(val, "[") {
			// An array may span lines; gather until the brackets balance.
			for depth := bracketDepth(val); depth > 0; depth = bracketDepth(val) {
				i++
				if i >= len(lines) {
					return nil, malformed("%s: `%s` opens an array that never closes", file, key)
				}
				val += " " + strings.TrimSpace(stripComment(lines[i]))
			}
			items, e := parseArray(file, key, val)
			if e != nil {
				return nil, e
			}
			out[current].lists[key] = items
			continue
		}
		s, ok := parseString(val)
		if !ok {
			return nil, malformed("%s line %d: `%s` is neither a string nor a string array; "+
				"this packer reads no other TOML value", file, i+1, key)
		}
		out[current].strs[key] = s
	}
	return out, nil
}

// stripComment removes a `#` comment, respecting quotes so a `#` inside a string stays.
func stripComment(line string) string {
	var quote byte
	for i := 0; i < len(line); i++ {
		c := line[i]
		switch {
		case quote != 0:
			if c == quote {
				quote = 0
			} else if c == '\\' && quote == '"' {
				i++
			}
		case c == '"' || c == '\'':
			quote = c
		case c == '#':
			return line[:i]
		}
	}
	return line
}

func tableName(raw string) (string, bool) {
	if !strings.HasSuffix(raw, "]") || strings.HasPrefix(raw, "[[") {
		return "", false
	}
	name := strings.TrimSpace(raw[1 : len(raw)-1])
	if name == "" {
		return "", false
	}
	for _, part := range strings.Split(name, ".") {
		if !bareKey(strings.TrimSpace(part)) {
			return "", false
		}
	}
	return name, true
}

func bareKey(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		ok := c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '-' || c == '_'
		if !ok {
			return false
		}
	}
	return true
}

func bracketDepth(s string) int {
	depth, quote := 0, byte(0)
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case quote != 0:
			if c == quote {
				quote = 0
			} else if c == '\\' && quote == '"' {
				i++
			}
		case c == '"' || c == '\'':
			quote = c
		case c == '[':
			depth++
		case c == ']':
			depth--
		}
	}
	return depth
}

func parseArray(file, key, val string) ([]string, *exit.Error) {
	inner := strings.TrimSpace(val)
	if !strings.HasPrefix(inner, "[") || !strings.HasSuffix(inner, "]") {
		return nil, malformed("%s: `%s` is not a closed array", file, key)
	}
	inner = strings.TrimSpace(inner[1 : len(inner)-1])
	if inner == "" {
		return []string{}, nil
	}
	var out []string
	for _, item := range splitTop(inner) {
		item = strings.TrimSpace(item)
		if item == "" {
			continue // a trailing comma
		}
		s, ok := parseString(item)
		if !ok {
			return nil, malformed("%s: `%s` holds %q, which is not a string", file, key, item)
		}
		out = append(out, s)
	}
	return out, nil
}

func splitTop(s string) []string {
	var out []string
	start, quote := 0, byte(0)
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case quote != 0:
			if c == quote {
				quote = 0
			} else if c == '\\' && quote == '"' {
				i++
			}
		case c == '"' || c == '\'':
			quote = c
		case c == ',':
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	return append(out, s[start:])
}

// parseString reads a basic or literal TOML string. Multi-line forms are not read: a
// metadata value the packer puts in METADATA is one line by construction.
func parseString(val string) (string, bool) {
	if len(val) < 2 {
		return "", false
	}
	if val[0] == '\'' && val[len(val)-1] == '\'' && !strings.HasPrefix(val, "'''") {
		return val[1 : len(val)-1], true
	}
	if val[0] != '"' || val[len(val)-1] != '"' || strings.HasPrefix(val, `"""`) {
		return "", false
	}
	var b strings.Builder
	body := val[1 : len(val)-1]
	for i := 0; i < len(body); i++ {
		if body[i] != '\\' {
			b.WriteByte(body[i])
			continue
		}
		i++
		if i >= len(body) {
			return "", false
		}
		switch body[i] {
		case '"', '\\', '/':
			b.WriteByte(body[i])
		case 'n':
			b.WriteByte('\n')
		case 't':
			b.WriteByte('\t')
		default:
			return "", false
		}
	}
	return b.String(), true
}
