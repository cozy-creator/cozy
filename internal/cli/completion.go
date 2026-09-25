package cli

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/alecthomas/kong"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// completeVerb is the shell-protocol entrance the scripts below call on every <TAB>:
//
//	cozy __complete <bash|zsh|fish> <line-before-cursor> [<bash COMP_WORDBREAKS>]
//
// Candidates are derived from the Kong model, so the grammar is the only command list.
// A value's `predictor` tag (or Kong's path type) chooses its value completion. It prints
// candidates only: no errors, no daemon, no network, no writes.
const completeVerb = "__complete"

// CompletionCmd prints the script that wires a shell's <TAB> to completeVerb.
type CompletionCmd struct {
	Shell string `arg:"" name:"shell" enum:"bash,zsh,fish" help:"bash (~/.bashrc: source <(cozy completion bash)), zsh (~/.zshrc, after compinit: source <(cozy completion zsh)) or fish (config.fish: cozy completion fish | source)."`
}

func (c *CompletionCmd) Run(r *Runtime) error {
	_, err := io.WriteString(r.Out, completionScripts[c.Shell])
	return err
}

var completionScripts = map[string]string{
	"bash": `# cozy bash completion: source <(cozy completion bash)
_cozy() {
	local IFS=$'\n'
	COMPREPLY=($(command cozy ` + completeVerb + ` bash "${COMP_LINE:0:COMP_POINT}" "$COMP_WORDBREAKS" 2>/dev/null))
	if [[ ${#COMPREPLY[@]} -eq 1 && ${COMPREPLY[0]} == *[/=] ]]; then compopt -o nospace; fi
}
complete -F _cozy cozy
`,
	"zsh": `# cozy zsh completion: source <(cozy completion zsh), after compinit
_cozy() {
	local -a spaced bare
	local c
	for c in "${(@f)$(command cozy ` + completeVerb + ` zsh "$LBUFFER" 2>/dev/null)}"; do
		[[ -z $c ]] && continue
		if [[ $c == *[/=] ]]; then bare+=("$c"); else spaced+=("$c"); fi
	done
	(( $#spaced )) && compadd -U -Q -- "${spaced[@]}"
	(( $#bare )) && compadd -U -Q -S '' -- "${bare[@]}"
}
compdef _cozy cozy
`,
	"fish": `# cozy fish completion: cozy completion fish | source
complete -c cozy -f -a '(command cozy ` + completeVerb + ` fish (commandline -cp) 2>/dev/null)'
`,
}

type candidate struct{ value, help string }

// complete answers one completeVerb call. It always exits 0: a shell shows nothing
// rather than an error when there is nothing to offer.
func complete(app *kong.Application, args []string, out io.Writer) int {
	if len(args) < 2 {
		return 0
	}
	shell, line := args[0], args[1]
	words, current, raw := splitLine(line)
	if len(words) == 0 {
		return 0
	}
	c := completer{}
	for _, cand := range c.candidates(app.Node, words[1:], current) {
		switch shell {
		case "bash":
			fmt.Fprintln(out, bashWord(shellEscape(cand.value), raw, args[2:]))
		case "zsh":
			fmt.Fprintln(out, shellEscape(cand.value))
		case "fish":
			if help := firstLine(cand.help); help != "" {
				fmt.Fprintf(out, "%s\t%s\n", cand.value, help)
			} else {
				fmt.Fprintln(out, cand.value)
			}
		}
	}
	return 0
}

// completer walks the typed words through the Kong model, then predicts the current one.
type completer struct {
	inventory *completionInventory
}

func (c *completer) candidates(root *kong.Node, words []string, current string) []candidate {
	node, positional, literal := root, 0, false
	var pending *kong.Flag
	if len(words) > 0 && words[0] == "help" {
		return commands(descend(root, words[1:]), current)
	}
	for _, word := range words {
		switch {
		case pending != nil:
			pending = nil
		case !literal && word == "--":
			literal = true
		case !literal && strings.HasPrefix(word, "-") && word != "-":
			name, _, inline := strings.Cut(word, "=")
			if flag := findFlag(node, positional, name); flag != nil && !inline && !flag.IsBool() && !flag.IsCounter() {
				pending = flag
			}
		default:
			if positional == 0 && !literal {
				if child := findChild(node, word); child != nil {
					node = child
					continue
				}
			}
			if len(node.Positional) == 0 && node.DefaultCmd != nil {
				node = node.DefaultCmd
			}
			positional++
		}
	}
	if pending != nil {
		return c.predict(pending.Value, current)
	}
	if !literal && strings.HasPrefix(current, "-") {
		if name, value, inline := strings.Cut(current, "="); inline {
			flag := findFlag(node, positional, name)
			if flag == nil || flag.IsBool() || flag.IsCounter() {
				return nil
			}
			return prefixed(name+"=", c.predict(flag.Value, value))
		}
		return flags(node, positional, current)
	}
	var out []candidate
	if positional == 0 && !literal {
		out = commands(node, current)
	}
	target := node
	if len(node.Positional) == 0 && node.DefaultCmd != nil && positional == 0 {
		target = node.DefaultCmd
	}
	if arg := positionalAt(target, positional); arg != nil {
		out = append(out, c.predict(arg, current)...)
	}
	return out
}

func descend(node *kong.Node, words []string) *kong.Node {
	for _, word := range words {
		child := findChild(node, word)
		if child == nil {
			break
		}
		node = child
	}
	return node
}

func findChild(node *kong.Node, name string) *kong.Node {
	for _, child := range node.Children {
		if child.Type != kong.CommandNode {
			continue
		}
		if child.Name == name {
			return child
		}
		for _, alias := range child.Aliases {
			if alias == name {
				return child
			}
		}
	}
	return nil
}

// flagScope is every flag Kong accepts at node: its own and its ancestors'. Before the
// first positional, a command with a default subcommand also accepts that default's flags.
func flagScope(node *kong.Node, positional int) []*kong.Flag {
	var scope []*kong.Flag
	if positional == 0 && node.DefaultCmd != nil {
		scope = append(scope, node.DefaultCmd.Flags...)
	}
	for n := node; n != nil; n = n.Parent {
		scope = append(scope, n.Flags...)
	}
	return scope
}

func findFlag(node *kong.Node, positional int, name string) *kong.Flag {
	for _, flag := range flagScope(node, positional) {
		if "--"+flag.Name == name || (flag.Short != 0 && "-"+string(flag.Short) == name) {
			return flag
		}
		for _, alias := range flag.Aliases {
			if "--"+alias == name {
				return flag
			}
		}
	}
	return nil
}

func flags(node *kong.Node, positional int, current string) []candidate {
	var out []candidate
	seen := map[string]bool{}
	for _, flag := range flagScope(node, positional) {
		name := "--" + flag.Name
		if flag.Hidden || seen[name] || !strings.HasPrefix(name, current) {
			continue
		}
		seen[name] = true
		out = append(out, candidate{name, flag.Help})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].value < out[j].value })
	return out
}

func commands(node *kong.Node, current string) []candidate {
	var out []candidate
	for _, child := range node.Children {
		if child.Type == kong.CommandNode && !child.Hidden && strings.HasPrefix(child.Name, current) {
			out = append(out, candidate{child.Name, child.Help})
		}
	}
	return out
}

func positionalAt(node *kong.Node, index int) *kong.Positional {
	if index < len(node.Positional) {
		return node.Positional[index]
	}
	if n := len(node.Positional); n > 0 && node.Positional[n-1].IsCumulative() {
		return node.Positional[n-1]
	}
	return nil
}

// predict completes one value. Predictors are named by the grammar's `predictor` tag:
//
//	file, dir            paths
//	binding-file/-dir    a path, or the path after `name=`
//	file-or-ref/dir-or-ref  a path once the value looks like one
//	callable             installed org/package/function, or a script path
//	package, rental      installed packages, current rentals
func (c *completer) predict(value *kong.Value, current string) []candidate {
	predictor := value.Tag.Get("predictor")
	if predictor == "" {
		switch value.Tag.Type {
		case "path", "existingfile":
			predictor = "file"
		case "existingdir":
			predictor = "dir"
		}
	}
	switch predictor {
	case "file", "dir":
		return paths(current, predictor == "dir")
	case "binding-file", "binding-dir":
		if name, path, ok := strings.Cut(current, "="); ok {
			return prefixed(name+"=", paths(path, predictor == "binding-dir"))
		}
		return paths(current, predictor == "binding-dir")
	case "file-or-ref", "dir-or-ref":
		if pathLike(current) {
			return paths(current, predictor == "dir-or-ref")
		}
		return nil
	case "callable":
		if pathLike(current) {
			return paths(current, false)
		}
		return matching(c.local().callables(), current)
	case "package":
		return matching(c.local().packageNames(), current)
	case "rental":
		return matching(c.local().rentalNames(), current)
	}
	if value.Enum != "" {
		return matching(value.EnumSlice(), current)
	}
	return nil
}

func pathLike(value string) bool {
	return strings.HasPrefix(value, ".") || strings.HasPrefix(value, "/") || strings.HasPrefix(value, "~")
}

// paths lists directory entries that extend current. Directories end in "/" so the
// shell keeps completing inside them; dotfiles appear once the typed name starts with ".".
func paths(current string, dirsOnly bool) []candidate {
	dir, base := "", current
	if i := strings.LastIndex(current, "/"); i >= 0 {
		dir, base = current[:i+1], current[i+1:]
	}
	lookup := dir
	if lookup == "" {
		lookup = "."
	} else if lookup == "~/" || strings.HasPrefix(lookup, "~/") {
		userHome, err := os.UserHomeDir()
		if err != nil {
			return nil
		}
		lookup = filepath.Join(userHome, lookup[2:])
	}
	entries, err := os.ReadDir(lookup)
	if err != nil {
		return nil
	}
	var out []candidate
	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasPrefix(name, base) || (strings.HasPrefix(name, ".") && !strings.HasPrefix(base, ".")) {
			continue
		}
		isDir := entry.IsDir()
		if entry.Type()&os.ModeSymlink != 0 {
			if info, err := os.Stat(filepath.Join(lookup, name)); err == nil {
				isDir = info.IsDir()
			}
		}
		switch {
		case isDir:
			out = append(out, candidate{value: dir + name + "/"})
		case !dirsOnly:
			out = append(out, candidate{value: dir + name})
		}
	}
	return out
}

func matching(values []string, current string) []candidate {
	var out []candidate
	for _, value := range values {
		if strings.HasPrefix(value, current) {
			out = append(out, candidate{value: value})
		}
	}
	return out
}

func prefixed(prefix string, in []candidate) []candidate {
	for i := range in {
		in[i].value = prefix + in[i].value
	}
	return in
}

// completionInventory is this machine's installed packages and rentals, read once per
// completion from the records database opened read-only. It never starts the daemon,
// contacts Tensorhub, or creates the local root; any failure completes nothing.
type completionInventory struct {
	installs []records.PackageInstall
	rentals  []records.Rental
}

func (c *completer) local() *completionInventory {
	if c.inventory != nil {
		return c.inventory
	}
	c.inventory = &completionInventory{}
	cfg, problem := config.Load()
	if problem != nil {
		return c.inventory
	}
	store, problem := records.OpenReadOnly(home.Paths(cfg.Home).DB)
	if problem != nil {
		return c.inventory
	}
	defer store.Close()
	c.inventory.installs, _ = store.Installed()
	c.inventory.rentals, _ = store.Rentals()
	return c.inventory
}

func (i *completionInventory) packageNames() []string {
	var names []string
	for _, install := range i.installs {
		names = append(names, install.Package)
	}
	return names
}

func (i *completionInventory) callables() []string {
	var names []string
	for _, install := range i.installs {
		raw, err := os.ReadFile(launch.PackageInterfacePath(install.Dir))
		if err != nil {
			continue
		}
		iface, problem := launch.DecodePackageInterface(raw)
		if problem != nil {
			continue
		}
		for _, function := range iface.PublicNames() {
			names = append(names, install.Package+"/"+function)
		}
	}
	return names
}

func (i *completionInventory) rentalNames() []string {
	var names []string
	for _, rental := range i.rentals {
		if rental.MachineName != "" && !hub.RentalAbsent(rental.State) {
			names = append(names, rental.MachineName)
		}
	}
	return names
}

// splitLine splits the command line before the cursor the way a POSIX shell would for
// quotes and backslashes. It returns the finished words, the unquoted word under the
// cursor, and that word's raw typed text.
func splitLine(line string) (words []string, current, raw string) {
	var word strings.Builder
	inWord, start := false, 0
	var quote rune
	escaped := false
	for i, r := range line {
		switch {
		case escaped:
			word.WriteRune(r)
			escaped = false
		case quote != 0:
			if r == quote {
				quote = 0
			} else if r == '\\' && quote == '"' {
				escaped = true
			} else {
				word.WriteRune(r)
			}
		case r == ' ' || r == '\t' || r == '\n':
			if inWord {
				words = append(words, word.String())
				word.Reset()
				inWord = false
			}
			continue
		case r == '\\':
			escaped = true
		case r == '\'' || r == '"':
			quote = r
		default:
			word.WriteRune(r)
		}
		if !inWord {
			inWord, start = true, i
		}
	}
	if inWord {
		return words, word.String(), line[start:]
	}
	return words, "", ""
}

// shellEscape backslash-quotes the characters bash and zsh would otherwise interpret.
func shellEscape(value string) string {
	var out strings.Builder
	for _, r := range value {
		if strings.ContainsRune(" \t'\"\\$`&;|<>()!*?[]{}#", r) {
			out.WriteByte('\\')
		}
		out.WriteRune(r)
	}
	return out.String()
}

// bashWord trims a candidate to the word readline replaces: the text after the last
// COMP_WORDBREAKS character. For `--asset=./f` that is `./f`; right after `=` it is empty.
func bashWord(candidate, raw string, wordBreaks []string) string {
	if len(wordBreaks) == 0 {
		return candidate
	}
	cut := strings.LastIndexAny(raw, strings.Trim(wordBreaks[0], " \t\n"))
	if cut < 0 {
		return candidate
	}
	return strings.TrimPrefix(candidate, raw[:cut+1])
}

func firstLine(text string) string {
	text, _, _ = strings.Cut(strings.TrimSpace(text), "\n")
	return strings.ReplaceAll(text, "\t", " ")
}
