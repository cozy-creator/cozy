// Command genmachinewords regenerates the machine-name vocabulary
// (internal/rentalid/words.go) from anime characters ranked by popularity.
//
// It pages Jikan's MyAnimeList mirror — GET https://api.jikan.moe/v4/top/characters
// (25/page, sorted by favorites) — caching each raw page under -cache so a rerun
// costs no requests, DROPS the top -drop
// characters (their names may never mint), and normalizes the rest to machine
// words: given-name token, ascii-folded, lowercased, stripped to [a-z0-9],
// 3-16 characters, never a reserved name, deduplicated popularity-first, and no
// two kept words one letter apart. It widens past -take until at least -min
// words are kept.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/format"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

type character struct {
	MalID     int    `json:"mal_id"`
	Name      string `json:"name"` // "Family, Given" for most; single names stand alone
	Favorites int    `json:"favorites"`
}

type page struct {
	Pagination struct {
		HasNextPage bool `json:"has_next_page"`
	} `json:"pagination"`
	Data []character `json:"data"`
}

func main() {
	cache := flag.String("cache", "", "directory of cached Jikan pages (required; fetches missing pages)")
	take := flag.Int("take", 10000, "characters to rank before widening")
	drop := flag.Int("drop", 100, "top ranks whose names are banned from minting")
	min := flag.Int("min", 5000, "widen past -take until this many words are kept")
	out := flag.String("out", "internal/rentalid/words.go", "generated vocabulary file")
	dropped := flag.String("dropped", "tests/product/testdata/machine-words-dropped.txt", "banned top-name list, consumed by tests/product")
	offline := flag.Bool("offline", false, "use the cache only; refuse rather than fetch")
	flag.Parse()
	if *cache == "" {
		fatal(fmt.Errorf("-cache is required"))
	}
	fatal(os.MkdirAll(*cache, 0o755))

	banned := map[string]bool{}
	var bannedInOrder []string
	bannedByLength := map[int][]string{} // a kept word may not sit one letter from a banned one
	kept := map[string]bool{}
	byLength := map[int][]string{} // kept words, for the one-letter-apart law
	rank, lastRank, exhausted := 0, 0, false
	for pageNo := 1; !exhausted && (rank < *take || len(kept) < *min); pageNo++ {
		p, err := load(*cache, pageNo, *offline)
		fatal(err)
		for _, c := range p.Data {
			rank++
			word, ok := normalize(c)
			if rank <= *drop {
				if ok && !banned[word] {
					banned[word] = true
					bannedInOrder = append(bannedInOrder, word)
					bannedByLength[len(word)] = append(bannedByLength[len(word)], word)
				}
				lastRank = rank
				continue
			}
			if rank > *take && len(kept) >= *min {
				break
			}
			lastRank = rank
			if !ok || banned[word] || kept[word] ||
				confusable(word, bannedByLength) || confusable(word, byLength) {
				continue
			}
			kept[word] = true
			byLength[len(word)] = append(byLength[len(word)], word)
		}
		exhausted = !p.Pagination.HasNextPage || len(p.Data) == 0
	}
	if len(kept) < *min {
		fatal(fmt.Errorf("only %d words kept after rank %d; the ranking is exhausted", len(kept), lastRank))
	}

	words := make([]string, 0, len(kept))
	for w := range kept {
		words = append(words, w)
	}
	sort.Strings(words)
	fatal(writeWords(*out, words, *drop, lastRank))
	fatal(writeDropped(*dropped, bannedInOrder, *drop))
	fmt.Printf("kept %d words from ranks %d-%d; banned %d top names\n", len(words), *drop+1, lastRank, len(bannedInOrder))
}

// load returns one cached page, fetching and caching it first when absent.
func load(cache string, pageNo int, offline bool) (*page, error) {
	path := filepath.Join(cache, fmt.Sprintf("page-%03d.json", pageNo))
	raw, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		if offline {
			return nil, fmt.Errorf("page %d is not cached and -offline is set", pageNo)
		}
		if raw, err = fetch(pageNo); err == nil {
			err = os.WriteFile(path, raw, 0o644)
		}
	}
	if err != nil {
		return nil, err
	}
	var p page
	if err := json.Unmarshal(raw, &p); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &p, nil
}

func fetch(pageNo int) ([]byte, error) {
	for attempt := 1; ; attempt++ {
		time.Sleep(1100 * time.Millisecond) // Jikan asks for at most 1 request/second sustained
		request, _ := http.NewRequest("GET", fmt.Sprintf("https://api.jikan.moe/v4/top/characters?page=%d", pageNo), nil)
		request.Header.Set("Accept", "application/json")
		request.Header.Set("User-Agent", "cozy-genmachinewords/1.0")
		resp, err := http.DefaultClient.Do(request)
		if err == nil {
			raw := make([]byte, 0, 1<<16)
			buf := make([]byte, 1<<15)
			for {
				n, readErr := resp.Body.Read(buf)
				raw = append(raw, buf[:n]...)
				if readErr != nil {
					break
				}
			}
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				fmt.Printf("fetched page %d\n", pageNo)
				return raw, nil
			}
			err = fmt.Errorf("page %d: HTTP %d", pageNo, resp.StatusCode)
			if wait, _ := strconv.Atoi(resp.Header.Get("Retry-After")); wait > 0 {
				time.Sleep(time.Duration(wait) * time.Second)
			}
		}
		if attempt >= 8 {
			return nil, err
		}
		fmt.Printf("%v; retrying\n", err)
		time.Sleep(time.Duration(10*attempt) * time.Second)
	}
}

// reserved: rentalid refuses "local"; the Windows device names would be hostile
// filenames the moment a word names a cert file.
var reserved = map[string]bool{
	"local": true, "con": true, "prn": true, "aux": true, "nul": true,
	"com1": true, "com2": true, "com3": true, "com4": true, "com5": true,
	"com6": true, "com7": true, "com8": true, "com9": true,
	"lpt1": true, "lpt2": true, "lpt3": true, "lpt4": true, "lpt5": true,
	"lpt6": true, "lpt7": true, "lpt8": true, "lpt9": true,
	"latina": true, // removed by the owner
}

// fold maps the accented latin these romanizations use onto ascii. A rune
// outside ascii and this table drops the whole entry — no guessing.
var fold = map[rune]string{
	'á': "a", 'à': "a", 'â': "a", 'ä': "a", 'ã': "a", 'å': "a", 'ā': "a", 'ă': "a", 'ą': "a",
	'é': "e", 'è': "e", 'ê': "e", 'ë': "e", 'ē': "e", 'ĕ': "e", 'ė': "e", 'ę': "e", 'ě': "e",
	'í': "i", 'ì': "i", 'î': "i", 'ï': "i", 'ī': "i", 'į': "i", 'ı': "i",
	'ó': "o", 'ò': "o", 'ô': "o", 'ö': "o", 'õ': "o", 'ō': "o", 'ŏ': "o", 'ő': "o", 'ø': "o",
	'ú': "u", 'ù': "u", 'û': "u", 'ü': "u", 'ū': "u", 'ŭ': "u", 'ů': "u", 'ű': "u",
	'ý': "y", 'ÿ': "y", 'ñ': "n", 'ń': "n", 'ň': "n", 'ç': "c", 'ć': "c", 'č': "c",
	'ś': "s", 'š': "s", 'ş': "s", 'ź': "z", 'ż': "z", 'ž': "z", 'ł': "l", 'ľ': "l",
	'ď': "d", 'đ': "d", 'ť': "t", 'ř': "r", 'ŕ': "r", 'ğ': "g", 'ĝ': "g",
	'æ': "ae", 'œ': "oe", 'ß': "ss", 'þ': "th", 'ð': "d",
}

// normalize turns one ranked character into its machine word: the given-name
// token (MyAnimeList writes "Family, Given"; a name with no comma stands whole,
// first token), folded to lowercase ascii and stripped to [a-z0-9], length 3-16,
// starting with a letter.
func normalize(c character) (string, bool) {
	given := c.Name
	if at := strings.LastIndex(given, ","); at >= 0 {
		given = given[at+1:]
	}
	given = strings.TrimSpace(given)
	if fields := strings.Fields(given); len(fields) > 0 {
		given = fields[0]
	}
	var b strings.Builder
	for _, r := range strings.ToLower(given) {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' {
			b.WriteRune(r)
		} else if folded, ok := fold[r]; ok {
			b.WriteString(folded)
		} else if r >= 128 {
			return "", false // a script this table cannot fold stays out
		}
		// ascii punctuation — ' - . — drops out of the word
	}
	word := b.String()
	if len(word) < 3 || len(word) > 16 || word[0] < 'a' || word[0] > 'z' || reserved[word] {
		return "", false
	}
	return word, true
}

// confusable reports whether word is one letter apart from any word in the set —
// the cl-088 law: no two vocabulary words may be confused when typed from memory,
// and no vocabulary word may be confused with a banned top name.
func confusable(word string, byLength map[int][]string) bool {
	for _, other := range byLength[len(word)] {
		if oneSubstitution(word, other) {
			return true
		}
	}
	for _, other := range byLength[len(word)+1] {
		if oneInsertion(word, other) {
			return true
		}
	}
	for _, other := range byLength[len(word)-1] {
		if oneInsertion(other, word) {
			return true
		}
	}
	return false
}

func oneSubstitution(a, b string) bool {
	differ := 0
	for i := range a {
		if a[i] != b[i] {
			differ++
		}
	}
	return differ == 1
}

// oneInsertion reports whether longer is shorter with one letter inserted.
func oneInsertion(shorter, longer string) bool {
	i := 0
	for i < len(shorter) && shorter[i] == longer[i] {
		i++
	}
	return shorter[i:] == longer[i+1:]
}

func writeWords(path string, words []string, drop, lastRank int) error {
	var b bytes.Buffer
	fmt.Fprintf(&b, `// Code generated by scripts/genmachinewords. DO NOT EDIT.

package rentalid

// words is the vocabulary a machine name is drawn from: the given names of anime
// characters, ranked by favorites on MyAnimeList via Jikan
// (GET https://api.jikan.moe/v4/top/characters, fetched %s). The top %d
// characters — the Goku/Naruto tier — are DROPPED and their names can never
// mint (tests/product/testdata/machine-words-dropped.txt), and no kept word
// sits one letter from a banned one — goku stays out because gokuu is banned;
// ranks %d-%d supply the pool. Each word is the character's given-name token,
// ascii-folded, lowercased, stripped to [a-z0-9], 3-16 characters, deduplicated
// popularity-first, never a reserved name, and no two words are one letter
// apart, so a name survives being typed from memory.
//
// Regenerate: go run ./scripts/genmachinewords -cache <dir>
var words = []string{
`, time.Now().Format("2006-01-02"), drop, drop+1, lastRank)
	for i := 0; i < len(words); i += 8 {
		quoted := make([]string, 0, 8)
		for _, w := range words[i:minInt(i+8, len(words))] {
			quoted = append(quoted, fmt.Sprintf("%q,", w))
		}
		fmt.Fprintf(&b, "\t%s\n", strings.Join(quoted, " "))
	}
	b.WriteString("}\n")
	formatted, err := format.Source(b.Bytes())
	if err != nil {
		return err
	}
	return os.WriteFile(path, formatted, 0o644)
}

func writeDropped(path string, banned []string, drop int) error {
	var b bytes.Buffer
	fmt.Fprintf(&b, "# The normalized given names of MyAnimeList's top %d characters by favorites\n", drop)
	fmt.Fprintf(&b, "# (fetched %s): banned from the machine-name vocabulary forever.\n", time.Now().Format("2006-01-02"))
	fmt.Fprintf(&b, "# Generated by scripts/genmachinewords. DO NOT EDIT.\n")
	for _, w := range banned {
		fmt.Fprintln(&b, w)
	}
	return os.WriteFile(path, b.Bytes(), 0o644)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func fatal(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
