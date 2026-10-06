package cli

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/resultfiles"
	"github.com/cozy-creator/cozy/internal/runoutputs"
)

// A run's outputs are items rewritten in place (progressive-outputs.md). The machine journals
// each revision as a `product` entry; the shared fold names its item, rev and how it changed;
// this follower writes it into the item's one stable file, `<n>-<output>[-<i>].<ext>` in the
// run's outputs folder: an append by writing only the new tail in place, a replace whole under
// a temporary name then renamed over. The file is the output from its first revision to its
// last. Nothing here reaches the Hub.

// printable is a peer's word as this client keeps it: printable ASCII, the rest `substitute`.
func printable(value string, bound int, substitute rune) string {
	runes := []rune(value)
	if len(runes) > bound {
		runes = runes[:bound]
	}
	for i, r := range runes {
		if r < 0x20 || r > 0x7e {
			runes[i] = substitute
		}
	}
	return string(runes)
}

// itemFile is the item's stable file name in its run's outputs folder.
func itemFile(run string, item runoutputs.Item, mediaType string) string {
	name := strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '.' || r == '_' || r == '-' {
			return r
		}
		return '_'
	}, item.Name(run))
	if mediaType == resultfiles.TreeMediaType {
		return name
	}
	return name + resultfiles.Extension(mediaType)
}
