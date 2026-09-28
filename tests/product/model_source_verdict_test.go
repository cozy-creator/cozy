package producttest

// The run-205 selection, reduced to the two members that matter: one that verifies and one
// that fails permanently in the first seconds. The digest is the transfer's own selection.
const (
	verdictSelection = "sha256:" + "22222222222222222222222222222222" + "22222222222222222222222222222222"
	verdictGoodSHA   = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	verdictBadSHA    = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	verdictGoodBytes = int64(110_331_470_098)
	verdictBadBytes  = int64(100_000_000_000)
	verdictCode      = "source_fetch_refused"
	verdictDetail    = "the origin answered 401 for b.safetensors and will not answer another way"
)
