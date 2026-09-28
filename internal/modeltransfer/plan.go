// Package modeltransfer holds one source-to-destination request as the caller accepted it.
// It deliberately contains no scheduler, provider capability, URL, worker, rental, grant,
// clock, or retry field.
package modeltransfer

type OutputPin struct {
	Name string `json:"name"`
}

type SourceFile struct {
	Member string `json:"member"`
	SHA256 string `json:"sha256"`
	Length int64  `json:"length"`
}

type Plan struct {
	Kind            string
	Destination     string
	Source          string
	SourceSelection string
	SourceLicense   string
	SourceFiles     []SourceFile
	InputLane       string
	SourceProfiles  map[string]string
	Outputs         []OutputPin
}
