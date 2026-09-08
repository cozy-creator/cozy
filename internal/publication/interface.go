package publication

import (
	_ "embed"
	"github.com/cozy-creator/cozy/internal/canonical"
)

const Module = "cozy_runtime.author.publication"

//go:embed interface.json
var Interface []byte

func InterfaceDigest() string {
	digest, _ := canonical.Spell(canonical.Digest(Interface))
	return digest
}
