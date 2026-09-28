package publication

import (
	_ "embed"
)

const Module = "cozy_runtime.author.publication"

//go:embed interface.json
var Interface []byte
