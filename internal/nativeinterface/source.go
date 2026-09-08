// Package nativeinterface contains closed first-party call contracts, independent of package builds.
package nativeinterface

import (
	_ "embed"
	"github.com/cozy-creator/cozy/internal/canonical"
)

//go:embed source.json
var SourceDocument []byte

const SourceModule = "cozy_runtime.author.sources"

func SourceDigest() []byte { return canonical.Digest(SourceDocument) }
