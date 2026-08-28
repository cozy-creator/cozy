package app

import (
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/render"
	"github.com/cozy-creator/cozy-creator/internal/wheel"
)

// handlePack is the local half of the ONE packing seam (th-039). The tensorhub env-lane
// build stage calls `wheel.Pack` directly (tensorhub-build.md §1 stage 2e); an author
// calls it through this verb, on the same tree, and gets the same
// `project_wheel_digest` — that equality is the whole point, so this handler adds no
// step of its own between argv and the packer.
func handlePack(ctx *Context) *exit.Error {
	res, e := wheel.Pack(wheel.Request{
		Tree:    ctx.Inv.Args[0],
		Name:    ctx.Inv.Value("--name"),
		Version: ctx.Inv.Value("--version"),
		OutDir:  ctx.Inv.Value("--out"),
	})
	if e != nil {
		return e
	}
	rec := render.Record{
		Kind: "pack",
		Fields: []render.Field{
			{K: "wheel", V: res.Path},
			{K: "name", V: res.Name},
			{K: "version", V: res.Version},
			{K: "tag", V: res.Tag},
			{K: "project_wheel_digest", V: res.Digest},
			{K: "tree_digest", V: res.TreeDigest},
			{K: "packer", V: res.PackerVersion},
			{K: "files", V: res.Files},
			{K: "size", V: render.Bytes(res.Bytes)},
			{K: "entries", V: strings.Join(res.Entries, " ")},
		},
		Notes: []string{
			"the digest is a function of (canonical tree, distribution identity, packer version) alone — " +
				"no clock, locale, timezone, umask or host reading enters it",
			"no `[build-system]` was consulted, the project was not imported, and no PEP 517 hook fired",
		},
	}
	return emit(ctx, rec)
}
