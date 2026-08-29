package cli

import (
	"errors"
	"runtime/debug"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/output"
)

// Build identity may be stamped with -ldflags -X. The Go VCS stamp supplies
// the client revision sent to Tensorhub when an explicit commit is absent.
var (
	tag    = "0.0.0-dev"
	commit = ""
)

func version() string { return tag }

func buildStamp() (string, bool) {
	revision, dirty := commit, false
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, setting := range info.Settings {
			switch setting.Key {
			case "vcs.revision":
				if revision == "" {
					revision = setting.Value
				}
			case "vcs.modified":
				dirty = setting.Value == "true"
			}
		}
	}
	if revision == "" {
		revision = "unknown"
	}
	if len(revision) > 12 {
		revision = revision[:12]
	}
	return revision, dirty
}

// emit is the only success-document seam. Command-specific documents keep
// state-dependent next actions; otherwise the selected Kong command supplies them.
func emit(ctx *Context, document output.Document) *exit.Error {
	if len(ctx.Next) > 0 {
		document = document.WithDefaultNext(ctx.Next)
	}
	if err := document.Emit(ctx.Out, ctx.Mode()); err != nil {
		var problem *output.Error
		if errors.As(err, &problem) && problem.Class == output.Usage {
			out := exit.Named(exit.Usage, problem.Code, "%s", problem.Message)
			if problem.Remedy != "" {
				out.WithRemedy("%s", problem.Remedy)
			}
			return out.WithNext(problem.Next...)
		}
		return exit.As(err)
	}
	return nil
}
