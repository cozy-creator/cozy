package cli

import (
	"errors"
	"fmt"
	"runtime/debug"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/output"
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

// emit is the only success-document seam.
func emit(ctx *Context, document output.Document) *exit.Error {
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

// compactRecord keeps command handlers authoritative for all available facts while
// making the smaller default an explicit product decision.
func compactRecord(fields []output.Field, defaults ...string) output.Record {
	byName := make(map[string]output.Field, len(fields))
	for _, field := range fields {
		byName[field.K] = field
	}
	compact := make([]output.Field, 0, len(defaults))
	for _, name := range defaults {
		field, ok := byName[name]
		if !ok {
			panic(fmt.Sprintf("compact output field %q is absent", name))
		}
		compact = append(compact, field)
	}
	return output.Record{Fields: compact, AllFields: fields}
}
