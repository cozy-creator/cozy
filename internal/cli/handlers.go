package cli

import (
	"errors"
	"fmt"

	"github.com/cozy-creator/cozy/internal/build"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/output"
)

func version() string { return build.Version }

func buildStamp() (string, bool) { return build.Revision() }

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
