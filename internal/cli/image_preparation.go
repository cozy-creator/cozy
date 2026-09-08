package cli

import (
	"context"
	"os/signal"
	"syscall"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
)

// imagePreparer is optional client work. Missing/unqualified local capability
// preserves the ordinary raw path; Fingerprint still enforces its admitted bound.
// Runtime retains resized derivatives in its shared system temporary media directory.
func imagePreparer(ctx *Context) launch.ImagePreparer {
	var tool launch.RuntimeCLI
	var profile launch.ImagePreparationProfile
	checked, unavailable := false, false
	prepare := func(source string, kind launch.AssetsKind) (string, *exit.Error) {
		if unavailable {
			return source, nil
		}
		callCtx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
		defer stop()
		if !checked {
			checked = true
			var problem *exit.Error
			tool, problem = launch.ImagePreparationTool(ctx.Cfg.Home, ctx.Cfg.Tool())
			if callCtx.Err() != nil {
				return "", exit.New(exit.Canceled, "image preparation was canceled")
			}
			if problem != nil {
				unavailable = true
				return source, nil
			}
			profile, problem = tool.ImagePreparationProfile(callCtx)
			if callCtx.Err() != nil {
				return "", exit.New(exit.Canceled, "image preparation was canceled")
			}
			if problem != nil || !profile.Qualified {
				unavailable = true
				return source, nil
			}
		}
		if kind.Preparation == nil || profile.Profile != kind.Preparation.Profile {
			return source, nil
		}
		prepared, problem := tool.PrepareImage(callCtx, source, kind)
		if problem != nil {
			return "", problem
		}
		if prepared.Profile != profile.Profile {
			return "", exit.Internalf("image helper changed its qualified profile")
		}
		if prepared.Status == "raw" {
			if prepared.Path != source {
				return "", exit.Internalf("image helper changed the raw source path")
			}
			return source, nil
		}
		if prepared.Status != "prepared" {
			return "", exit.Internalf("image helper returned unknown status %q", prepared.Status)
		}
		return prepared.Path, nil
	}
	return prepare
}
