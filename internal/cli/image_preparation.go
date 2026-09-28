package cli

import (
	"context"
	"os/signal"
	"syscall"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
)

// imagePreparer is optional client work. An image its header shows already fits the policy
// is sent as it is, with no Runtime process; only a larger one asks the host Runtime, whose
// image-prepare answers raw when it lacks the profile or its codecs. Any answer other than a
// prepared image under the declared profile uploads the source unchanged. Runtime retains
// resized derivatives in its shared system temporary media directory.
func imagePreparer(ctx *Context) launch.ImagePreparer {
	var tool *launch.RuntimeCLI
	unavailable := false
	return func(source string, kind launch.AssetsKind) (string, *exit.Error) {
		if kind.Preparation == nil || unavailable {
			return source, nil
		}
		if fits, known := launch.ImageFits(source, *kind.Preparation); known && fits {
			return source, nil
		}
		callCtx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
		defer stop()
		if tool == nil {
			found, problem := launch.ImagePreparationTool(ctx.Cfg.Home, ctx.Cfg.Tool())
			if problem != nil {
				unavailable = true
				return source, nil
			}
			tool = &found
		}
		prepared, problem := tool.PrepareImage(callCtx, source, kind)
		if callCtx.Err() != nil {
			return "", exit.New(exit.Canceled, "image preparation was cancelled")
		}
		if problem != nil {
			unavailable = true // a Runtime that cannot answer one image answers no other
			return source, nil
		}
		if prepared.Profile != kind.Preparation.Profile || prepared.Status != "prepared" || prepared.Path == "" {
			return source, nil
		}
		return prepared.Path, nil
	}
}
