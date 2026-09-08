package cli

import (
	"context"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/scratch"
)

// imagePreparer is optional client work. Missing/unqualified local capability
// preserves the ordinary raw path; Fingerprint still enforces its admitted bound.
// Claimed scratch survives a hard-killed caller only until the existing sweep.
func imagePreparer(ctx *Context) (launch.ImagePreparer, func()) {
	var work *scratch.Dir
	var tool launch.RuntimeCLI
	var profile launch.ImagePreparationProfile
	checked, unavailable := false, false
	cleanup := func() {
		if work != nil {
			work.Release()
			work = nil
		}
	}
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
		if work == nil {
			layout, problem := home.Open(ctx.Cfg.Home)
			if problem != nil {
				return "", problem
			}
			work, problem = scratch.Temp(layout.Tmp, "image-preparation-")
			if problem != nil {
				return "", problem
			}
		}
		prepared, problem := tool.PrepareImage(callCtx, source, work.Path, kind)
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
		resolved, err := filepath.EvalSymlinks(prepared.Path)
		if err != nil {
			return "", exit.Internalf("prepared image is absent: %s", err)
		}
		owned, err := filepath.EvalSymlinks(work.Path)
		if err != nil {
			return "", exit.Internalf("image preparation scratch is absent: %s", err)
		}
		relative, err := filepath.Rel(owned, resolved)
		if err != nil || relative == "." || relative == ".." || strings.HasPrefix(relative, ".."+string(os.PathSeparator)) || filepath.IsAbs(relative) {
			return "", exit.Internalf("image helper returned a path outside its owned scratch")
		}
		return resolved, nil
	}
	return prepare, cleanup
}
