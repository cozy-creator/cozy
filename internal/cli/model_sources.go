package cli

import (
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// sourceRentalBytes sizes a rental bought for a provider-source call as `cozy rental new
// --model` does: the source's reviewed carriers, after the header preflight that refuses
// before anything is bought (tfs-076). Provider reads only; the machine makes the source.
func sourceRentalBytes(ctx *Context, models []orchestrator.ModelRef) (int64, *exit.Error) {
	cwd, err := os.Getwd()
	if err != nil {
		return 0, exit.Internalf("cannot resolve the current directory: %s", err)
	}
	parsed, problem := modelsource.Parse(models[0].Source, cwd)
	if problem != nil {
		return 0, problem
	}
	var profiles []string
	for _, model := range models {
		profiles = append(profiles, model.Profiles...)
	}
	slices.Sort(profiles)
	plan, problem := planNativeIngest(ctx, parsed, slices.Compact(profiles))
	return plan.source.Bytes, problem
}

// uploadDestination is where a function's weights outputs go: `--upload-to`, or a
// conversion's second positional. The repository must be the caller's own.
func uploadDestination(ctx *Context, ep *launch.Entrypoint, positional string) (string, *exit.Error) {
	destination := strings.TrimSpace(ctx.Inv.Value("--upload-to"))
	if positional != "" {
		if destination != "" && destination != positional {
			return "", exit.Usagef("destination %s disagrees with --upload-to=%s", positional, destination)
		}
		destination = positional
	}
	if destination == "" {
		return "", nil
	}
	if len(ep.WeightsOutputs) == 0 {
		return "", exit.Usagef("--upload-to needs a function that declares weights outputs; %s declares none", ep.Name)
	}
	ref, problem := hub.ParseRef(destination)
	if problem != nil {
		return "", problem
	}
	if _, problem := ownedPublication(ctx, ref); problem != nil {
		return "", problem
	}
	return ref.String(), nil
}

// treeDirectories are `--asset <field>=<dir>` trees as `ref=<absolute directory>`.
func treeDirectories(values []string) ([]string, *exit.Error) {
	out := make([]string, 0, len(values))
	for _, value := range values {
		ref, dir, _ := strings.Cut(value, "=")
		dir, err := config.ExpandHome(dir)
		if err != nil {
			return nil, exit.New(exit.NotFound, "cannot resolve input directory home: %s", err)
		}
		if info, err := os.Stat(dir); err != nil || !info.IsDir() {
			return nil, exit.New(exit.NotFound, "--asset %s: %s is not a directory", strings.TrimPrefix(ref, "asset:"), dir)
		}
		absolute, err := filepath.Abs(dir)
		if err != nil {
			return nil, exit.New(exit.Validation, "input tree directory is invalid")
		}
		out = append(out, ref+"="+absolute)
	}
	return out, nil
}

// modelPublishHint names each new checkpoint the run published as a lane of its destination.
func modelPublishHint(life api.Lifecycle) string {
	if life.ModelDestination == "" || len(life.ModelOutputs) == 0 || strings.HasPrefix(life.ModelDestination, "local/") {
		return ""
	}
	command := "cozy model publish " + life.ModelDestination + " --release <label>"
	for _, name := range slices.Sorted(maps.Keys(life.ModelOutputs)) {
		command += " --lane " + name + "=" + life.ModelOutputs[name]
	}
	return command
}

// modelQuantizeHint follows one new checkpoint that is not itself a quantized lane with the
// verb that writes one, on the rental the command named.
func modelQuantizeHint(ctx *Context, life api.Lifecycle) string {
	if len(life.ModelOutputs) != 1 {
		return ""
	}
	for name, checkpoint := range life.ModelOutputs {
		if slices.Contains(quantizedLanes, name) {
			return ""
		}
		command := "cozy model quantize " + life.ModelDestination + "#" + checkpoint + " --" + quantizedLanes[0]
		if rental := ctx.Inv.Value("--rental"); rental != "" {
			command += " --rental=" + rental
		}
		return command + " --await"
	}
	return ""
}
