package launch

import (
	"context"
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
)

// The host Runtime owns the image profile and codec behavior. These are ordinary
// typed questions through RuntimeCLI; package code is never imported here.
type ImagePreparationProfile struct {
	Profile   string `json:"profile"`
	Qualified bool   `json:"qualified"`
}

type PreparedImage struct {
	Status    string `json:"status"`
	Path      string `json:"path"`
	MediaType string `json:"media_type"`
	Length    int64  `json:"length"`
	Profile   string `json:"profile"`
}

// ImagePreparationTool selects the same qualified host Runtime as other neutral
// capabilities; a package's importing interpreter is never an analysis tool.
func ImagePreparationTool(root string, env []string) (RuntimeCLI, *exit.Error) {
	binary, problem := hostruntime.Path(env)
	if problem != nil {
		return RuntimeCLI{}, problem
	}
	return RuntimeCLI{Bin: binary, Dir: root, Home: root, Env: env}, nil
}

func (r RuntimeCLI) ImagePreparationProfile(ctx context.Context) (ImagePreparationProfile, *exit.Error) {
	var result ImagePreparationProfile
	problem := r.callContext(ctx, &result, "image-preparation-profile")
	return result, problem
}

func (r RuntimeCLI) PrepareImage(ctx context.Context, source string, kind AssetsKind) (PreparedImage, *exit.Error) {
	var result PreparedImage
	body, err := json.Marshal(struct {
		Source          string            `json:"source_path"`
		Prepare         *ImagePreparation `json:"prepare"`
		MediaTypes      []string          `json:"media_types"`
		MaxBytes        int64             `json:"max_bytes"`
		MaxDecodedBytes int64             `json:"max_decoded_bytes"`
	}{source, kind.Preparation, kind.MediaTypes, effectiveAssetMax(kind.MaxBytes), kind.MaxDecodedBytes})
	if err != nil {
		return result, exit.Internalf("cannot encode image preparation: %s", err)
	}
	problem := r.callInputContext(ctx, body, &result, "image-prepare")
	return result, problem
}
