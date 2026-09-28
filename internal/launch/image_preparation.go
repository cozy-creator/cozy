package launch

import (
	"context"
	"encoding/json"
	"image"
	_ "image/gif"  // header geometry only
	_ "image/jpeg" // header geometry only
	_ "image/png"  // header geometry only
	"os"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
)

// The host Runtime owns the image profile and codec behavior. These are ordinary
// typed questions through RuntimeCLI; package code is never imported here.
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

// ImageFits answers from the image header alone whether image-fit/1 leaves it unchanged:
// its longest edge and pixel count are within the policy's caps. known is false for another
// profile or a header this host cannot read; only then is the Runtime asked.
func ImageFits(path string, policy ImagePreparation) (fits, known bool) {
	if policy.Profile != "image-fit/1" {
		return false, false
	}
	file, err := os.Open(path)
	if err != nil {
		return false, false
	}
	defer file.Close()
	header, _, err := image.DecodeConfig(file)
	if err != nil || header.Width <= 0 || header.Height <= 0 {
		return false, false
	}
	edge, pixels := int64(max(header.Width, header.Height)), int64(header.Width)*int64(header.Height)
	return (policy.MaxEdge == nil || edge <= *policy.MaxEdge) && (policy.MaxPixels == nil || pixels <= *policy.MaxPixels), true
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
