package packagepublish

import (
	"encoding/hex"
	"net/url"
	"path"
	"regexp"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

var pytorchIndexPath = regexp.MustCompile(`^/whl/(cpu|cu[0-9]{3}|rocm[0-9]+\.[0-9]+|xpu)$`)

// Official framework wheels retain their exact URL and hash in each private
// environment. Runtime fetches them through bounded public artifact storage.
func pytorchRegistryWheel(name, version, index string, wheels []registryWheel, targetPython ...string) (registryWheel, *exit.Error) {
	refuse := func() (registryWheel, *exit.Error) {
		return registryWheel{}, exit.Named(exit.Conflict, "base_dependency_origin_unsupported", "base dependency %s has no exact supported official PyTorch wheel", name)
	}
	if name != "torch" && name != "torchvision" && name != "torchaudio" {
		return refuse()
	}
	origin, err := url.Parse(index)
	if err != nil || origin.Scheme != "https" || origin.Host != "download.pytorch.org" || origin.User != nil || origin.RawQuery != "" || origin.Fragment != "" || !pytorchIndexPath.MatchString(origin.Path) {
		return refuse()
	}
	for _, candidate := range wheels {
		if candidate.Size < 0 || candidate.Size > MaxRegistryWheelBytes {
			return refuse()
		}
		object, err := url.Parse(candidate.URL)
		if err != nil || object.Scheme != "https" || (object.Host != "download.pytorch.org" && object.Host != "download-r2.pytorch.org") || object.User != nil || object.RawQuery != "" || object.Fragment != "" || object.Path != origin.Path+"/"+path.Base(object.Path) {
			return refuse()
		}
		filename := path.Base(object.Path)
		if !strings.HasPrefix(filename, strings.ReplaceAll(name, "-", "_")+"-"+version+"-") || strings.ContainsAny(filename, "%\\") {
			return refuse()
		}
		hash := candidate.Hashes["sha256"]
		if _, err := hex.DecodeString(hash); err != nil || len(hash) != 64 || strings.ToLower(hash) != hash {
			return registryWheel{}, exit.Named(exit.Conflict, "base_dependency_hash_invalid", "base dependency %s has no exact SHA-256 wheel hash", name)
		}
	}
	selected, problem := selectRegistryWheel(name, registryPackage{Name: name, Version: version, Wheels: wheels}, targetPython...)
	if problem != nil {
		return registryWheel{}, problem
	}
	// Preserve the official object path and captured hash while using the
	// canonical host; the r2 mirror rejects the selected CPU artifact.
	object, _ := url.Parse(selected.URL)
	object.Host = "download.pytorch.org"
	selected.URL = object.String()
	return selected, nil
}
