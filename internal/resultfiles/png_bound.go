package resultfiles

import (
	"image/png"
	"os"

	"github.com/cozy-creator/cozy/internal/exit"
)

// VerifyPNGDecodedBound checks the immutable RGB8 representation promised by
// AssetBound, without allocating a pixel buffer for an oversized compressed image.
func VerifyPNGDecodedBound(path string, maximum int64) *exit.Error {
	file, err := os.Open(path)
	if err != nil {
		return exit.New(exit.Conflict, "cannot inspect retained PNG result")
	}
	defer file.Close()
	config, err := png.DecodeConfig(file)
	if err != nil || config.Width <= 0 || config.Height <= 0 || maximum <= 0 ||
		int64(config.Width) > maximum/3/int64(config.Height) {
		return exit.New(exit.Conflict, "PNG result exceeds its captured decoded bound or is invalid")
	}
	return nil
}
