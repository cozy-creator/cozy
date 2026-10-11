package packagepublish

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"

	"github.com/cozy-creator/cozy/internal/wheel"
)

func capturedWheelMatches(path string, row RegistryRow) bool {
	file, err := os.Open(path)
	if err != nil {
		return false
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > MaxDependencyWheelBytes || row.Size > 0 && info.Size() != row.Size {
		return false
	}
	hash := sha256.New()
	n, err := io.Copy(hash, io.LimitReader(file, MaxDependencyWheelBytes+1))
	if err != nil || n != info.Size() || hex.EncodeToString(hash.Sum(nil)) != row.SHA256 {
		return false
	}
	identity, problem := wheel.InspectIdentity(path)
	return problem == nil && identity.Distribution == row.Name && identity.Version == row.Version
}
