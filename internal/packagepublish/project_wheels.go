package packagepublish

import (
	"io"
	"os"
	"path/filepath"
	"sort"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/wheel"
)

// Snapshot explicit artifacts into publication-owned staging before inspection
// and upload. The user's build output remains untouched when Package.Close runs.
func stageProjectWheels(root string, supplied []string) ([]string, *exit.Error) {
	if len(supplied) > 128 {
		return nil, exit.Named(exit.Validation, "project_wheel_count_invalid", "at most 128 project wheels may be published")
	}
	out := filepath.Join(root, "project-wheels")
	if err := os.Mkdir(out, 0o700); err != nil {
		return nil, exit.Internalf("cannot stage project wheels: %v", err)
	}
	paths := make([]string, 0, len(supplied))
	seen := map[string]bool{}
	for _, source := range supplied {
		name := filepath.Base(source)
		if seen[name] {
			return nil, exit.Named(exit.Validation, "project_wheel_duplicate", "project wheel filename %s was supplied more than once", name)
		}
		seen[name] = true
		src, err := os.Open(source)
		if err != nil {
			return nil, exit.Named(exit.Validation, "project_wheel_unreadable", "cannot read %s: %v", source, err)
		}
		info, err := src.Stat()
		if err != nil || !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > wheel.MaxWheelBytes {
			_ = src.Close()
			return nil, exit.Named(exit.Validation, "wheel_size_invalid", "%s must be a nonempty regular wheel at or below %d B", source, wheel.MaxWheelBytes)
		}
		target := filepath.Join(out, name)
		dst, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err != nil {
			_ = src.Close()
			return nil, exit.Internalf("cannot stage project wheel: %v", err)
		}
		n, copyErr := io.Copy(dst, io.LimitReader(src, wheel.MaxWheelBytes+1))
		readClose, writeClose := src.Close(), dst.Close()
		if copyErr != nil || readClose != nil || writeClose != nil || n != info.Size() {
			return nil, exit.Named(exit.Validation, "project_wheel_changed", "%s changed or could not be copied during publication", source)
		}
		paths = append(paths, target)
	}
	sort.Strings(paths)
	return paths, nil
}
