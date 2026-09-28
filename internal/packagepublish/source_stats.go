package packagepublish

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"

	"github.com/cozy-creator/cozy/internal/exit"
)

// SourceStamp is a sync optimization, never installation identity or admission.
type SourceStamp struct{ Size, Modified int64 }

func SourceStatsUnchanged(installDir string, current map[string]SourceStamp) bool {
	raw, err := os.ReadFile(filepath.Join(installDir, "source-stats.json"))
	var prior map[string]SourceStamp
	return err == nil && len(raw) <= 16<<20 && json.Unmarshal(raw, &prior) == nil && reflect.DeepEqual(prior, current)
}

func RecordSourceStats(installDir string, stats map[string]SourceStamp) *exit.Error {
	raw, err := json.Marshal(stats)
	if err != nil {
		return exit.Internalf("cannot encode editable sync metadata: %s", err)
	}
	if err := os.WriteFile(filepath.Join(installDir, "source-stats.json"), raw, 0600); err != nil {
		return exit.Internalf("cannot retain editable sync metadata: %s", err)
	}
	return nil
}

// InvocationSourceUnchanged reports whether a run's snapshot install was captured from
// exactly this live tree: its files and its local dependencies, by size and modification.
func InvocationSourceUnchanged(installDir string, live map[string]SourceStamp) bool {
	raw, err := os.ReadFile(filepath.Join(installDir, "invocation-source-stats.json"))
	var prior map[string]SourceStamp
	return err == nil && len(raw) <= 16<<20 && json.Unmarshal(raw, &prior) == nil && reflect.DeepEqual(prior, live)
}

// RecordInvocationSource names the live tree a run's snapshot install was captured from.
func RecordInvocationSource(installDir string, live map[string]SourceStamp) *exit.Error {
	raw, err := json.Marshal(live)
	if err != nil {
		return exit.Internalf("cannot encode the snapshot's source: %s", err)
	}
	if err := os.WriteFile(filepath.Join(installDir, "invocation-source-stats.json"), raw, 0600); err != nil {
		return exit.Internalf("cannot retain the snapshot's source: %s", err)
	}
	return nil
}
