package producttest

import (
	"database/sql"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func compositionTail(value string) string {
	if len(value) > 8192 {
		return strings.ToValidUTF8(value[len(value)-8192:], "?")
	}
	return value
}

// A successful result is not proof that the daemon released its owned processes.
// Keep the home (including shutdown logs) if normal teardown leaves one behind.
func compositionDown(t *testing.T, root, path string) {
	t.Helper()
	processes := map[int]string{}
	db, err := sql.Open("sqlite", "file:"+filepath.Join(root, "creator.sqlite")+"?mode=ro")
	if err == nil {
		db.SetMaxOpenConns(1)
		_, err = db.Exec(`PRAGMA busy_timeout=5000`)
		var rows *sql.Rows
		if err == nil {
			rows, err = db.Query(`SELECT pid,birth FROM worker_processes WHERE pid>0`)
		}
		if err == nil {
			for rows.Next() {
				var pid int
				var birth string
				if err = rows.Scan(&pid, &birth); err != nil {
					break
				}
				processes[pid] = birth
			}
			if err == nil {
				err = rows.Err()
			}
			_ = rows.Close()
		}
		_ = db.Close()
	}
	if err != nil {
		t.Errorf("read owned worker identities before teardown: %v", err)
	}
	code, out := runCozyPath(t, root, path, "down", "--all", "--json")
	if code != 0 {
		t.Errorf("normal composition teardown refused (%d): %s", code, compositionTail(out))
	}
	if !reapMachineRuntimeRoot(root) {
		t.Error("owned test Runtime remains alive; preserve its home")
	}
	deadline := time.Now().Add(30 * time.Second)
	for len(processes) > 0 {
		for pid, birth := range processes {
			raw, err := os.ReadFile(filepath.Join("/proc", strconv.Itoa(pid), "stat"))
			if errors.Is(err, os.ErrNotExist) {
				delete(processes, pid)
				continue
			}
			if err != nil {
				t.Errorf("read owned worker %d during teardown: %v", pid, err)
				return
			}
			// Compare the kernel start time so a recycled PID is never mistaken
			// for a worker this test owns. Field 22 follows the parenthesized name.
			fields := strings.Fields(string(raw)[strings.LastIndex(string(raw), ")")+1:])
			if len(fields) < 20 || birth == "" {
				t.Errorf("owned worker %d has no verifiable process birth", pid)
				return
			}
			if fields[19] != birth {
				delete(processes, pid)
			}
		}
		if len(processes) == 0 {
			return
		}
		if time.Now().After(deadline) {
			t.Errorf("normal teardown left owned worker identities alive: %v; down: %s", processes, compositionTail(out))
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
}
