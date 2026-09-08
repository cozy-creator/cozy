package producttest

import (
	"database/sql"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

func TestDaemonCloseFencesLateWorkspaceStarts(t *testing.T) {
	o := hostOwner(t, "close-fences-late-workspace")
	o.c.Close(100 * time.Millisecond)
	_, _, problem := o.c.EnsureWorker(fakeSpec("late-workspace", "", "--arm", "idle"))
	if problem == nil || problem.ErrName() != "daemon.closing" {
		t.Fatalf("late start was not refused: %v", problem)
	}
	workers, problem := o.store.LiveWorkers()
	fatal(t, problem)
	if len(workers) != 0 {
		t.Fatal("closed daemon left a worker process")
	}
}

func TestDaemonCloseJoinsAnAlreadyAdmittedWorkerStart(t *testing.T) {
	o := hostOwner(t, "close-joins-admitted-start")
	db, err := sql.Open("sqlite", o.l.DB)
	must(t, err)
	defer db.Close()
	_, err = db.Exec("BEGIN IMMEDIATE")
	must(t, err)
	defer db.Exec("ROLLBACK")
	spec := fakeSpec("starting-workspace", "", "--arm", "idle")
	started := make(chan *exit.Error, 1)
	go func() { _, _, problem := o.c.EnsureWorker(spec); started <- problem }()
	// The log exists before the native worker grant crosses our SQLite gate.
	until := time.Now().Add(5 * time.Second)
	log := filepath.Join(o.l.WorkerDir(spec.InstanceID()), "worker.log")
	for {
		if _, err := os.Stat(log); err == nil {
			break
		}
		if time.Now().After(until) {
			t.Fatal("worker start did not reach the grant gate")
		}
		time.Sleep(5 * time.Millisecond)
	}
	closed := make(chan struct{})
	go func() { o.c.Close(100 * time.Millisecond); close(closed) }()
	select {
	case <-closed:
		t.Fatal("close escaped before the admitted process was registered")
	case <-time.After(50 * time.Millisecond):
	}
	_, err = db.Exec("ROLLBACK")
	must(t, err)
	select {
	case problem := <-started:
		fatal(t, problem)
	case <-time.After(5 * time.Second):
		t.Fatal("worker start did not finish")
	}
	select {
	case <-closed:
	case <-time.After(5 * time.Second):
		t.Fatal("close did not reap the admitted start")
	}
	workers, problem := o.store.LiveWorkers()
	fatal(t, problem)
	if len(workers) != 0 {
		t.Fatal("an admitted worker escaped the shutdown census")
	}
}
