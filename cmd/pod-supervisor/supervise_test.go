package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestRunLegsReapsAdapterAfterReceiptFailure(t *testing.T) {
	temp := t.TempDir()
	readyPath := filepath.Join(temp, "ready")
	termPath := filepath.Join(temp, "term")
	scriptPath := filepath.Join(temp, "fake-adapter")
	script := fmt.Sprintf("#!/bin/sh\n/bin/sleep 60 &\nchild=$!\n"+
		"trap 'kill \"$child\" 2>/dev/null; wait \"$child\" 2>/dev/null; printf term > %q; exit 0' TERM\n"+
		"printf ready > %q\nwait \"$child\"\n", termPath, readyPath)
	if err := os.WriteFile(scriptPath, []byte(script), 0o700); err != nil {
		t.Fatalf("write fake adapter: %v", err)
	}
	var adapter *leg
	t.Cleanup(func() {
		if adapter != nil && !adapter.exited() {
			adapter.kill()
			<-adapter.done
		}
	})

	mediaDone := make(chan struct{})
	media := &leg{name: "media plane", done: mediaDone, terminate: func() { close(mediaDone) }}
	planted := errors.New("planted receipt failure")
	err := runLegs(context.Background(), context.Background(), func() {}, media, legRuntime{
		startAdapter: func() (started *leg, err error) {
			adapter, err = startAdapterAt(scriptPath, config{})
			return adapter, err
		},
		publishReceipt: func(context.Context, []*leg) error {
			waitForFile(t, readyPath)
			return planted
		},
		supervise: func(context.Context, []*leg) error {
			t.Fatal("supervise ran after the planted receipt failure")
			return nil
		},
	})
	if !errors.Is(err, planted) {
		t.Fatalf("runLegs error = %v, want planted receipt failure", err)
	}
	if _, err := os.Stat(termPath); err != nil {
		t.Fatalf("fake adapter did not record SIGTERM before supervisor return: %v", err)
	}
	if adapter == nil || !adapter.exited() || adapter.err != nil {
		t.Fatal("fake adapter was not reaped before supervisor return")
	}
}

func waitForFile(t *testing.T, path string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(path); err == nil {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", path)
		}
		time.Sleep(10 * time.Millisecond)
	}
}
