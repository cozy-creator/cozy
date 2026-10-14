package packagepublish

import (
	"fmt"
	"strings"
	"testing"
)

func TestLockedHubCalleesUseTheRunHub(t *testing.T) {
	lock := func(index string) []byte {
		return []byte(fmt.Sprintf("version=1\n[[package]]\nname='qwen-image-2'\nversion='0.5.0'\nsource={registry=%q}\n", index))
	}
	for _, target := range []string{"https://tensorhub.com", "https://TENSORHUB.COM:443/"} {
		_, callees, problem := HubWheels(lock("https://tensorhub.com/v1/index/fidika/simple/"), target)
		if problem != nil || callees["qwen-image-2"] != "fidika/qwen-image-2" {
			t.Fatalf("same Hub changed the callee namespace: %v %v", callees, problem)
		}
	}
	_, _, problem := HubWheels(lock("https://tensorhub.com/v1/index/fidika/simple/"), "http://127.0.0.1:8819")
	if problem == nil || problem.Name != "local_package_lock_hub_mismatch" {
		t.Fatalf("different Hub admitted: %v", problem)
	}
	for _, detail := range []string{"qwen-image-2", "https://tensorhub.com", "fidika", "http://127.0.0.1:8819"} {
		if !strings.Contains(problem.Error(), detail) {
			t.Fatalf("refusal omits %q: %v", detail, problem)
		}
	}
	if !strings.Contains(problem.Remedy, "cozy package lock") || !strings.Contains(problem.Remedy, "http://127.0.0.1:8819") {
		t.Fatalf("refusal lacks the selected Hub's relock remedy: %v", problem.Remedy)
	}
	_, callees, problem := HubWheels(lock("https://pypi.org/simple/"), "http://127.0.0.1:8819")
	if problem != nil || len(callees) != 0 {
		t.Fatalf("public dependency became a Hub callee: %v %v", callees, problem)
	}
}
