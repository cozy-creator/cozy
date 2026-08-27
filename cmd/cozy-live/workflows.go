package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/app"
	"github.com/cozy-creator/cozy-creator-v2/internal/config"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
)

const workflowCreativeDigest = "sha256:7777777777777777777777777777777777777777777777777777777777777777"

func sectionWorkflows()     { workflowLive(false) }
func sectionWorkflowCrash() { workflowLive(true) }

func workflowLive(crashOnly bool) {
	root := flag("home", filepath.Join(os.TempDir(), "cozy-live", "workflows"))
	must("clearing the workflow live root", os.RemoveAll(root))
	must("creating the workflow live root", os.MkdirAll(root, 0o755))
	installTransportEndpoint(root)
	port := freePort(2981)
	service := startService(root, port, false)
	defer func() { service.stop() }()
	plan := liveWorkflowPlan(root, 150)
	if crashOnly {
		plan = liveWorkflowPlan(root, 1500)
	}

	if !crashOnly {
		head("nine ordinary children, exact backward image handoffs")
		handle := service.call(http.MethodPost, "/v1/local/workflows",
			map[string]any{"plan": plan}, "Idempotency-Key", "workflow-nine")
		workflowID := fmt.Sprint(handle.json()["workflow_id"])
		check("workflow accepted", handle.Status == http.StatusAccepted && workflowID != "",
			handle.brief())
		state, ok := waitWorkflow(service, workflowID, "succeeded", 90*time.Second)
		if !ok {
			return
		}
		steps, _ := state["steps"].([]any)
		children, digests := workflowChildren(steps)
		check("nine serial ordinary children completed", len(steps) == 9 && len(children) == 9,
			fmt.Sprintf("%d steps · %d children", len(steps), len(children)))
		check("every handoff reproduced one accepted image digest", len(digests) == 1,
			strings.Join(mapKeys(digests), ","))
		code, statusDoc, out := cozyJSON(root, "workflow", "status", workflowID)
		structured, _ := statusDoc["steps"].([]any)
		bindingVisible := false
		if len(structured) > 1 {
			second, _ := structured[1].(map[string]any)
			bindings, _ := second["resolved_bindings"].([]any)
			bindingVisible = second["materialized_submission_digest"] != "" && len(bindings) == 1
		}
		check("workflow status --json preserves structured materialization and binding evidence",
			code == 0 && len(structured) == 9 && bindingVisible, firstLine(out))
		download := filepath.Join(root, "workflow-download")
		code, out = cozyRun(root, "workflow", "download", workflowID, "--out", download)
		_, workflowErr := os.Stat(filepath.Join(download, "workflow.json"))
		_, manifestErr := os.Stat(filepath.Join(download, "download-manifest.json"))
		_, receiptErr := os.Stat(filepath.Join(download, "step-09-receipt.json"))
		check("workflow download writes local workflow, child receipt, and manifest documents",
			code == 0 && workflowErr == nil && manifestErr == nil && receiptErr == nil,
			firstLine(out))

		head("durable cancellation prevents every later child")
		cancelPlan := liveWorkflowPlan(root, 750)
		cancelHandle := service.call(http.MethodPost, "/v1/local/workflows",
			map[string]any{"plan": cancelPlan}, "Idempotency-Key", "workflow-cancel")
		cancelID := fmt.Sprint(cancelHandle.json()["workflow_id"])
		before, ok := waitWorkflowActive(service, cancelID, 30*time.Second)
		if !ok {
			return
		}
		beforeChildren, _ := workflowChildren(before["steps"].([]any))
		canceled := service.call(http.MethodPost, "/v1/local/workflows/"+cancelID+"/cancel", nil)
		check("cancellation fact accepted", canceled.Status == http.StatusAccepted ||
			canceled.Status == http.StatusOK, canceled.brief())
		final, ok := waitWorkflow(service, cancelID, "canceled", 30*time.Second)
		if !ok {
			return
		}
		afterChildren, _ := workflowChildren(final["steps"].([]any))
		check("no child was minted after durable cancellation",
			len(afterChildren) == len(beforeChildren), fmt.Sprintf("children %d -> %d",
				len(beforeChildren), len(afterChildren)))
		return
	}

	head("kill the LocalService with a workflow child active")
	handle := service.call(http.MethodPost, "/v1/local/workflows",
		map[string]any{"plan": plan}, "Idempotency-Key", "workflow-crash")
	workflowID := fmt.Sprint(handle.json()["workflow_id"])
	pre, activeWorker, activeChild, ok := waitWorkflowProgress(service, root, workflowID, 4, 60*time.Second)
	if !ok {
		return
	}
	preChildren, _ := workflowChildren(pre["steps"].([]any))
	preDigests := workflowOutputDigests(pre["steps"].([]any))
	service.kill9()
	if !check("record owner was killed without draining its accepted child",
		!service.alive() && activeWorker.PID > 0 && alivePID(activeWorker.PID),
		fmt.Sprintf("%s · worker %s pid %d survived", workflowID,
			activeWorker.InstanceID, activeWorker.PID)) {
		return
	}
	service = startService(root, port, false)
	restartLog, _ := os.ReadFile(filepath.Join(root, "driver-service.log"))
	check("restart fenced the exact orphan worker before replay",
		strings.Contains(string(restartLog), "orphan worker "+activeWorker.InstanceID) &&
			strings.Contains(string(restartLog), fmt.Sprintf("pid %d", activeWorker.PID)) &&
			strings.Contains(string(restartLog), "killed on reconcile"), activeWorker.InstanceID)
	replay := service.call(http.MethodPost, "/v1/local/workflows",
		map[string]any{"plan": plan}, "Idempotency-Key", "workflow-crash")
	check("the workflow key recovered the same workflow", replay.Status == http.StatusOK &&
		fmt.Sprint(replay.json()["workflow_id"]) == workflowID, replay.brief())
	final, ok := waitWorkflow(service, workflowID, "succeeded", 90*time.Second)
	if !ok {
		return
	}
	postChildren, _ := workflowChildren(final["steps"].([]any))
	postDigests := workflowOutputDigests(final["steps"].([]any))
	stable := true
	for ordinal, child := range preChildren {
		stable = stable && postChildren[ordinal] == child
	}
	check("committed child identities survived the owner kill", stable,
		fmt.Sprintf("%d pre-kill child id(s)", len(preChildren)))
	check("committed output digests survived the owner kill",
		equalWorkflowDigests(preDigests, postDigests),
		fmt.Sprintf("%d pre-kill completed step(s)", len(preDigests)))
	check("the accepted child replayed as a new attempt on the same request",
		workflowRequestReplayed(root, activeChild), activeChild)
	check("the recovered workflow completed exactly nine children", len(postChildren) == 9,
		fmt.Sprintf("%d children", len(postChildren)))
}

func liveWorkflowPlan(root string, delayMS int) map[string]any {
	must("setting workflow COZY_HOME", os.Setenv("COZY_HOME", root))
	cfg, problem := config.Load()
	must("workflow config", errOf(problem))
	cfg.Home = root
	store, problem := records.Open(filepath.Join(root, "records.db"))
	must("workflow records", errOf(problem))
	defer store.Close()
	placement, problem := app.NewResolver(store, cfg).ResolvePlacement(transportEndpointRef)
	must("workflow placement", errOf(problem))
	plans := map[string]string{}
	outputs := map[string][]string{}
	for _, binding := range placement.Bindings {
		id, problem := binding.PlanID()
		must("workflow binding plan", errOf(problem))
		plans[binding.Entrypoint] = id
		outputs[binding.Entrypoint] = binding.Outputs
	}
	steps := make([]any, 0, 9)
	steps = append(steps, map[string]any{
		"endpoint": transportEndpointRef, "endpoint_release_id": placement.ReleaseID,
		"entrypoint": "tile", "entrypoint_binding_plan_id": plans["tile"],
		"payload_b64": b64(map[string]any{"size": 8, "seed": 17}),
		"outputs":     outputs["tile"], "assets": []any{}, "bindings": []any{},
	})
	for ordinal := 2; ordinal <= 9; ordinal++ {
		steps = append(steps, map[string]any{
			"endpoint": transportEndpointRef, "endpoint_release_id": placement.ReleaseID,
			"entrypoint": "relay", "entrypoint_binding_plan_id": plans["relay"],
			"payload_b64": b64(map[string]any{"image": "", "delay_ms": delayMS}),
			"outputs":     outputs["relay"], "assets": []any{},
			"bindings": []any{map[string]any{"field_path": "image",
				"prior_step": ordinal - 1, "output_name": "image", "expected_media_kind": "image"}},
		})
	}
	return map[string]any{"format": "cozy.workflow.Plan/1",
		"creative_plan_digest": workflowCreativeDigest, "steps": steps}
}

func b64(value any) string {
	data, err := json.Marshal(value)
	must("workflow payload", err)
	return base64.StdEncoding.EncodeToString(data)
}

func waitWorkflow(service *liveService, id, want string, timeout time.Duration) (map[string]any, bool) {
	deadline := time.Now().Add(timeout)
	var last map[string]any
	var lastResponse reply
	for time.Now().Before(deadline) {
		response := service.call(http.MethodGet, "/v1/local/workflows/"+id, nil)
		lastResponse = response
		if response.Status == http.StatusOK {
			last = response.json()
			if fmt.Sprint(last["status"]) == want {
				return last, true
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	check("workflow "+id+" reached "+want, false, workflowWaitDetail(last, lastResponse))
	return last, false
}

func waitWorkflowActive(service *liveService, id string, timeout time.Duration) (map[string]any, bool) {
	deadline := time.Now().Add(timeout)
	var last map[string]any
	var lastResponse reply
	for time.Now().Before(deadline) {
		response := service.call(http.MethodGet, "/v1/local/workflows/"+id, nil)
		lastResponse = response
		if response.Status == http.StatusOK {
			last = response.json()
			steps, _ := last["steps"].([]any)
			for _, raw := range steps {
				row, _ := raw.(map[string]any)
				if row["status"] == "in_progress" {
					return last, true
				}
			}
		}
		time.Sleep(25 * time.Millisecond)
	}
	check("workflow "+id+" reached an active child", false, workflowWaitDetail(last, lastResponse))
	return last, false
}

func waitWorkflowProgress(service *liveService, root, id string, completed int,
	timeout time.Duration) (map[string]any, records.WorkerProcess, string, bool) {
	deadline := time.Now().Add(timeout)
	var last map[string]any
	var lastResponse reply
	for time.Now().Before(deadline) {
		response := service.call(http.MethodGet, "/v1/local/workflows/"+id, nil)
		lastResponse = response
		if response.Status == http.StatusOK {
			last = response.json()
			steps, _ := last["steps"].([]any)
			done, active := 0, false
			for _, raw := range steps {
				row, _ := raw.(map[string]any)
				done += map[bool]int{true: 1}[row["status"] == "completed"]
				active = active || row["status"] == "in_progress"
			}
			if done >= completed && active {
				if worker, child, accepted := acceptedWorkflowWorker(root, steps); accepted {
					return last, worker, child, true
				}
			}
		}
		time.Sleep(25 * time.Millisecond)
	}
	check(fmt.Sprintf("workflow %s completed %d children with another active", id, completed),
		false, workflowWaitDetail(last, lastResponse))
	return last, records.WorkerProcess{}, "", false
}

func workflowWaitDetail(last map[string]any, response reply) string {
	if last == nil {
		return "deadline · no successful status response · " + response.brief()
	}
	return fmt.Sprintf("deadline · last status %v · %s", last["status"], response.brief())
}

func acceptedWorkflowWorker(root string, steps []any) (records.WorkerProcess, string, bool) {
	store, problem := records.Open(filepath.Join(root, "records.db"))
	if problem != nil {
		return records.WorkerProcess{}, "", false
	}
	defer store.Close()
	workers, problem := store.LiveWorkers()
	if problem != nil {
		return records.WorkerProcess{}, "", false
	}
	for _, raw := range steps {
		row, _ := raw.(map[string]any)
		if row["status"] != "in_progress" {
			continue
		}
		child := fmt.Sprint(row["child_request_id"])
		attempts, problem := store.Attempts(child)
		if problem != nil || len(attempts) == 0 || attempts[len(attempts)-1].State != "accepted" {
			continue
		}
		instance := attempts[len(attempts)-1].InstanceID
		for _, worker := range workers {
			if worker.InstanceID == instance && worker.PID > 0 && worker.Birth != "" {
				return worker, child, true
			}
		}
	}
	return records.WorkerProcess{}, "", false
}

func workflowRequestReplayed(root, child string) bool {
	store, problem := records.Open(filepath.Join(root, "records.db"))
	if problem != nil {
		return false
	}
	defer store.Close()
	attempts, problem := store.Attempts(child)
	if problem != nil || len(attempts) != 2 {
		return false
	}
	return attempts[0].State == "closed" && attempts[0].TerminalStatus == "ABANDONED" &&
		attempts[0].TerminalCause == "EXECUTOR_INVALIDATED" &&
		attempts[1].State == "closed" && attempts[1].TerminalStatus == "SUCCEEDED"
}

func workflowChildren(steps []any) (map[int]string, map[string]bool) {
	children, digests := map[int]string{}, map[string]bool{}
	for _, raw := range steps {
		row, _ := raw.(map[string]any)
		ordinal := int(asFloat(row["ordinal"]))
		if child := fmt.Sprint(row["child_request_id"]); child != "" && child != "<nil>" {
			children[ordinal] = child
		}
		outputs, _ := row["outputs"].([]any)
		for _, output := range outputs {
			entry, _ := output.(map[string]any)
			if digest := fmt.Sprint(entry["digest"]); digest != "" {
				digests[digest] = true
			}
		}
	}
	return children, digests
}

func workflowOutputDigests(steps []any) map[int]map[string]string {
	out := map[int]map[string]string{}
	for _, raw := range steps {
		row, _ := raw.(map[string]any)
		ordinal := int(asFloat(row["ordinal"]))
		outputs, _ := row["outputs"].([]any)
		for _, rawOutput := range outputs {
			output, _ := rawOutput.(map[string]any)
			name, digest := fmt.Sprint(output["output_id"]), fmt.Sprint(output["digest"])
			if name == "" || digest == "" || name == "<nil>" || digest == "<nil>" {
				continue
			}
			if out[ordinal] == nil {
				out[ordinal] = map[string]string{}
			}
			out[ordinal][name] = digest
		}
	}
	return out
}

func equalWorkflowDigests(before, after map[int]map[string]string) bool {
	for ordinal, outputs := range before {
		for name, digest := range outputs {
			if after[ordinal][name] != digest {
				return false
			}
		}
	}
	return true
}

func mapKeys(values map[string]bool) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	return keys
}
