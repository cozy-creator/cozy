// Reconcile only an explicit private backup. No daemon, scheduler, or provider.
package main

import (
	"encoding/json"
	"fmt"
	"github.com/cozy-creator/cozy/internal/records"
	"os"
)

func main() {
	if len(os.Args) != 2 {
		panic("provide one private backup database path")
	}
	store, problem := records.OpenForDaemon(os.Args[1], "")
	if problem != nil {
		panic(problem.Message)
	}
	defer store.Close()
	var result []map[string]any
	for _, id := range []string{"job-08d3fda1f241758d02aee627", "job-dc1816b74a4f4c446fce5205"} {
		row, problem := store.RequestByReference(id)
		if problem != nil {
			panic(problem.Message)
		}
		ended, problem := store.TerminalEventAt(id)
		if problem != nil {
			panic(problem.Message)
		}
		lost, problem := store.MachineExecutionLost(id)
		if problem != nil {
			panic(problem.Message)
		}
		owed, problem := store.MachineExecutionOwesWork(id)
		if problem != nil {
			panic(problem.Message)
		}
		cause, _, message, problem := store.SettledFailure(id)
		if problem != nil {
			panic(problem.Message)
		}
		result = append(result, map[string]any{"run": row.Number, "id": id, "state": row.State, "retained": row.RetainWork, "created_at": row.CreatedAt, "ended_at": ended, "machine_lost": lost, "machine_owed": owed, "error_type": cause, "error": message})
	}
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		panic(err)
	}
	fmt.Println(string(data))
}
