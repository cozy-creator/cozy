package cli

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
)

// isOperation is a journaled number that is not a run: a download, installation, Runtime
// update or output upload on a machine. Model uploads and local transfers are runs of their kind.
func isOperation(life api.Lifecycle) bool {
	return life.Kind == "download" || life.Kind == "install" || life.Kind == "update" || life.Kind == "upload"
}

// journalKind is the KIND a list shows.
func journalKind(life api.Lifecycle) string {
	if life.Journal == "" {
		return "run"
	}
	return life.Journal
}

// lifeTarget is what a numbered row acts on.
func lifeTarget(life api.Lifecycle) string {
	if life.Target != "" {
		return life.Target
	}
	return strings.Trim(life.Package+"/"+life.Function, "/")
}

// modelRows renders each model's bytes and the in-flight tail: what is left, how fast it
// moves, and how long since the machine last reported.
type modelRows []orchestrator.ModelDownloadProgress

func (rows modelRows) lines() []string {
	out := make([]string, 0, len(rows))
	for _, row := range rows {
		name := row.Model
		if row.Release != "" {
			name += "@" + row.Release
		}
		if row.Lane != "" {
			name += "/" + row.Lane
		}
		parts := []string{output.Bytes(int64(row.Moved))}
		if row.Total > 0 {
			parts[0] += " / " + output.Bytes(int64(row.Total))
			if row.Moved >= row.Total {
				parts = append(parts, "done")
			} else {
				parts = append(parts, output.Bytes(int64(row.Total-row.Moved))+" left")
			}
		}
		if row.Rate > 0 {
			parts = append(parts, output.Bytes(int64(row.Rate))+"/s")
		}
		if row.RemainingMS != nil {
			parts = append(parts, "~"+shortDuration(time.Duration(*row.RemainingMS)*time.Millisecond))
		}
		if row.SampleAgeMS > 0 && time.Duration(row.SampleAgeMS)*time.Millisecond > orchestrator.PreparationRateMaxAge {
			parts = append(parts, "no report for "+shortDuration(time.Duration(row.SampleAgeMS)*time.Millisecond))
		}
		out = append(out, name+": "+strings.Join(parts, " · "))
	}
	return out
}

func (rows modelRows) Human() string { return strings.Join(rows.lines(), "; ") }

// operationStatus is the lifecycle status, worded for a person and exact for a machine.
type operationStatus api.Lifecycle

func (s operationStatus) Human() string {
	if s.Status == "canceled" {
		return humanCancellationStatus(s.CanceledBy)
	}
	return runStatus(s.Status)
}

func (s operationStatus) MarshalJSON() ([]byte, error) { return json.Marshal(s.Status) }

// operationRecord is one operation's number, what it lands and where, and what it does now.
func operationRecord(life api.Lifecycle) output.Record {
	progress := progressValue(life)
	fields := []output.Field{
		{K: "number", V: life.Number}, {K: "kind", V: life.Kind}, {K: "target", V: lifeTarget(life)},
		{K: "machine", V: life.Machine}, {K: "status", V: operationStatus(life)},
		{K: "progress", V: progress}, {K: "id", V: life.RequestID}, {K: "rental_id", V: life.RentalID},
		{K: "models", V: modelRows(life.PhaseModels)}, {K: "created_at", V: life.CreatedAt},
	}
	defaults := []string{"number", "kind", "target", "machine", "status"}
	if progress != "" && progress != "-" {
		defaults = append(defaults, "progress")
	}
	if len(life.PhaseModels) > 0 {
		defaults = append(defaults, "models")
	}
	if life.WaitingFor != nil {
		fields = append(fields, output.Field{K: "waiting_for", V: life.WaitingFor.String()})
		defaults = append(defaults, "waiting_for")
	}
	if life.Error != "" {
		fields = append(fields, output.Field{K: "error_code", V: life.ErrorCode}, output.Field{K: "error", V: life.Error})
		defaults = append(defaults, "error")
	}
	if life.Upload != nil && life.Upload.Checkpoint != "" {
		fields = append(fields, output.Field{K: "checkpoint", V: life.Upload.Checkpoint})
		defaults = append(defaults, "checkpoint")
	}
	record := compactRecord(fields, defaults...)
	number := strconv.FormatInt(life.Number, 10)
	if life.Upload != nil && life.Status == "completed" {
		record.Notes = []string{"the checkpoint is private to the repository's owners until a release publishes it"}
		record.Next = []string{"cozy model publish " + life.Upload.Destination + " --release <label> --lane " + life.Upload.Output + "=" + life.Upload.Checkpoint}
	}
	switch {
	case life.Status == "queued" || life.Status == "in_progress":
		record.Next = []string{"cozy run watch " + number, "cozy run cancel " + number}
		if life.Kind == "update" && life.Status == "in_progress" {
			record.Next = record.Next[:1]
		}
	case life.Status == "failed" && life.ErrorCode == "rental.unusable":
		record.Next = []string{"cozy rental update " + life.Machine, "cozy rental end " + life.Machine}
	}
	return record
}

// watchOperation follows an operation to its end, printing what changes on stderr; stdout
// keeps one record. A signal only detaches: the daemon keeps the operation.
func watchOperation(ctx *Context, client *localapi.Client, life api.Lifecycle) *exit.Error {
	interrupt, restore, _, problem := liveSignals(ctx, nil)
	if problem != nil {
		return problem
	}
	defer restore()
	reference := strconv.FormatInt(life.Number, 10)
	began := time.Now()
	last := ""
	tick := time.NewTicker(time.Second)
	defer tick.Stop()
	for life.Status == "queued" || life.Status == "in_progress" {
		if line := operationLine(life); line != last {
			_ = output.Progress(ctx.Err, line)
			last = line
		}
		select {
		case <-interrupt:
			_ = output.Progress(ctx.Err, "detached from #"+reference+"; it continues")
			return emit(ctx, operationRecord(life))
		case <-tick.C:
		}
		if life, problem = client.Request(reference); problem != nil {
			return problem
		}
	}
	settled := operationRecord(life)
	elapsed := output.Field{K: "elapsed", V: time.Since(began).Round(time.Second).String()}
	settled.Fields, settled.AllFields = append(settled.Fields, elapsed), append(settled.AllFields, elapsed)
	if err := emit(ctx, settled); err != nil {
		return err
	}
	switch life.Status {
	case "failed":
		return exit.Named(exit.Failed, life.ErrorCode, "#%d %s failed: %s", life.Number, life.Kind, life.Error)
	case "canceled":
		return exit.Named(exit.Canceled, "operation.canceled", "#%d %s was %s", life.Number, life.Kind, operationStatus(life).Human())
	}
	return nil
}

func operationLine(life api.Lifecycle) string {
	parts := []string{fmt.Sprintf("#%d %s %s on %s", life.Number, life.Kind, lifeTarget(life), life.Machine), operationStatus(life).Human()}
	if progress := progressValue(life); progress != "" && progress != "-" {
		parts = append(parts, progress)
	}
	if life.WaitReason != "" {
		parts = append(parts, life.WaitReason)
	}
	return strings.Join(parts, " · ")
}

// cancelOperation withdraws a queued operation, detaches its owner from a download or
// installation in flight, and refuses an update that has started.
func cancelOperation(ctx *Context, client *localapi.Client, before api.Lifecycle) *exit.Error {
	reference := strconv.FormatInt(before.Number, 10)
	after := before
	if before.Status == "queued" || before.Status == "in_progress" {
		if problem := client.Cancel(reference, "cozy run cancel"); problem != nil {
			return problem
		}
		var problem *exit.Error
		if after, problem = client.Request(reference); problem != nil {
			return problem
		}
	}
	record := operationRecord(after)
	changed := after.Status != before.Status
	record.Fields = append(record.Fields, output.Field{K: "changed", V: changed})
	record.AllFields = append(record.AllFields, output.Field{K: "changed", V: changed})
	if changed && before.Status == "in_progress" {
		record.Notes = append(record.Notes, "the machine stops the transfer once no other work needs it; landed bytes stay, so the same download resumes")
	}
	return emit(ctx, record)
}
