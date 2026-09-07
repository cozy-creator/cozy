// Explicit operator bridge for attaching already-produced checkpoint measurements.
// Defaults to verification only; it does not run inference or create a report.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/assessment"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

func main() {
	request := flag.String("request", "", "retained producer job id")
	report := flag.String("report", "", "canonical producer report file")
	evaluator := flag.String("evaluator", "cozy-eval", "installed canonical evaluator command")
	attach := flag.Bool("attach", false, "attach verified report and read back exact bytes")
	flag.Parse()
	if *request == "" || *report == "" {
		fatal("--request and --report are required")
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	cfg, problem := config.Load()
	if problem != nil {
		fatal(problem.Message)
	}
	f, err := os.Open(*report)
	if err != nil {
		fatal("report is unreadable")
	}
	raw, err := io.ReadAll(io.LimitReader(f, assessment.MaxBytes+1))
	f.Close()
	if err != nil || len(raw) > assessment.MaxBytes {
		fatal("report exceeds the canonical inspection bound")
	}
	command := exec.CommandContext(ctx, *evaluator, "report", "inspect")
	command.Env = cfg.Tool("PYTHONNOUSERSITE=1")
	input, err := command.StdinPipe()
	if err != nil {
		fatal("cannot open evaluator input")
	}
	output, err := command.StdoutPipe()
	if err != nil {
		fatal("cannot open evaluator output")
	}
	command.Stderr = io.Discard
	if command.Start() != nil {
		fatal("canonical evaluator unavailable")
	}
	go func() { _, _ = input.Write(raw); input.Close() }()
	inspected, err := io.ReadAll(io.LimitReader(output, assessment.MaxBytes+1))
	if err != nil || len(inspected) > assessment.MaxBytes {
		command.Process.Kill()
		command.Wait()
		fatal("canonical evaluator output exceeds its bound")
	}
	if command.Wait() != nil {
		fatal("canonical evaluator refused the report")
	}
	st, problem := records.Open(filepath.Join(cfg.Home, "creator.sqlite"))
	if problem != nil {
		fatal(problem.Message)
	}
	defer st.Close()
	ref, info, problem := assessment.VerifyProducer(st, *request, raw, inspected)
	if problem != nil {
		fatal(problem.Message)
	}
	if *attach {
		client := hub.New(cfg, "cozy-producer-assessment").WithTokenSource(accountauth.New(cfg))
		account, problem := client.CurrentAccount(ctx)
		if problem != nil {
			fatal(problem.Message)
		}
		if account.Name != ref.Org {
			fatal("producer destination is not the authenticated account")
		}
		if problem = client.AttachAssessment(ctx, ref, info.Subject.Candidate, info.Report.Digest, raw); problem != nil {
			fatal(problem.Message)
		}
	}
	_ = json.NewEncoder(os.Stdout).Encode(map[string]any{"verified": true, "attached": *attach, "model": ref.String(), "checkpoint": info.Subject.Candidate, "report": info.Report.Digest, "scope": "publisher_assessment", "verdict": info.Verdict})
}
func fatal(message string) { fmt.Fprintln(os.Stderr, message); os.Exit(1) }
