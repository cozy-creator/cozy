package cli

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
)

func handleAuthLogin(ctx *Context) *exit.Error {
	manager := accountauth.New(ctx.Cfg)
	input := bufio.NewReader(os.Stdin) //cozy:stdin-value login owns the two interactive values
	hctx, cancel := hub.Context()
	session, problem := manager.Authenticate(hctx)
	cancel()
	if problem == nil {
		return emitAuthSession(ctx, session, "authenticated", input)
	}
	if !canEnroll(problem) {
		return problem
	}

	hctx, cancel = hub.Context()
	enrollment, expires, problem := manager.BeginEnrollment(hctx, ctx.Inv.Args[0])
	cancel()
	if problem != nil {
		return problem
	}
	fmt.Fprintf(ctx.Err, "A verification code was sent to %s (expires %s).\nCode: ",
		strings.TrimSpace(ctx.Inv.Args[0]), expires.Local().Format("15:04:05 MST"))
	code, err := input.ReadString('\n') //cozy:stdin-value the email code is read interactively, never from argv
	if err != nil && strings.TrimSpace(code) == "" {
		return exit.Named(exit.Credential, "auth.code_unreadable",
			"the email verification code could not be read: %s", err)
	}
	hctx, cancel = hub.Context()
	session, problem = manager.FinishEnrollment(hctx, enrollment, code)
	cancel()
	if problem != nil {
		return problem
	}
	return emitAuthSession(ctx, session, "registered", input)
}

func handleAuthStatus(ctx *Context) *exit.Error {
	manager := accountauth.New(ctx.Cfg)
	hctx, cancel := hub.Context()
	session, problem := manager.Authenticate(hctx)
	cancel()
	if problem != nil {
		if canEnroll(problem) {
			record := compactRecord([]output.Field{
				{K: "status", V: "not logged in"},
				{K: "hub", V: ctx.Cfg.HubURL},
			}, "status")
			record.Next = []string{"cozy auth login <email>"}
			return emit(ctx, record)
		}
		return problem
	}
	return emitAuthSession(ctx, session, "logged in", nil, "cozy auth login <email>")
}

func canEnroll(problem *exit.Error) bool {
	if problem == nil {
		return false
	}
	switch problem.ErrName() {
	case "auth.machine_key_missing", "auth.machine_key_invalid", "auth.machine_key_unreadable",
		"invalid_credentials":
		return true
	}
	return false
}

func emitAuthSession(ctx *Context, session accountauth.Session, status string,
	input *bufio.Reader, next ...string,
) *exit.Error {
	hctx, cancel := hub.Context()
	c := client(ctx).WithToken(session.AccessToken, "machine login")
	account, problem := c.CurrentAccount(hctx)
	cancel()
	if problem != nil && problem.ErrName() == "account.name_required" && input != nil {
		fmt.Fprint(ctx.Err, "Tensorhub account name: ")
		name, err := input.ReadString('\n') //cozy:stdin-value the account name is an interactive value, never argv
		name = strings.TrimSpace(name)
		if err != nil && name == "" {
			return exit.Named(exit.Credential, "account.name_unreadable",
				"the Tensorhub account name could not be read: %s", err)
		}
		if name == "" {
			return exit.Usagef("Tensorhub account name is required")
		}
		hctx, cancel = hub.Context()
		account, problem = c.RegisterAccount(hctx, name)
		cancel()
	}
	if problem != nil && problem.ErrName() == "account.name_required" && input == nil {
		account.Name = "not registered"
		problem = nil
	}
	if problem != nil {
		return problem
	}
	record := compactRecord([]output.Field{
		{K: "status", V: status},
		{K: "email", V: session.Email},
		{K: "account", V: account.Name},
		{K: "machine", V: session.DeviceKeyID},
		{K: "hub", V: ctx.Cfg.HubURL},
		{K: "access_expires", V: session.ExpiresAt},
	}, "status", "email", "account")
	record.Next = next
	return emit(ctx, record)
}
