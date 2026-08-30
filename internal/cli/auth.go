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
	hctx, cancel := hub.Context()
	session, problem := manager.Authenticate(hctx)
	cancel()
	if problem == nil {
		return emitAuthSession(ctx, session, false)
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
	code, err := bufio.NewReader(os.Stdin).ReadString('\n') //cozy:stdin-value the email code is read interactively, never from argv
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
	return emitAuthSession(ctx, session, true)
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

func emitAuthSession(ctx *Context, session accountauth.Session, enrolled bool) *exit.Error {
	hctx, cancel := hub.Context()
	_, problem := client(ctx).WithToken(session.AccessToken, "machine login").CurrentUser(hctx)
	cancel()
	if problem != nil {
		return problem
	}
	status := "authenticated"
	if enrolled {
		status = "registered"
	}
	return emit(ctx, output.Record{Fields: []output.Field{
		{K: "status", V: status},
		{K: "email", V: session.Email},
		{K: "machine", V: session.DeviceKeyID},
		{K: "hub", V: ctx.Cfg.HubURL},
		{K: "access_expires", V: session.ExpiresAt},
	}})
}
