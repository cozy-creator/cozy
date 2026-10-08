package cli

import (
	"context"
	"os"
	"os/signal"
	"strconv"
	"syscall"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/providerrelease"
)

func handleExternalRentalEnd(ctx *Context) *exit.Error {
	if !ctx.Inv.Bool("--token-stdin") {
		return exit.Usagef("external provider cleanup requires --token-stdin; supply the provider API key explicitly")
	}
	id, err := strconv.ParseInt(ctx.Inv.Value("--resource-id"), 10, 64)
	if err != nil || id <= 0 {
		return exit.Usagef("--resource-id must be a positive provider instance id")
	}
	token, problem := providerrelease.Token(os.Stdin) //cozy:stdin-value explicit provider credential, never a prompt or argv secret
	if problem != nil {
		return problem
	}
	call, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	result, problem := providerrelease.Vast(call, ctx.Inv.Value("--provider-url"), id, ctx.Inv.Value("--expected-label"), token)
	if problem != nil {
		return problem
	}
	return emit(ctx, output.Record{Fields: []output.Field{
		{K: "provider", V: result.Provider}, {K: "provider_resource_id", V: result.ResourceID},
		{K: "label", V: result.Label}, {K: "state", V: result.State}, {K: "changed", V: result.Changed},
		{K: "provider_absent", V: result.ProviderAbsent},
	}})
}
