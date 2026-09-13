package cli

import (
	"encoding/json"
	"os"

	"github.com/cozy-creator/cozy/internal/executionowner"
	"github.com/cozy-creator/cozy/internal/exit"
)

type ExecutionOwnerCmd struct {
	Validate ExecutionOwnerValidateCmd `cmd:"" help:"Validate the Host-provided initial private capture."`
	Serve    ExecutionOwnerServeCmd    `cmd:"" help:"Own the admitted client script on this worker."`
}

type ExecutionOwnerServeCmd struct {
	CapsuleFD   int `name:"capsule-fd" required:""`
	AuthorityFD int `name:"authority-fd" required:""`
}

func (command *ExecutionOwnerServeCmd) Run(runtime *Runtime) error {
	bootstrap, problem := readExecutionBootstrap(command.CapsuleFD, command.AuthorityFD, true)
	if problem != nil {
		return problem
	}
	return serveExecutionOwner(runtime, bootstrap)
}

type ExecutionOwnerValidateCmd struct {
	CapsuleFD   int `name:"capsule-fd" required:""`
	AuthorityFD int `name:"authority-fd" required:""`
}

func (command *ExecutionOwnerValidateCmd) Run(runtime *Runtime) error {
	bootstrap, problem := readExecutionBootstrap(command.CapsuleFD, command.AuthorityFD, false)
	if problem != nil {
		return problem
	}
	return json.NewEncoder(runtime.Out).Encode(map[string]any{
		"validated": true, "capsule_digest": bootstrap.Capsule.Digest,
		"package_count": len(bootstrap.Capsule.Packages),
	})
}

func readExecutionBootstrap(capsuleFD, authorityFD int, serving bool) (*executionowner.Bootstrap, *exit.Error) {
	if capsuleFD < 3 || authorityFD < 3 || capsuleFD == authorityFD {
		return nil, exit.New(exit.Validation, "execution bootstrap requires two distinct inherited descriptors")
	}
	capsule := os.NewFile(uintptr(capsuleFD), "execution-capsule")
	authority := os.NewFile(uintptr(authorityFD), "execution-authority")
	if capsule == nil || authority == nil {
		return nil, exit.New(exit.Validation, "execution bootstrap descriptors are unavailable")
	}
	defer capsule.Close()
	defer authority.Close()
	return executionowner.ReadBootstrap(capsule, authority, serving)
}
