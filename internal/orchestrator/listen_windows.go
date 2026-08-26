//go:build windows

package orchestrator

// bootstrapRequired: a loopback TCP port has no SO_PEERCRED, so the kernel cannot say
// which process dialled. The substitute (#449) is a PER-SPAWN bootstrap credential the
// launcher mints and hands to the child through its environment.
//
// It also decides the worker's LISTEN grant: Windows' AF_UNIX exists but the Python side
// of this protocol cannot dial it (asyncio has no unix sockets there), so a worker there
// binds loopback and publishes the address it bound.
const bootstrapRequired = true
