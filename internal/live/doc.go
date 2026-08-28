// Package live is this repository's verification suite. It has no non-test source: every
// file beside this one is a `go test` file that starts the REAL system — the real local
// orchestrator and record owner, the real `cozy up` process, and `internal/live/fakeworker`, a
// second independent implementation of the worker protocol — and observes what it does.
// Nothing here is a mock.
package live
