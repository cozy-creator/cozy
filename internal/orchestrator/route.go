package orchestrator

// WHICH LANE AN OFFER GOES TO (residency-aware-routing.md §3.1, cl-092). Routing is one
// ordering over facts the worker reports, never a model of its device:
//
//	score(l) = held(l) + cost(l)
//	pick     = argmin score over candidates with room(l) > 0;
//	           ties → p ∈ resident(l), then the local worker, then lowest (instance_id, lane_id)
//
// held(l) is the attempt-equivalents already ahead on the lane: every attempt the worker
// holds on it (QUEUED, RUNNING, outcome pending ack) plus this owner's own offers the worker
// has not answered yet. cost(l) is the load the attempt pays on arrival, priced per BINDING:
// 0 when the live executor already holds the entrypoint's construction
// (`loaded_binding_digests`), else 1 on an idle lane and 2 when the lane holds another
// tenant. Ties break on the known load time, the request's model bytes at 20 GB/s. A
// worker older than minor 61 reports no loaded set; its DISPATCHABLE placement holds its
// one construction, priced by the lane's `resident_placement_ids`. The constants are
// attempt-equivalents and live here, never in config or env (#1312). Nothing here is a
// timer: `age_ms` rides the decision log for the audit and ranks nothing.
//
// WHICH WORKERS ARE ASKED is the request's permission (D6): local lanes unless
// `--rental-only`; every attached rental's lanes with `--rental`, local winning ties; a
// request PINNED to a rental (`Worker`) sees that rental alone. The pin is routing's own
// output — written by `dispatch` when the argmin is a rental, or by the capacity decision
// when no worker holds the placement (`selectOrStart`) — never a submission's guess.
