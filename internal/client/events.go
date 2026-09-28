package client

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
)

// The SSE consumer, written against `docs/client-contract.md` §5 and modelled on the
// reference client (cozy.art's `startSSE`). Three rules, and the CLI depends on all three:
//
//   - CURSOR RESUME. Every durable event carries `id:`. A stream that drops without a
//     terminal is NOT a verdict — reconnect from the last id and the server replays
//     everything after it, in order, across a daemon restart.
//   - TERMINAL-STOP. A terminal event closes the per-request stream, and that is where
//     `cozy run` gets its exit code: the request's own settled status, mapped through the
//     shared matrix. There is no polling loop and no timeout guessing what happened.
//   - LIVE FRAMES ARE NOT DURABLE. `event_id: 0` says so in the data. They are what the
//     progress line renders and are never counted as history.

// Event is one envelope off the stream.
type Event = api.Envelope

// Terminal answers whether this event settles the REQUEST. An attempt ending is not a
// request ending (decisions #360): `request.attempt_failed` is deliberately not here.
func Terminal(t string) bool {
	switch t {
	case "request.completed", "request.failed", "request.canceled":
		return true
	}
	return false
}

// Watch consumes ONE request's stream to its terminal, calling `on` for every event. It
// reconnects from its own cursor when the connection drops without a terminal, because a
// close is not a verdict; `on` returning false stops the watch deliberately (SIGINT, a
// caller that has seen enough) and is reported as `stopped`.
//
// It returns the terminal event when one arrived. A nil terminal with no error means the
// caller stopped it.
func (c *Client) Watch(requestID string, from int64, on func(Event) bool) (*Event, *exit.Error) {
	return c.WatchContext(context.Background(), requestID, from, on)
}

// WatchContext is Watch with an explicit caller stop. Canceling the context stops the
// open SSE read without changing the durable request; the caller decides whether it has
// already requested server-side cancellation.
func (c *Client) WatchContext(ctx context.Context, requestID string, from int64,
	on func(Event) bool,
) (*Event, *exit.Error) {
	cursor := from
	attempts := 0
	recovering := false
	for {
		terminal, last, stopped, e := c.readStream(ctx, requestID, cursor, on)
		if last > cursor {
			cursor = last
			attempts = 0 // progress resets the budget: a long run is not a broken one
		}
		if c.reattached != nil && lostDaemon(e, recovering) {
			// The run is durable and the daemon owns it; a restarted daemon replays the
			// stream from this cursor.
			if !c.reattach(ctx) {
				return nil, nil
			}
			recovering, attempts = true, 0
			continue
		}
		if e != nil {
			return nil, e
		}
		if recovering {
			recovering = false
			c.reattached()
		}
		if terminal != nil || stopped {
			return terminal, nil
		}
		// The stream ended with no terminal. That means reconnect, and it is bounded so a
		// server that closes instantly cannot spin: three consecutive closes with no new
		// event is a daemon that is not going to answer.
		attempts++
		if attempts > 3 {
			return nil, exit.Unavailablef(
				"the event stream for %s closed %d times without a terminal", requestID, attempts).
				WithRemedy("the request is still recorded; read it with `cozy run list`").
				WithNext("cozy run list")
		}
	}
}

func (c *Client) readStream(ctx context.Context, requestID string, cursor int64, on func(Event) bool) (
	terminal *Event, last int64, stopped bool, fail *exit.Error) {
	path := "/v1/requests/" + requestID + "/events?cursor=" + strconv.FormatInt(cursor, 10)
	req, e := c.request("GET", path, nil, "Accept", "text/event-stream")
	if e != nil {
		return nil, cursor, false, e
	}
	req = req.WithContext(ctx)
	res, err := c.http.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, cursor, true, nil
		}
		return nil, cursor, false, c.unreachable(err)
	}
	defer res.Body.Close()
	if res.StatusCode >= 300 {
		data, _ := io.ReadAll(io.LimitReader(res.Body, 1<<20))
		return nil, cursor, false, Refusal(res.StatusCode, data)
	}
	last = cursor
	reader := bufio.NewReader(res.Body)
	var id, data string
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if ctx.Err() != nil {
				return nil, last, true, nil
			}
			// A transport end. Not a verdict — Watch decides whether to resume.
			return nil, last, false, nil
		}
		line = strings.TrimRight(line, "\r\n")
		switch {
		case strings.HasPrefix(line, "id: "):
			id = strings.TrimPrefix(line, "id: ")
		case strings.HasPrefix(line, "data: "):
			data = strings.TrimPrefix(line, "data: ")
		case line == "" && data != "":
			var ev Event
			if json.Unmarshal([]byte(data), &ev) == nil {
				if n, err := strconv.ParseInt(id, 10, 64); err == nil && n > last {
					last = n
				}
				keep := on(ev)
				if Terminal(ev.Type) {
					settled := ev
					return &settled, last, false, nil
				}
				if !keep {
					return nil, last, true, nil
				}
			}
			id, data = "", ""
		}
	}
}

// StreamStatus is the status a terminal event carries, for the exit mapping. The event
// TYPE is the authority — `request.completed` / `failed` / `canceled` — because it is
// what terminal-stop already keyed on.
func StreamStatus(e *Event) string {
	if e == nil {
		return ""
	}
	switch e.Type {
	case "request.completed":
		return "succeeded"
	case "request.failed":
		return "failed"
	case "request.canceled":
		return "canceled"
	}
	return ""
}
