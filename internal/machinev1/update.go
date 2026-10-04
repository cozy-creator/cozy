package machinev1

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"time"

	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Member is one distribution of a software cohort: a published Version, or a Wheel file name
// whose bytes were sent with Write as Digest ("sha256:<hex>").
type Member struct {
	Version string
	Wheel   string
	Digest  string
	Length  uint64
}

// Cohort is the software an update installs. Agent is "bundled" (a Runtime wheel that
// bundles a machine also replaces the machine's binary) or "explicit" (keep it).
type Cohort struct {
	Runtime, TensorFS *Member
	Agent             string
}

func (c Cohort) spec() (*pb.RunSpec, error) {
	payload := map[string]string{}
	var inputs []*pb.InputFile
	for field, member := range map[string]*Member{"runtime": c.Runtime, "tensorfs": c.TensorFS} {
		switch {
		case member == nil:
		case member.Wheel != "":
			payload[field] = member.Wheel
			inputs = append(inputs, &pb.InputFile{Field: field, Digest: member.Digest, Length: member.Length})
		default:
			payload[field] = member.Version
		}
	}
	if len(payload) == 0 {
		return nil, errors.New("an update names a Runtime or TensorFS")
	}
	if c.Agent != "" {
		payload["agent"] = c.Agent
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	return &pb.RunSpec{Kind: pb.RunKind_RUN_KIND_UPDATE, Payload: raw, Inputs: inputs}, nil
}

// Update submits update id (the same id attaches and never runs twice) and follows its log to
// the outcome, calling each with every event. The update restarts the machine's service, so
// a lost stream is attached again from the last event seen for as long as ctx allows.
func (c *Client) Update(ctx context.Context, id string, cohort Cohort, each func(*pb.RunEvent)) (*pb.Outcome, error) {
	spec, err := cohort.spec()
	if err != nil {
		return nil, err
	}
	var after uint64
	for {
		stream, err := c.Run(ctx, id, after, spec)
		for err == nil {
			var event *pb.RunEvent
			event, err = stream.Recv()
			if err != nil {
				break
			}
			// The submission is accepted once the machine answers; later attaches carry no spec.
			spec = nil
			after = max(after, event.GetSequence())
			if each != nil {
				each(event)
			}
			if outcome := event.GetOutcome(); outcome != nil {
				return outcome, nil
			}
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if !errors.Is(err, io.EOF) && status.Code(err) != codes.Unavailable && !expiredCap(err) {
			return nil, err
		}
		// The service is restarting onto the update: wait for it to answer again.
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
	}
}
