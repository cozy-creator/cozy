package machinev1

import (
	"context"
	"errors"
	"fmt"
	"io"

	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// Write sends one content-addressed object (digest "sha256:<hex>", length bytes read from r)
// and returns once the machine holds all of it, for this signer. An earlier partial attempt
// resumes where the machine's copy ends; an object it already holds sends no bytes.
func (c *Client) Write(ctx context.Context, digest string, length uint64, r io.ReaderAt) error {
	for {
		held, err := c.write(ctx, &pb.WriteFrame{Digest: digest, Length: length}, nil)
		for err == nil && held < length {
			var next uint64
			next, err = c.write(ctx, &pb.WriteFrame{Digest: digest, Length: length, Offset: held},
				io.NewSectionReader(r, int64(held), int64(length-held)))
			if err == nil && next <= held {
				err = fmt.Errorf("the machine holds %d of %d bytes after a write", next, length)
			}
			held = next
		}
		if !expiredCap(err) || ctx.Err() != nil {
			return err
		}
	}
}

func (c *Client) write(ctx context.Context, first *pb.WriteFrame, body io.Reader) (uint64, error) {
	stream, err := c.Machine.Write(ctx)
	if err != nil {
		return 0, err
	}
	if err := stream.Send(first); err != nil {
		return 0, err
	}
	if body != nil {
		buffer := make([]byte, 1<<20)
		for {
			n, err := body.Read(buffer)
			if n > 0 {
				if err := stream.Send(&pb.WriteFrame{Data: buffer[:n]}); err != nil {
					// The machine's refusal, if it sent one, is the answer.
					if _, refusal := stream.CloseAndRecv(); refusal != nil {
						return 0, refusal
					}
					return 0, err
				}
			}
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				return 0, err
			}
		}
	}
	result, err := stream.CloseAndRecv()
	return result.GetHeld(), err
}
