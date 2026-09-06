package main

import (
	"context"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type fakePreparation struct {
	pb.UnimplementedRuntimePreparationServer
}

func (*fakePreparation) ProtocolInfo(context.Context, *pb.ProtocolInfoRequest) (*pb.ProtocolInfoResult, error) {
	return &pb.ProtocolInfoResult{WireMinor: pb.WireMinor, MinimumWireMinor: pb.MinCompatibleWireMinor}, nil
}
