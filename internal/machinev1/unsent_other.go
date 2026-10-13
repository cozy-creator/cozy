//go:build !linux && !darwin

package machinev1

import "net"

func boundUnsent(net.Conn) {}
