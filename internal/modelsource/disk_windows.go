//go:build windows

package modelsource

func availableBytes(string) (uint64, error) { return ^uint64(0), nil }
