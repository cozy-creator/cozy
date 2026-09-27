// Package hostgpu reads local NVIDIA compatibility facts without importing
// Torch or opening a CUDA context.
package hostgpu

import (
	"context"
	"encoding/csv"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
)

type GPU struct {
	Index             int    `json:"index"`
	Model             string `json:"model"`
	VRAMFreeBytes     int64  `json:"vram_free_bytes"`
	VRAMTotalBytes    int64  `json:"vram_total_bytes"`
	DriverVersion     string `json:"driver_version"`
	DriverCUDAVersion string `json:"driver_cuda_version"`
	ComputeCapability string `json:"compute_capability"`
	SM                string `json:"sm"`
}

type Inventory struct {
	GPUs       []GPU
	Diagnostic string
}

var driverCUDA = regexp.MustCompile(`CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)`)

func Probe(cfg config.Config) Inventory {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=index,name,memory.free,memory.total,driver_version,compute_cap",
		"--format=csv,noheader,nounits")
	cmd.Env = cfg.Tool()
	raw, err := cmd.Output()
	if err != nil {
		var missing *exec.Error
		if errors.As(err, &missing) {
			return Inventory{}
		}
		return Inventory{Diagnostic: "nvidia-smi could not read the local NVIDIA GPUs"}
	}
	gpus, err := parseGPUs(string(raw))
	if err != nil {
		return Inventory{Diagnostic: err.Error()}
	}
	version := readDriverCUDA(ctx, cfg)
	for i := range gpus {
		gpus[i].DriverCUDAVersion = version
	}
	return Inventory{GPUs: gpus}
}

func parseGPUs(raw string) ([]GPU, error) {
	reader := csv.NewReader(strings.NewReader(raw))
	reader.FieldsPerRecord = -1
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi returned unreadable GPU data")
	}
	// A row this build cannot read is skipped, not a reason to discard every GPU; a
	// device that reports no compute capability keeps its model and memory.
	gpus := make([]GPU, 0, len(records))
	for _, record := range records {
		if len(record) < 6 {
			continue
		}
		index, indexErr := strconv.Atoi(strings.TrimSpace(record[0]))
		freeMiB, freeErr := strconv.ParseInt(strings.TrimSpace(record[2]), 10, 64)
		totalMiB, totalErr := strconv.ParseInt(strings.TrimSpace(record[3]), 10, 64)
		if indexErr != nil || freeErr != nil || totalErr != nil || freeMiB < 0 || totalMiB <= 0 {
			continue
		}
		gpu := GPU{
			Index: index, Model: strings.TrimSpace(record[1]),
			VRAMFreeBytes: freeMiB << 20, VRAMTotalBytes: totalMiB << 20,
			DriverVersion: strings.TrimSpace(record[4]),
		}
		if compute := strings.Trim(strings.TrimSpace(record[5]), "[]"); compute != "" && !strings.EqualFold(compute, "N/A") {
			gpu.ComputeCapability, gpu.SM = compute, "sm_"+strings.ReplaceAll(compute, ".", "")
		}
		gpus = append(gpus, gpu)
	}
	if len(gpus) == 0 && strings.TrimSpace(raw) != "" {
		return nil, fmt.Errorf("nvidia-smi returned no readable GPU record")
	}
	return gpus, nil
}

func readDriverCUDA(ctx context.Context, cfg config.Config) string {
	cmd := exec.CommandContext(ctx, "nvidia-smi")
	cmd.Env = cfg.Tool()
	raw, err := cmd.Output()
	if err != nil {
		return ""
	}
	match := driverCUDA.FindSubmatch(raw)
	if match == nil {
		return ""
	}
	return string(match[1])
}
