package packagepublish

// cl-084, Paul ruling 2026-09-01: image-owned distributions may not be pinned.
// The DECLARED specifier set (pyproject, hence Requires-Dist) must admit at
// least two distinct minor release lines, so the fleet can move between image
// builds (torch 2.13 -> 2.14) without a republish. `torch>=2.13,<3` passes;
// `==2.13.0`, `==2.13.*`, `~=2.13.0`, and `>=2.13,<2.14` are refused.
// Package-owned and registry dependencies pin freely.
//
// The evaluator is closed and semantic: candidate minor lines come from the
// specifier's own boundary versions plus fixed neighbors, and every candidate
// release is judged by PEP 440 evaluation, never by string shape. Pre, post,
// dev, and local versions never count as an admitted line on their own —
// candidates are plain final releases.

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	pep440 "github.com/aquasecurity/go-pep440-version"
	"github.com/cozy-creator/cozy/internal/exit"
)

type minorLine struct {
	epoch, major, minor uint64
}

type boundaryVersion struct {
	epoch   uint64
	release []uint64
}

var boundaryVersionPattern = regexp.MustCompile(
	`(?:===|==|!=|~=|<=|>=|<|>)\s*v?(?:([0-9]+)!)?([0-9]+(?:\.[0-9]+)*)`)

func refuseImageOwnedPin(req requirement) *exit.Error {
	if !ImageOwnedDistribution(req.name) || !req.hasSpec {
		return nil
	}
	if admitsTwoMinorLines(req.specifier, req.spec) {
		return nil
	}
	problem := exit.Named(exit.Validation, "package_publish.image_owned_pin",
		"project dependency %q confines image-owned distribution %s to fewer than two minor releases",
		req.raw, req.name)
	if example := compliantImageOwnedRange(req.name, req.spec); example != "" {
		return problem.WithRemedy(
			"declare a range admitting at least two minor releases, such as %s, so the fleet can move between image builds",
			example)
	}
	return problem.WithRemedy(
		"declare a range admitting at least two minor releases of %s so the fleet can move between image builds",
		req.name)
}

// admitsTwoMinorLines reports whether the specifier set admits final releases
// in at least two distinct (epoch, major, minor) lines.
func admitsTwoMinorLines(set pep440.Specifiers, spec string) bool {
	boundaries := boundaryVersions(spec)
	lines := map[minorLine]bool{}
	epochs := map[uint64]bool{0: true}
	for _, boundary := range boundaries {
		epochs[boundary.epoch] = true
	}
	for epoch := range epochs {
		lines[minorLine{epoch, 0, 0}] = true
		lines[minorLine{epoch, 0, 1}] = true
	}
	for _, boundary := range boundaries {
		major, minor := boundary.majorMinor()
		lines[minorLine{boundary.epoch, major, minor}] = true
		lines[minorLine{boundary.epoch, major, minor + 1}] = true
		lines[minorLine{boundary.epoch, major + 1, 0}] = true
		if minor > 0 {
			lines[minorLine{boundary.epoch, major, minor - 1}] = true
		}
		if major > 0 {
			lines[minorLine{boundary.epoch, major - 1, 0}] = true
		}
	}
	admitted := 0
	for line := range lines {
		if lineAdmitsRelease(set, line, boundaries) {
			admitted++
			if admitted >= 2 {
				return true
			}
		}
	}
	return false
}

// lineAdmitsRelease asks whether any plain final release inside one minor line
// satisfies the whole specifier set. Patch candidates are 0, 1, and each
// boundary patch in this exact line plus its successor, so bounds like
// `>2.13.5` are answered by 2.13.6, not by a string comparison.
func lineAdmitsRelease(set pep440.Specifiers, line minorLine, boundaries []boundaryVersion) bool {
	patches := map[uint64]bool{0: true, 1: true}
	for _, boundary := range boundaries {
		major, minor := boundary.majorMinor()
		if boundary.epoch != line.epoch || major != line.major || minor != line.minor {
			continue
		}
		patch := uint64(0)
		if len(boundary.release) > 2 {
			patch = boundary.release[2]
		}
		patches[patch] = true
		patches[patch+1] = true
	}
	for patch := range patches {
		candidate := fmt.Sprintf("%d.%d.%d", line.major, line.minor, patch)
		if line.epoch > 0 {
			candidate = fmt.Sprintf("%d!%s", line.epoch, candidate)
		}
		version, err := pep440.Parse(candidate)
		if err != nil {
			continue
		}
		if set.Check(version) {
			return true
		}
	}
	return false
}

func boundaryVersions(spec string) []boundaryVersion {
	matches := boundaryVersionPattern.FindAllStringSubmatch(spec, -1)
	out := make([]boundaryVersion, 0, len(matches))
	for _, match := range matches {
		boundary := boundaryVersion{}
		if match[1] != "" {
			epoch, err := strconv.ParseUint(match[1], 10, 32)
			if err != nil {
				continue
			}
			boundary.epoch = epoch
		}
		valid := true
		for _, field := range strings.Split(match[2], ".") {
			number, err := strconv.ParseUint(field, 10, 32)
			if err != nil {
				valid = false
				break
			}
			boundary.release = append(boundary.release, number)
		}
		if valid && len(boundary.release) > 0 {
			out = append(out, boundary)
		}
	}
	return out
}

func (b boundaryVersion) majorMinor() (uint64, uint64) {
	major, minor := b.release[0], uint64(0)
	if len(b.release) > 1 {
		minor = b.release[1]
	}
	return major, minor
}

// compliantImageOwnedRange renders one range that keeps the author's floor and
// admits every later minor of the same major line.
func compliantImageOwnedRange(name, spec string) string {
	boundaries := boundaryVersions(spec)
	if len(boundaries) == 0 {
		return ""
	}
	boundary := boundaries[0]
	major, minor := boundary.majorMinor()
	floor := fmt.Sprintf("%d.%d", major, minor)
	ceiling := fmt.Sprintf("%d", major+1)
	if boundary.epoch > 0 {
		floor = fmt.Sprintf("%d!%s", boundary.epoch, floor)
		ceiling = fmt.Sprintf("%d!%s", boundary.epoch, ceiling)
	}
	return fmt.Sprintf("%s>=%s,<%s", name, floor, ceiling)
}
