package cli

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"

	pep440 "github.com/aquasecurity/go-pep440-version"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hostgpu"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/transfer"
)

func handleRegistryInstall(ctx *Context) *exit.Error {
	ref, release, problem := registryPackageRef(ctx.Inv.Args[0], ctx.Inv.Value("--version"))
	if problem != nil {
		return problem
	}
	c := client(ctx)
	hctx, cancel := hub.LongContext()
	defer cancel()
	packagePublishStatus(ctx, "Resolving %s...", ref.String())
	if release == "" {
		card, problem := c.PackageCard(hctx, ref)
		if problem != nil {
			return problem
		}
		if len(card.Releases) == 0 {
			return exit.New(exit.NotFound, "%s has no published releases", ref.String()).
				WithNext("cozy package search " + ref.String())
		}
		release, problem = latestPackageRelease(card.Releases)
		if problem != nil {
			return problem
		}
	}
	target, problem := localPackageInstallTarget(ctx)
	if problem != nil {
		return problem
	}
	plan, problem := c.PackageInstallPlan(hctx, ref, release, target)
	if problem != nil {
		return problem
	}
	_, existing, _, problem := open(ctx.Cfg, false)
	if problem != nil {
		return problem
	}
	_, generation, problem := existing.ActivePackage(ref.String())
	if problem != nil {
		existing.Close()
		return problem
	}
	if generation != nil && generation.SourceDigest == plan.PackageRelease.Digest &&
		generation.PlacementSetDigest == plan.PlacementSet.Digest {
		defer existing.Close()
		return emitInstallResult(ctx, existing, &install.Result{Gen: *generation, Idempotent: true})
	}
	existing.Close()
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return problem
	}
	if err := os.MkdirAll(layout.Transfer, 0o700); err != nil {
		return exit.Internalf("cannot create package download scratch: %s", err)
	}
	scratch, err := os.MkdirTemp(layout.Transfer, "package-install-")
	if err != nil {
		return exit.Internalf("cannot create package download scratch: %s", err)
	}
	defer os.RemoveAll(scratch)
	published, problem := downloadPackageInstallPlan(hctx, ctx, scratch, ref, release, plan)
	if problem != nil {
		return problem
	}

	_, st, writer, problem := open(ctx.Cfg, true)
	if problem != nil {
		return problem
	}
	defer st.Close()
	defer writer.Unlock()
	var result *install.Result
	problem = packagePublishStage(ctx, "Creating local package environment", func() *exit.Error {
		var installProblem *exit.Error
		result, installProblem = install.Run(layout, st, install.Request{
			Ref: install.Ref{Package: ref.String()}, Force: true, Published: published,
		})
		return installProblem
	})
	if problem != nil {
		return problem
	}
	return emitInstallResult(ctx, st, result)
}

func localPackageInstallTarget(ctx *Context) (hub.PackageInstallTarget, *exit.Error) {
	target := hub.PackageInstallTarget{
		Accelerator: "cpu", OS: runtime.GOOS, Architecture: packageInstallArchitecture(runtime.GOARCH),
	}
	inventory := hostgpu.Probe(ctx.Cfg)
	if len(inventory.GPUs) == 0 {
		return target, nil
	}
	gpu := inventory.GPUs[0]
	for _, candidate := range inventory.GPUs[1:] {
		if gpuCapability(candidate) > gpuCapability(gpu) ||
			gpuCapability(candidate) == gpuCapability(gpu) && candidate.Index < gpu.Index {
			gpu = candidate
		}
	}
	if gpu.DriverCUDAVersion == "" {
		return target, exit.Named(exit.Unavailable, "local_gpu.compatibility_unknown",
			"the local NVIDIA driver did not report its CUDA compatibility").
			WithRemedy("check that `nvidia-smi` runs successfully, then retry the install")
	}
	target.Accelerator = "nvidia"
	target.DriverCUDA = gpu.DriverCUDAVersion
	target.ComputeCapability = gpu.ComputeCapability
	return target, nil
}

func packageInstallArchitecture(arch string) string {
	switch arch {
	case "amd64":
		return "x86_64"
	case "arm64":
		return "aarch64"
	default:
		return arch
	}
}

func gpuCapability(gpu hostgpu.GPU) int {
	value, _ := strconv.Atoi(strings.TrimPrefix(gpu.SM, "sm_"))
	return value
}

func latestPackageRelease(releases []hub.ReleaseSummary) (string, *exit.Error) {
	var chosen pep440.Version
	name := ""
	for _, release := range releases {
		version, err := pep440.Parse(release.Release)
		if err != nil {
			return "", exit.Named(exit.Structural, "package.release_version_invalid",
				"Tensorhub returned package release %q, which is not a Python package version", release.Release)
		}
		if name == "" || version.GreaterThan(chosen) {
			chosen, name = version, release.Release
		}
	}
	return name, nil
}

func registryPackageRef(value, release string) (hub.Ref, string, *exit.Error) {
	name := strings.TrimSpace(value)
	if strings.Contains(name, "@") {
		return hub.Ref{}, "", exit.Usagef("a version does not belong in the package name").
			WithRemedy("use cozy package install org/package --version 1.2.3")
	}
	release = strings.TrimSpace(release)
	if strings.ContainsAny(release, `@/\`) {
		return hub.Ref{}, "", exit.Usagef("--version %q is not a release name", release)
	}
	ref, problem := hub.ParseRef(name)
	return ref, release, problem
}

func downloadPackageInstallPlan(ctx context.Context, cli *Context, scratch string, ref hub.Ref,
	release string, plan hub.PackageInstallPlan,
) (*install.PublishedSource, *exit.Error) {
	if plan.Profile == "" || len(plan.Downloads) == 0 || len(plan.Downloads) > 20_033 {
		return nil, exit.Internalf("Tensorhub returned an invalid package install plan")
	}
	published := &install.PublishedSource{
		Package: ref.String(), Release: release, SourceDigest: plan.PackageRelease.Digest,
		Selection: install.Selection{
			Profile:           plan.Profile,
			PlacementSet:      install.ExactDocument{Bytes: plan.PlacementSet.CanonicalBytes, Digest: plan.PlacementSet.Digest, Length: plan.PlacementSet.Length},
			PackageRelease:    install.ExactDocument{Bytes: plan.PackageRelease.CanonicalBytes, Digest: plan.PackageRelease.Digest, Length: plan.PackageRelease.Length},
			PackageDescriptor: install.ExactDocument{Bytes: plan.PackageDescriptor.CanonicalBytes, Digest: plan.PackageDescriptor.Digest, Length: plan.PackageDescriptor.Length},
			Qualification:     install.ExactDocument{Bytes: plan.Qualification.CanonicalBytes, Digest: plan.Qualification.Digest, Length: plan.Qualification.Length},
		},
	}
	seen := map[string]bool{}
	type job struct {
		download hub.PackageInstallDownload
		dst      string
	}
	jobs := make([]job, 0, len(plan.Downloads))
	for _, download := range plan.Downloads {
		var dst string
		switch download.Kind {
		case "project_wheel", "dependency_wheel":
			if filepath.Base(download.Path) != download.Path || !strings.HasSuffix(download.Path, ".whl") {
				return nil, exit.Internalf("Tensorhub returned unsafe package wheel path %q", download.Path)
			}
			dst = filepath.Join(scratch, "wheels", download.Path)
			if download.Kind == "project_wheel" {
				if published.ProjectWheel != "" {
					return nil, exit.Internalf("Tensorhub returned more than one project wheel")
				}
				published.ProjectWheel = dst
			} else {
				published.Wheels = append(published.Wheels, dst)
			}
		default:
			return nil, exit.Internalf("Tensorhub returned unknown package file kind %q", download.Kind)
		}
		if seen[dst] || download.URL == "" {
			return nil, exit.Internalf("Tensorhub returned a duplicate or unreadable package file")
		}
		seen[dst] = true
		jobs = append(jobs, job{download: download, dst: dst})
	}
	if published.ProjectWheel == "" {
		return nil, exit.Internalf("Tensorhub package install plan has no project wheel")
	}
	queue := make(chan job, len(jobs))
	results := make(chan *exit.Error, len(jobs))
	for _, item := range jobs {
		queue <- item
	}
	close(queue)
	var group sync.WaitGroup
	for range min(16, len(jobs)) {
		group.Add(1)
		go func() {
			defer group.Done()
			for item := range queue {
				results <- transfer.DownloadExact(ctx, item.download.Path, item.download.URL, item.dst,
					item.download.Digest, item.download.Length)
			}
		}()
	}
	go func() {
		group.Wait()
		close(results)
	}()
	progress := packageFileCounter(cli, "Downloading files")
	progress(0, len(jobs))
	completed := 0
	var firstProblem *exit.Error
	for problem := range results {
		completed++
		progress(completed, len(jobs))
		if firstProblem == nil && problem != nil {
			firstProblem = problem
		}
	}
	if firstProblem != nil {
		return nil, firstProblem
	}
	return published, nil
}
