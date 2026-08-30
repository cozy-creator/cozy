package cli

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hostgpu"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/transfer"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
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
	target, problem := localPackageInstallTarget(ctx)
	if problem != nil {
		return problem
	}
	plan, problem := c.PackageDownloads(hctx, ref, release, target)
	if problem != nil {
		return problem
	}
	if plan.Release == "" || release != "" && plan.Release != release {
		return exit.Internalf("Tensorhub returned a changed or absent package release")
	}
	release = plan.Release
	releaseDigest, problem := selectedReleaseDigest(plan.PlacementSet, ref.String(), release)
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
	if generation != nil && generation.SourceDigest == releaseDigest &&
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
	published, problem := downloadPackageInstallPlan(hctx, ctx, scratch, ref, release, releaseDigest, plan)
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
		Accelerator: "cpu", OS: runtime.GOOS, Arch: packageInstallArchitecture(runtime.GOARCH),
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
		return "x86"
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
	release, releaseDigest string, plan hub.PackageDownloadPlan,
) (*install.PublishedSource, *exit.Error) {
	if plan.Profile == "" || len(plan.Downloads) == 0 || len(plan.Downloads) > 20_033 {
		return nil, exit.Internalf("Tensorhub returned an invalid package install plan")
	}
	published := &install.PublishedSource{
		Package: ref.String(), Release: release, SourceDigest: releaseDigest,
		Artifacts: map[string]string{},
		Selection: install.Selection{
			Profile:            plan.Profile,
			PlacementSet:       install.ExactDocument{Bytes: plan.PlacementSet.CanonicalBytes, Digest: plan.PlacementSet.Digest, Length: plan.PlacementSet.Length},
			PackageDescriptor:  install.ExactDocument{Bytes: plan.PackageDescriptor.CanonicalBytes, Digest: plan.PackageDescriptor.Digest, Length: plan.PackageDescriptor.Length},
			Qualification:      install.ExactDocument{Bytes: plan.Qualification.CanonicalBytes, Digest: plan.Qualification.Digest, Length: plan.Qualification.Length},
			EnvironmentReceipt: install.ExactDocument{Bytes: plan.EnvironmentReceipt.CanonicalBytes, Digest: plan.EnvironmentReceipt.Digest, Length: plan.EnvironmentReceipt.Length},
			WheelhouseManifest: install.ExactDocument{Bytes: plan.WheelhouseManifest.CanonicalBytes, Digest: plan.WheelhouseManifest.Digest, Length: plan.WheelhouseManifest.Length},
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
				published.ProjectWheelDigest = download.Digest
			} else {
				published.Wheels = append(published.Wheels, dst)
			}
		case "artifact":
			name := strings.TrimPrefix(download.Digest, "sha256:")
			if len(name) != 64 || download.Path != name || filepath.Base(download.Path) != download.Path {
				return nil, exit.Internalf("Tensorhub returned unsafe package artifact path %q", download.Path)
			}
			dst = filepath.Join(scratch, "artifacts", name)
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
	set, err := canonical.Read(plan.PlacementSet.CanonicalBytes, &pb.PlacementSet{})
	if err != nil || len(set.List("placements")) != 1 ||
		set.List("placements")[0].Sub("package").Sub("project_wheel").Sub("ref").Str("digest") != published.ProjectWheelDigest {
		return nil, exit.Named(exit.Conflict, "package_selection_project_wheel_mismatch",
			"Tensorhub project wheel download does not match the selected PlacementSet")
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
	for _, item := range jobs {
		published.Artifacts[item.download.Digest] = item.dst
		published.Files++
		published.Bytes += item.download.Length
	}
	return published, nil
}

func selectedReleaseDigest(document hub.ExactDocument, pkg, release string) (string, *exit.Error) {
	set, err := canonical.Read(document.CanonicalBytes, &pb.PlacementSet{})
	if err != nil || len(set.List("placements")) != 1 {
		return "", exit.Named(exit.Structural, "package_selection_placement_invalid",
			"Tensorhub returned an invalid single-package PlacementSet")
	}
	fact := set.List("placements")[0].Sub("package")
	digest := fact.Str("release_digest")
	if fact.Str("package") != pkg || fact.Str("release") != release {
		return "", exit.Named(exit.Conflict, "package_selection_release_mismatch",
			"Tensorhub selected %s@%s while %s@%s was requested",
			fact.Str("package"), fact.Str("release"), pkg, release)
	}
	if _, err := canonical.Raw(digest); err != nil {
		return "", exit.Named(exit.Structural, "package_selection_release_digest_invalid",
			"Tensorhub selected an invalid release digest")
	}
	return digest, nil
}
