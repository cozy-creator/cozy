package cli

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/transfer"
)

func handleRegistryInstall(ctx *Context) *exit.Error {
	ref, release, problem := registryPackageRef(ctx.Inv.Args[0])
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
		release = card.Releases[0].Release
	}
	profile := strings.TrimSpace(ctx.Inv.Value("--profile"))
	if profile == "" {
		return exit.Usagef("cozy package install requires --profile <approved-profile>")
	}
	plan, problem := c.PackageInstallPlan(hctx, ref, release, profile)
	if problem != nil {
		return problem
	}
	major, majorProblem := install.MajorOf(release)
	if majorProblem != nil {
		return majorProblem
	}
	_, existing, _, problem := open(ctx.Cfg, false)
	if problem != nil {
		return problem
	}
	_, generation, problem := existing.ActivePin(ref.String(), major)
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

func registryPackageRef(value string) (hub.Ref, string, *exit.Error) {
	name, release, hasRelease := strings.Cut(strings.TrimSpace(value), "@")
	if hasRelease && (release == "" || strings.ContainsAny(release, `@/\`)) {
		return hub.Ref{}, "", exit.Usagef("%q is not org/package[@release]", value)
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
