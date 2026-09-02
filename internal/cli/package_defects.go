package cli

// cl-078/th-106: the defect relay's entrypoint half. The orchestrator holds no
// Tensorhub client, so a descriptor-falsifying pod refusal reaches the hub
// through this owner — carrying the exact creator-signed delegation the
// download ran under, which is the report's whole chain of authority. The hub's
// tombstone is idempotent, so this is fire-and-forget: a failure is logged and
// the next rental of the same defective release files the same report.

import (
	"encoding/base64"
	"fmt"
	"io"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
)

type defectReporter struct {
	cfg  config.Config
	auth *accountauth.Manager
	log  io.Writer
}

func newDefectReporter(cfg config.Config, log io.Writer,
	auth *accountauth.Manager,
) *defectReporter {
	if log == nil {
		log = io.Discard
	}
	return &defectReporter{cfg: cfg, log: log, auth: auth}
}

func (o *defectReporter) report(defect orchestrator.ReleaseDefect) {
	c := client(&Context{Out: io.Discard, Err: o.log, Cfg: o.cfg, AccountAuth: o.auth})
	ref, problem := hub.ParseRef(defect.Package)
	if problem != nil {
		fmt.Fprintf(o.log, "defect report for %q refused locally: %s\n", defect.Package, problem.Message)
		return
	}
	hctx, cancel := hub.LongContext()
	defer cancel()
	result, problem := c.ReportPackageDefect(hctx, ref, defect.Release, hub.PackageDefectReport{
		Code:                defect.Code,
		Detail:              defect.Detail,
		RentalID:            defect.RentalID,
		DelegationBase64URL: base64.RawURLEncoding.EncodeToString(defect.Delegation),
		SignatureBase64URL:  base64.RawURLEncoding.EncodeToString(defect.Signature),
	}, "descriptor-falsifying refusal on rental "+defect.RentalID)
	if problem != nil {
		fmt.Fprintf(o.log, "defect report for %s@%s refused: %s\n",
			defect.Package, defect.Release, problem.Message)
		return
	}
	fmt.Fprintf(o.log, "release %s@%s marked %s (changed=%v)\n",
		defect.Package, defect.Release, result.State, result.Changed)
}

// localDefectReporter is the LOCAL-preparation arm of the relay: a published
// install on this machine derived a different descriptor than the committed
// release. There is no rental chain here, so under the hub's sound
// authorization only an admin-credentialed daemon's report lands; the local
// install still refuses either way, which is the load-bearing half.
func localDefectReporter(cli *Context, ref hub.Ref, release string) func(code, detail string) {
	return func(code, detail string) {
		c := client(cli)
		hctx, cancel := hub.LongContext()
		defer cancel()
		result, problem := c.ReportPackageDefect(hctx, ref, release, hub.PackageDefectReport{
			Code: code, Detail: detail,
		}, "descriptor falsification observed during local preparation")
		if problem != nil {
			fmt.Fprintf(cli.Err, "defect report for %s@%s refused: %s\n",
				ref.String(), release, problem.Message)
			return
		}
		fmt.Fprintf(cli.Err, "release %s@%s marked %s (changed=%v)\n",
			ref.String(), release, result.State, result.Changed)
	}
}
