package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// reconsiderAutomaticRental drops an early automatic assignment only when the
// ordinary router already has a compatible free seat elsewhere. It never starts
// preparation or purchases capacity. Explicit affinity and any attempted or
// retained work stay with their existing owner.
func (c *Orchestrator) reconsiderAutomaticRental(req records.Request) (records.Request, *exit.Error) {
	if req.RentNew || !req.Rental || req.Worker == "" || req.RequestedRental != "" || req.RetainWork || req.Ordinal != 0 || req.LocalPackageUploadedBootID != "" {
		return req, nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	slotReq := req
	slotReq.Package = pinnedPackage(req.Package, req.Worker)
	if c.starting[requestSlot(slotReq)] || c.starting["rental/"+requestSlot(req)] || c.route(req).pick() != nil {
		return req, nil
	}
	unassigned := req
	unassigned.Worker = ""
	alternative := c.route(unassigned).pick()
	if alternative == nil || alternative.worker.spec.Connection == nil || alternative.worker.spec.Connection.RentalID == req.Worker {
		return req, nil
	}
	changed, problem := c.opt.Store.ReleaseUnattemptedRentalAssignment(req.ID, req.Worker)
	if problem != nil {
		return req, problem
	}
	if changed {
		return unassigned, nil
	}
	return req, nil
}
