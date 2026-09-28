package orchestrator

// rentalCanServe answers whether the rental a request is pinned to could still take it.
// A row that is absent, terminally failed, or released cannot, and that is a durable fact
// this host can read without asking anything remote — which matters, because the thing it
// would have to ask is the thing that is gone.
func (c *Orchestrator) rentalCanServe(rentalID string) bool {
	if rentalID == "" {
		return false
	}
	row, problem := c.opt.Store.RentalRow(rentalID)
	if problem != nil {
		// An unreadable store is not an observation about the pod. Leave the attempt alone.
		return true
	}
	return row != nil && row.State != "failed" && row.State != "released" && row.State != "release_requested"
}
