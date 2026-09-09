package cli

// th-122 — the volume verbs. A volume is an optional per-datacenter cache of
// immutable repo objects from model and dataset snapshots. It accelerates
// downloads but is disposable and never authoritative. The hub is the only
// state: nothing volume-shaped lives in the local records store.

import (
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
)

func handleVolumeLs(ctx *Context) *exit.Error {
	hctx, cancel := hub.Context()
	volumes, e := client(ctx).Volumes(hctx)
	cancel()
	if e != nil {
		return e
	}
	var burn int64
	for _, v := range volumes {
		burn += v.USDMicrosPerHour
	}
	list := output.List{
		Name:      "volumes",
		Fields:    []string{"datacenter", "provider", "state", "size", "rate", "volume"},
		AllFields: []string{"datacenter", "provider", "state", "size", "rate", "volume", "created", "live", "last_bound"},
		Lead:      []string{"Optional cache storage per hour: " + usdPerHourBare(burn)},
		Trail:     []string{"Cache volumes hold disposable copies of immutable model and dataset snapshot objects; they are never authoritative.", "Cache usage is not measured."},
		Next:      []string{"cozy help volume warm"},
	}
	for _, v := range volumes {
		list.Rows = append(list.Rows, map[string]string{
			"datacenter": v.Datacenter, "provider": v.Provider, "state": v.State,
			"size": fmt.Sprintf("%dGB", v.SizeGB), "rate": usdPerHourBare(v.USDMicrosPerHour),
			"volume": v.ID, "created": stamp(v.CreatedAt), "live": orNone(stamp(v.LiveAt)),
			"last_bound": orNone(stamp(v.LastBoundAt)),
		})
	}
	if len(list.Rows) > 0 {
		list.Next = []string{"cozy volume drop " + list.Rows[0]["volume"]}
	}
	return emit(ctx, list)
}

func handleVolumeWarm(ctx *Context) *exit.Error {
	datacenter := strings.TrimSpace(ctx.Inv.Args[0])
	if datacenter == "" {
		return exit.Usagef("name the datacenter to warm").
			WithNext("cozy volume")
	}
	provider := ctx.Inv.Value("--provider")
	hctx, cancel := hub.Context()
	volume, e := client(ctx).WarmVolume(hctx, provider, datacenter,
		"cozy volume warm "+datacenter)
	cancel()
	if e != nil {
		return e
	}
	return emit(ctx, output.Record{Fields: []output.Field{
		{K: "volume", V: volume.ID},
		{K: "datacenter", V: volume.Datacenter},
		{K: "provider", V: volume.Provider},
		{K: "state", V: volume.State},
		{K: "size", V: fmt.Sprintf("%dGB", volume.SizeGB)},
		{K: "rate", V: usdPerHourBare(volume.USDMicrosPerHour)},
	}, Notes: []string{"Rentals in " + volume.Datacenter + " may cache immutable model and dataset snapshot objects here. The cache is optional and disposable."}})
}

func handleVolumeDrop(ctx *Context) *exit.Error {
	target := strings.TrimSpace(ctx.Inv.Args[0])
	if target == "" {
		return exit.Usagef("name the volume id or datacenter to drop").WithNext("cozy volume")
	}
	c := client(ctx)
	id := target
	if !strings.HasPrefix(target, "pvl-") {
		hctx, cancel := hub.Context()
		volumes, e := c.Volumes(hctx)
		cancel()
		if e != nil {
			return e
		}
		id = ""
		for _, v := range volumes {
			if v.Datacenter == target {
				id = v.ID
				break
			}
		}
		if id == "" {
			return exit.Named(exit.NotFound, "volume.not_found",
				"no standing volume in %q", target).WithNext("cozy volume")
		}
	}
	hctx, cancel := hub.Context()
	e := c.DropVolume(hctx, id, "cozy volume drop "+target)
	cancel()
	if e != nil {
		return e
	}
	return emit(ctx, output.Record{Fields: []output.Field{
		{K: "volume", V: id},
		{K: "state", V: "dropped"},
	}, Notes: []string{"The cached copies are gone; the next rental there fetches snapshots from another source."}})
}

func gigabytes(bytes int64) string {
	return fmt.Sprintf("%.1fGB", float64(bytes)/(1<<30))
}
