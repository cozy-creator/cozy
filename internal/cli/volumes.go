package cli

// th-122 — the volume verbs. A volume is the standing per-datacenter model
// store the hub mounts under the pods you rent there; warm it before the
// first rental, list what it costs and holds, drop it when the standing cost
// stops being worth it. The hub is the only state: nothing volume-shaped
// lives in the local records store.

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
		Fields:    []string{"datacenter", "provider", "state", "size", "rate", "warm", "volume"},
		AllFields: []string{"datacenter", "provider", "state", "size", "rate", "warm", "warm_objects", "volume", "created", "live", "last_bound"},
		Lead:      []string{"Standing storage per hour: " + usdPerHourBare(burn)},
		Trail:     []string{"A volume keeps its datacenter's model store warm between rentals; it bills until dropped."},
		Next:      []string{"cozy help volume warm"},
	}
	for _, v := range volumes {
		list.Rows = append(list.Rows, map[string]string{
			"datacenter": v.Datacenter, "provider": v.Provider, "state": v.State,
			"size": fmt.Sprintf("%dGB", v.SizeGB), "rate": usdPerHourBare(v.USDMicrosPerHour),
			"warm":         gigabytes(v.WarmBytes),
			"warm_objects": fmt.Sprintf("%d", v.WarmObjects),
			"volume":       v.ID, "created": stamp(v.CreatedAt), "live": orNone(stamp(v.LiveAt)),
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
	}, Notes: []string{"Models warm onto it as rentals in " + volume.Datacenter + " download them."}})
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
	}, Notes: []string{"The store and its bytes are gone; the next rental there downloads cold."}})
}

func gigabytes(bytes int64) string {
	return fmt.Sprintf("%.1fGB", float64(bytes)/(1<<30))
}
