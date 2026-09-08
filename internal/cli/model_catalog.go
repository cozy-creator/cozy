package cli

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
)

// Releases are opaque labels. Search gives each available label its own row,
// and info lists them all unless the caller explicitly selects one.
type modelSearchView struct {
	Cards []hub.ModelCard
	Notes []string
}

func (view modelSearchView) Emit(w io.Writer, mode output.Mode) error {
	fields := []string{"model", "family", "release", "lanes"}
	list := output.List{Name: "models", Fields: fields, AllFields: fields,
		TypedFields: fields, TypedAllFields: fields,
		TypedRows: []map[string]any{}, Notes: append([]string(nil), view.Notes...)}
	truncated := false
	for _, card := range view.Cards {
		count := 0
		for _, release := range card.Releases {
			if release.Yanked || release.Release == "" {
				continue
			}
			lanes := modelLaneNames(release.Lanes)
			cell := "[" + strings.Join(lanes, ", ") + "]"
			if mode.Human && !mode.JSON && !mode.Full {
				short := shortModelLanes(lanes)
				truncated = truncated || short != cell
				cell = short
			}
			row := map[string]string{"model": card.Model.Ref(), "family": card.Model.Family,
				"release": release.Release, "lanes": cell}

			list.Rows = append(list.Rows, row)
			list.TypedRows = append(list.TypedRows, map[string]any{"model": card.Model.Ref(), "family": card.Model.Family,
				"release": release.Release, "lanes": lanes})
			if len(list.Next) == 0 || strings.Contains(cell, "…") {
				list.Next = []string{"cozy model info " + card.Model.Ref() + "@" + release.Release}
			}
			count++
		}
		if count == 0 {
			list.Rows = append(list.Rows, map[string]string{"model": card.Model.Ref(), "family": card.Model.Family, "lanes": "[]"})
			list.TypedRows = append(list.TypedRows, map[string]any{"model": card.Model.Ref(), "family": card.Model.Family,
				"release": nil, "lanes": []string{}})
			if len(list.Next) == 0 {
				list.Next = []string{"cozy model info " + card.Model.Ref()}
			}
		}
	}
	if mode.Human && !mode.JSON && !mode.Full {
		for _, row := range list.Rows {
			for column, width := range map[string]int{"model": 28, "family": 14, "release": 24} {
				short := compactModelCell(row[column], width)
				truncated = truncated || short != row[column]
				row[column] = short
			}
		}
	}
	if truncated {
		list.Notes = append(list.Notes, "Use --full for complete values, or cozy model info <model> for details.")
	}
	list.Total = len(list.Rows)
	if len(list.Rows) == 0 {
		list.Next = []string{"cozy model search"}
	}
	return list.Emit(w, mode)
}

func modelLaneNames(lanes []hub.ModelLaneSummary) []string {
	seen := map[string]bool{}
	for _, lane := range lanes {
		if name := strings.TrimSpace(lane.Lane); name != "" {
			seen[name] = true
		}
	}
	names := make([]string, 0, len(seen))
	for name := range seen {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// compactModelCell keeps listing cells bounded; --full and model info carry the detail.
func compactModelCell(value string, width int) string {
	runes := []rune(value)
	if len(runes) <= width {
		return value
	}
	return string(runes[:width-1]) + "…"
}

func shortModelLanes(lanes []string) string {
	const width = 40
	full := "[" + strings.Join(lanes, ", ") + "]"
	if utf8.RuneCountInString(full) <= width {
		return full
	}
	short := ""
	for count := 1; count < len(lanes); count++ {
		text := "[" + strings.Join(lanes[:count], ", ") + fmt.Sprintf(", … +%d]", len(lanes)-count)
		if utf8.RuneCountInString(text) > width {
			break
		}
		short = text
	}
	if short != "" {
		return short
	}
	suffix := "]"
	if len(lanes) > 1 {
		suffix = fmt.Sprintf(", … +%d]", len(lanes)-1)
	}
	return "[" + compactModelCell(lanes[0], width-1-utf8.RuneCountInString(suffix)) + suffix
}

func handleModelInfo(ctx *Context) *exit.Error {
	name, release, selected := strings.Cut(strings.TrimSpace(ctx.Inv.Args[0]), "@")
	ref, problem := hub.ParseRef(name)
	if problem != nil {
		return problem
	}
	if selected && (release == "" || strings.Contains(release, "@")) {
		return exit.Usagef("model info expects org/name or org/name@release")
	}
	hctx, cancel := hub.Context()
	defer cancel()
	card, problem := client(ctx).ModelCard(hctx, ref)
	if problem != nil {
		return problem
	}
	if selected {
		found := false
		for _, row := range card.Releases {
			if row.Release == release {
				card.Releases, found = []hub.ModelReleaseSummary{row}, true
				break
			}
		}
		if !found {
			return exit.Named(exit.NotFound, "model_release_not_found", "model %s has no available release %q", ref.String(), release).
				WithNext("cozy model info " + ref.String())
		}
	}
	return emit(ctx, modelInfoView{Card: card})
}

type modelInfoView struct{ Card hub.ModelCard }

func (view modelInfoView) Emit(w io.Writer, mode output.Mode) error {
	card := view.Card
	metadata := []output.Field{{K: "model", V: card.Model.Ref()}, {K: "family", V: card.Model.Family},
		{K: "model_created", V: stamp(card.Model.CreatedAt)}, {K: "last_modified", V: nil}}
	releases := make([]map[string]any, 0, len(card.Releases))
	missingDate := false
	for _, release := range card.Releases {
		lanes := make([]map[string]any, 0, len(release.Lanes))
		for _, lane := range release.Lanes {
			row := map[string]any{"lane": lane.Lane, "bytes": lane.Bytes, "components": lane.Components,
				"checkpoint_id": lane.ManifestID, "checkpoint_ref": card.Model.Ref() + "@" + lane.ManifestID}
			if lane.ComponentBytes != nil {
				row["component_bytes"] = lane.ComponentBytes
			}
			lanes = append(lanes, row)
		}
		var created any
		if release.CutAt != "" {
			created = stamp(release.CutAt)
		} else {
			missingDate = true
		}
		releases = append(releases, map[string]any{"release": release.Release, "release_ref": card.Model.Ref() + "@" + release.Release,
			"revision": release.Revision, "release_created": created, "yanked": release.Yanked, "lanes": lanes})
	}
	note := "Tensorhub does not provide a last-modified timestamp."
	if missingDate {
		note = "Tensorhub does not provide last-modified or release-created timestamps."
	}
	// Info is the full detail surface: the only truncation is in search.
	mode.Full = true
	if !mode.Human || mode.JSON || len(mode.Fields) > 0 {
		return (output.Record{Fields: append(metadata, output.Field{K: "releases", V: releases}), Notes: []string{note}}).Emit(w, mode)
	}
	if err := (output.Record{Fields: metadata, Notes: []string{note}}).Emit(w, mode); err != nil {
		return err
	}
	if len(releases) == 0 {
		_, err := fmt.Fprintln(w, "\nNo releases are available.")
		return err
	}
	for i, release := range card.Releases {
		if _, err := fmt.Fprintln(w); err != nil {
			return err
		}
		fields := []output.Field{{K: "release", V: release.Release}, {K: "revision", V: release.Revision},
			{K: "release_created", V: releases[i]["release_created"]}, {K: "release_ref", V: releases[i]["release_ref"]}}
		if release.Yanked {
			fields = append(fields, output.Field{K: "status", V: "yanked"})
		}
		if err := (output.Record{Fields: fields}).Emit(w, mode); err != nil {
			return err
		}
		if _, err := fmt.Fprintln(w); err != nil {
			return err
		}
		columns := []string{"lane", "bytes", "checkpoint_ref"}
		list := output.List{Name: "lanes", Fields: columns, AllFields: columns, Bytes: []string{"bytes"}}
		for _, lane := range release.Lanes {
			list.Rows = append(list.Rows, map[string]string{"lane": lane.Lane, "bytes": output.Int(lane.Bytes),
				"checkpoint_ref": card.Model.Ref() + "@" + lane.ManifestID})
		}
		list.Total = len(list.Rows)
		if err := list.Emit(w, mode); err != nil {
			return err
		}
	}
	return nil
}
