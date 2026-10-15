package records

import "strings"

type runStatusGroup struct {
	public string
	stored []string
}

var runStatusGroups = []runStatusGroup{
	{"queued", []string{"submitted", "queued", "requeue_pending"}},
	{"in_progress", []string{"dispatching"}},
	{"completed", []string{"succeeded"}},
	{"failed", []string{"failed", "abandoned", "refused"}},
	{"canceled", []string{"canceled"}},
	{"canceling", []string{"releasing"}},
}

func PublicRunStatus(state string) string {
	for _, group := range runStatusGroups {
		for _, stored := range group.stored {
			if state == stored {
				return group.public
			}
		}
	}
	return state
}

func publicRunStatusSQL() string {
	var text strings.Builder
	text.WriteString("CASE")
	for _, group := range runStatusGroups {
		text.WriteString(" WHEN requests.state IN ('" + strings.Join(group.stored, "','") + "') THEN '" + group.public + "'")
	}
	text.WriteString(" ELSE requests.state END")
	return text.String()
}
