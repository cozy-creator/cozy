package install

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
)

type Writer = home.Writer

func Lock(layout home.Layout) (*Writer, *exit.Error) { return home.LockWriter(layout) }
