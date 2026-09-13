// runtimecarrier builds the exact Creator carrier for coordinated builtin tests.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/runtimeoperation"
)

func main() {
	input := flag.String("interface", "", "trusted Runtime builtin interface file")
	version := flag.String("version", "", "exact Runtime version")
	output := flag.String("out", "", "owned carrier output directory")
	flag.Parse()
	if *input == "" || *version == "" || !filepath.IsAbs(*output) {
		panic("interface, version and absolute owned output are required")
	}
	raw, err := os.ReadFile(*input)
	if err != nil {
		panic(err)
	}
	surface, problem := launch.DecodePackageInterface(raw)
	if problem != nil {
		panic(problem)
	}
	if surface.Application != runtimeoperation.Application {
		panic("not the fixed Runtime App")
	}
	name, body, problem := runtimeoperation.Carrier(*version, surface.Digest)
	if problem != nil {
		panic(problem)
	}
	if err := os.MkdirAll(*output, 0700); err != nil {
		panic(err)
	}
	path := filepath.Join(*output, name)
	if err := os.WriteFile(path, body, 0400); err != nil {
		panic(err)
	}
	fmt.Println(path)
}
