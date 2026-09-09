package install

import (
	"reflect"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestExactPrivateWheelRequirementsExcludeImageOwnedClosure(t *testing.T) {
	dependencies := map[string]packagepublish.CapturedDependency{
		"fixture": {Name: "fixture", Version: "1.0"},
		"torch":   {Name: "torch", Version: "2.13.0"},
		"scipy":   {Name: "scipy", Version: "1.18.1"},
	}
	if got := exactPrivateWheelRequirements("fixture", dependencies); !reflect.DeepEqual(got, []string{"scipy==1.18.1"}) {
		t.Fatalf("image-owned framework was repinned in callable wheel metadata: %v", got)
	}
	if got := portablePrivateWheelClosure("fixture==1.0\ntorch==2.13.0\nscipy==1.18.1"); got != "fixture==1.0\nscipy==1.18.1" {
		t.Fatalf("image-owned framework entered the portable private wheel closure: %v", got)
	}
}
