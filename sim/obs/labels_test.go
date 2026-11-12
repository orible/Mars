package obs

import (
	"testing"

	"mars/sim/body"
	"mars/sim/command"
	"mars/sim/model"
)

// Labels names exactly the tokens Build makes, and the feature name lists
// match the feature sizes.
func TestLabelsMatchBuild(t *testing.T) {
	for _, n := range [][2]int{{len(TokenFeatures), TokenDim}, {len(PartFeatures), PartDim}, {len(SensorFeatures), SensorDim}, {len(GlobalFeatures), GlobalDim}} {
		if n[0] != n[1] {
			t.Fatalf("feature names %d, feature size %d", n[0], n[1])
		}
	}
	m, err := model.New(body.Human(body.DefaultHuman()))
	if err != nil {
		t.Fatal(err)
	}
	w := model.NewWorld(m, true)
	o := Build(w, ReadIMU(w), command.Stop)
	tok, parts, sens := Labels(w)
	if len(tok) != len(o.Tokens) || len(parts) != len(o.Parts) || len(sens) != len(o.Sensors) {
		t.Fatalf("labels %d/%d/%d, tokens %d/%d/%d", len(tok), len(parts), len(sens), len(o.Tokens), len(o.Parts), len(o.Sensors))
	}
	if len(sens) == 0 {
		t.Fatal("the human body should have sole pads")
	}
}
