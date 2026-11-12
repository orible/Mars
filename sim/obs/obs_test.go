package obs

import (
	"math"
	"math/rand"
	"strings"
	"testing"

	"mars/sim/body"
	"mars/sim/command"
	"mars/sim/geom"
	"mars/sim/model"
)

// twin returns the name of a link or cylinder's left–right twin.
func twin(name string) string {
	switch {
	case strings.HasSuffix(name, "_l"):
		return strings.TrimSuffix(name, "_l") + "_r"
	case strings.HasSuffix(name, "_r"):
		return strings.TrimSuffix(name, "_r") + "_l"
	}
	return name
}

// The observation of the mirror image of a state must be the mirrored
// observation of the state, token for token (each cylinder's token
// becomes its twin's). Symmetric training relies on this.
func TestMirrorMatchesMirroredState(t *testing.T) {
	s := body.DefaultHuman()
	s.Spine = 1 // the thoracic twist cylinder has no twin
	m, err := model.New(body.Human(s))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1))
	a, b := model.NewWorld(m, true), model.NewWorld(m, true)
	idx := map[string]int{}
	for q := 0; q < m.NDof(); q++ {
		idx[m.Names[q+1]] = q
	}
	for q := 0; q < m.NDof(); q++ {
		a.Q[q] = 0.5 * (m.Lo[q+1] + rng.Float64()*(m.Hi[q+1]-m.Lo[q+1]))
		a.Qd[q] = rng.NormFloat64()
	}
	for q := 0; q < m.NDof(); q++ {
		tq := idx[twin(m.Names[q+1])]
		sign := 1.0
		if ax := m.Axis[q+1]; math.Abs(ax[2]) < 0.5 { // roll and yaw joints turn the other way in the mirror
			sign = -1
		}
		b.Q[tq], b.Qd[tq] = sign*a.Q[q], sign*a.Qd[q]
	}
	mir := geom.M3{{1, 0, 0}, {0, 1, 0}, {0, 0, -1}}
	a.Base.R = geom.AxisAngle(geom.V3{0.2, 1, 0.4}, 0.9)
	a.Base.P = geom.V3{0.3, 3, 0.7}
	a.Base.V = geom.V3{0.4, -0.2, 0.3}
	a.Base.W = geom.V3{0.5, -0.7, 0.9}
	b.Base.R = mir.Mul(a.Base.R).Mul(mir)
	b.Base.P = mir.MulV(a.Base.P)
	b.Base.V = mir.MulV(a.Base.V)
	b.Base.W = mir.MulV(a.Base.W).Scale(-1)
	cyl := map[string]int{}
	for k, c := range m.Cyl {
		cyl[c.Name] = k
	}
	for k, c := range m.Cyl {
		a.Cyl[k].X, a.Cyl[k].V, a.Cyl[k].PA, a.Cyl[k].PB, a.Cyl[k].Spool = rng.Float64()*c.P.Stroke, rng.NormFloat64(), 1e7*rng.Float64(), 1e7*rng.Float64(), rng.Float64()
		b.Cyl[cyl[twin(c.Name)]] = a.Cyl[k]
	}
	a.Refresh()
	b.Refresh()
	cmd := command.Command{HeadingDeg: 30, Speed: 0.4}
	cmdM := command.Command{HeadingDeg: -30, Speed: 0.4}
	oa := Mirror(Build(a, ReadIMU(a), cmd))
	ob := Build(b, ReadIMU(b), cmdM)

	same := func(x, y []float64) bool {
		for i := range x {
			if math.Abs(x[i]-y[i]) > 1e-9 {
				return false
			}
		}
		return true
	}
	if !same(oa.Global, ob.Global) {
		t.Fatalf("global:\n mirrored %v\n of twin  %v", oa.Global, ob.Global)
	}
	for k, c := range m.Cyl {
		if !same(oa.Tokens[k], ob.Tokens[cyl[twin(c.Name)]]) {
			t.Errorf("%s:\n mirrored %v\n of twin  %v", c.Name, oa.Tokens[k], ob.Tokens[cyl[twin(c.Name)]])
		}
	}
	for i, p := range oa.Parts {
		found := false
		for _, q := range ob.Parts {
			found = found || same(p, q)
		}
		if !found {
			t.Errorf("part %d has no twin: %v", i, p)
		}
	}
	for i, p := range oa.Sensors {
		found := false
		for _, q := range ob.Sensors {
			found = found || same(p, q)
		}
		if !found {
			t.Errorf("sensor %d has no twin: %v", i, p)
		}
	}
	if len(oa.Parts) != 16 {
		t.Errorf("%d parts, want pelvis, abdomen, 2×(thigh, shin, heel, midfoot, toes, upper arm, forearm) = 16", len(oa.Parts))
	}
	if len(oa.Sensors) != 2*(9+1) {
		t.Errorf("%d sensors, want 2×(9 sole pads + 1 plantar tendon) = 20", len(oa.Sensors))
	}
}
