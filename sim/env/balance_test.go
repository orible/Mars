package env

import (
	"math"
	"testing"

	"mars/sim/geom"
)

// shockRun stands the human in its neutral pose (valves holding it), with
// an optional push at 0.5 s, and returns the peak accelerometer reading
// while quiet (before any push, after settling), the peak overall and
// the largest capture point distance.
func shockRun(t *testing.T, push float64, seconds float64) (quiet, peak, cp float64, fell bool) {
	t.Helper()
	sc, _ := LoadScenario("balance")
	sc.RandomBodyFrac, sc.RoughFloor, sc.PushEvery, sc.Assist = 0, 0, 0, 0
	e := New(biped(t), sc, 1)
	e.Reset()
	a := make([]float64, len(e.W.M.Cyl))
	for s := 0; float64(s)*PolicyDt < seconds; s++ {
		if push != 0 && s == int(0.5/PolicyDt) {
			e.W.Base.V = e.W.Base.V.Add(geom.V3{0, 0, push})
		}
		_, term, _ := e.Step(a)
		tm := float64(s) * PolicyDt
		if tm > 0.2 && tm < 0.5 {
			quiet = math.Max(quiet, e.shock)
		}
		peak = math.Max(peak, e.shock)
		cp = math.Max(cp, e.cpDist)
		if term || e.W.Fallen() {
			fell = true
		}
	}
	return
}

// Quiet standing must read well under the free threshold, so the penalty
// only bites on real jolts, and a fall must read far above it.
func TestShockThresholds(t *testing.T) {
	quiet, _, cp0, _ := shockRun(t, 0, 0.5)
	t.Logf("quiet standing: peak %.2f g, capture point within %.3f m", quiet, cp0)
	if quiet > shockFree/2 {
		t.Errorf("quiet standing reads %.2f g, too close to the %.1f g threshold", quiet, shockFree)
	}
	_, peak, cp, fell := shockRun(t, 1.5, 4)
	t.Logf("hard sideways push: peak %.2f g, capture point up to %.3f m, fell %v", peak, cp, fell)
	if !fell {
		t.Skip("the valves held it; no fall to measure")
	}
	if peak < 2*shockFree {
		t.Errorf("a fall reads only %.2f g", peak)
	}
	if cp < 3*shuffleCP {
		t.Errorf("capture point only moved %.3f m in a fall", cp)
	}
}
