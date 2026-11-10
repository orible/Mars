package hydraulics

import (
	"math"
	"testing"
)

func testParams() Params {
	return Params{
		BoreArea:   2.0e-4, // 16 mm bore
		RodArea:    1.2e-4,
		Stroke:     0.05,
		DeadVolume: 5e-6,
		Mass:       2,
		Viscous:    200,
		ValveGain:  9e-9,
		ValveTau:   0.01,
	}
}

func run(t *testing.T, cmd float64) State {
	t.Helper()
	p, s := testParams(), DefaultSupply()
	st := NewState(p, s)
	const dt = 50e-6
	for i := 0; i < int(0.3/dt); i++ {
		Step(p, s, &st, cmd, 20, dt)
		for _, v := range []float64{st.X, st.V, st.PA, st.PB} {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("non-finite state at step %d: %+v", i, st)
			}
		}
		if st.PA < 0 || st.PB < 0 || st.PA > 1.2*s.Pressure+1 || st.PB > 1.2*s.Pressure+1 {
			t.Fatalf("pressure out of range at step %d: %+v", i, st)
		}
	}
	return st
}

func TestExtendsOnPositiveCommand(t *testing.T) {
	st := run(t, 1)
	if st.X < 0.025+0.005 {
		t.Fatalf("expected extension, got x=%.4f", st.X)
	}
}

func TestRetractsOnNegativeCommand(t *testing.T) {
	st := run(t, -1)
	if st.X > 0.025-0.005 {
		t.Fatalf("expected retraction, got x=%.4f", st.X)
	}
}
