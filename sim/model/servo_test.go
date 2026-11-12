package model

import (
	"math"
	"math/rand"
	"testing"
)

// The servo holds the standing pose (all actions 0) without oscillating,
// and under random action noise the body stays up longer than with the
// same noise on raw valve openings, where it adds up into drift. Nothing
// stands for ever without balancing: the body is an inverted pendulum, so
// a held posture topples once disturbed; that feedback is the policy's
// job. The servo's is to keep noise and errors from accumulating.
func TestServoHoldsPoseAndBeatsFlowNoise(t *testing.T) {
	m := loadBiped(t)
	run := func(sigma float64, servo bool, seed int64) float64 {
		w := NewWorld(m, true)
		rng := rand.New(rand.NewSource(seed))
		a := make([]float64, len(m.Cyl))
		up := 0.0
		for s := 0; s < 10*1000; s++ { // 10 s at 1 kHz, actions changing at 100 Hz
			if s%10 == 0 {
				for k := range a {
					a[k] = sigma * rng.NormFloat64()
				}
				if servo {
					w.SetActionTargets(a)
				} else {
					copy(w.Cmd, a)
				}
			}
			for i := 0; i < 20; i++ {
				w.Step(50e-6)
			}
			if !w.Healthy() {
				t.Fatalf("diverged at %.2f s", w.T)
			}
			if w.Fallen() {
				break
			}
			up = w.T
		}
		return up
	}
	if up := run(0, true, 1); up < 6 {
		t.Errorf("holding the standing pose: fell after %.1f s", up)
	} else {
		t.Logf("holding the standing pose: up %.1f s of 10", up)
	}
	var flow, servo float64
	for seed := int64(1); seed <= 3; seed++ {
		flow += run(0.05, false, seed) / 3
		servo += run(0.05, true, seed) / 3
	}
	t.Logf("noise 0.05: valve openings %.1f s up, servo targets %.1f s up", flow, servo)
	if servo < 1.4*flow {
		t.Errorf("the servo should stand clearly longer under noise: %.1f s vs %.1f s", servo, flow)
	}
}

// The servo moves a cylinder to a new target and holds it there.
func TestServoTracksTarget(t *testing.T) {
	m := loadBiped(t)
	w := NewWorld(m, false) // test stand: the legs hang free
	k := -1
	for i, c := range m.Cyl {
		if !c.P.PressureControl && len(c.Span) == 1 {
			k = i
			break
		}
	}
	a := make([]float64, len(m.Cyl))
	a[k] = 0.3
	w.SetActionTargets(a)
	goal := w.Targets[k]
	var settled float64 = -1
	for s := 0; s < 1000; s++ { // 1 s
		for i := 0; i < 20; i++ {
			w.Step(50e-6)
		}
		x := w.Len[k] - m.Cyl[k].Retracted
		if settled < 0 && math.Abs(x-goal) < 0.02*m.Cyl[k].P.Stroke {
			settled = w.T
		}
	}
	x := w.Len[k] - m.Cyl[k].Retracted
	t.Logf("%s: target %.4f m, reached within 2%% of stroke at %.2f s, after 1 s at %.4f m", m.Cyl[k].Name, goal, settled, x)
	if settled < 0 || math.Abs(x-goal) > 0.02*m.Cyl[k].P.Stroke {
		t.Errorf("%s did not settle on its target", m.Cyl[k].Name)
	}
}
