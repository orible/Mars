package model

import (
	"math"
	"math/rand"
	"testing"

	"mars/sim/body"
	"mars/sim/geom"
	"mars/sim/hydraulics"
)

func loadBiped(t *testing.T) *Model {
	t.Helper()
	m, err := loadBipedB()
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func loadBipedB() (*Model, error) {
	b, err := body.Load("../bodies/biped.json")
	if err != nil {
		return nil, err
	}
	return New(b)
}

func cylParams() hydraulics.Params {
	return hydraulics.Params{BoreArea: 2e-4, RodArea: 1.2e-4, Stroke: 0.05, DeadVolume: 5e-6, Viscous: 200, ValveGain: 9e-9, ValveTau: 0.01}
}

// pendulum is a chain of rod links hanging from the pelvis, each length l,
// mass mass, all hinged about Z.
func pendulum(t *testing.T, links int, l, mass float64) *Model {
	t.Helper()
	b := body.Body{Name: "pendulum", Supply: hydraulics.DefaultSupply()}
	b.Links = append(b.Links, body.Link{Name: "base", Mass: 1, Inertia: geom.V3{1, 1, 1}})
	parent := "base"
	for i := 0; i < links; i++ {
		name := string(rune('a' + i))
		origin := geom.V3{}
		if i > 0 {
			origin = geom.V3{0, -l, 0}
		}
		b.Links = append(b.Links, body.Link{
			Name: name, Parent: parent, Mass: mass, Com: geom.V3{0, -l, 0}, Inertia: geom.V3{1e-6, 1e-6, 1e-6},
			Joint: &body.Joint{Origin: origin, Axis: geom.V3{0, 0, 1}, Limits: [2]float64{-4, 4}},
		})
		parent = name
	}
	// One cylinder so the spec validates; the tests below never step it.
	b.Cylinders = []body.Cylinder{{Name: "c", Base: body.Anchor{Link: "base", Point: geom.V3{0.1, 0, 0}},
		Rod: body.Anchor{Link: "a", Point: geom.V3{0, -0.3, 0}}, RetractedLength: 0.3, Params: cylParams()}}
	m, err := New(b)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestLengthGradientMatchesFiniteDifference(t *testing.T) {
	m := loadBiped(t)
	rng := rand.New(rand.NewSource(1))
	q := make([]float64, m.NDof())
	for i := range q {
		l := i + 1
		q[i] = m.Lo[l] + rng.Float64()*(m.Hi[l]-m.Lo[l])
	}
	ps := m.NewPose()
	const h = 1e-6
	for k, c := range m.Cyl {
		m.FK(q, ps)
		grad := make([]float64, len(c.Span))
		m.CylLength(ps, k, grad)
		for i, j := range c.Span {
			q0 := q[j]
			q[j] = q0 + h
			m.FK(q, ps)
			lp := m.CylLength(ps, k, nil)
			q[j] = q0 - h
			m.FK(q, ps)
			lm := m.CylLength(ps, k, nil)
			q[j] = q0
			fd := (lp - lm) / (2 * h)
			if math.Abs(fd-grad[i]) > 1e-6 {
				t.Errorf("%s joint %s: analytic %.6f, finite difference %.6f", c.Name, m.Names[j+1], grad[i], fd)
			}
		}
	}
}

func TestSpans(t *testing.T) {
	m := loadBiped(t)
	want := map[string][]string{
		"hip_pitch_l": {"hip_roll_l", "thigh_l"},
		"hip_roll_r":  {"hip_roll_r", "thigh_r"},
		"knee_l":      {"shin_l"},
		"ankle_r":     {"foot_r"},
	}
	for _, c := range m.Cyl {
		w, ok := want[c.Name]
		if !ok {
			continue
		}
		if len(c.Span) != len(w) {
			t.Fatalf("%s spans %d joints, want %v", c.Name, len(c.Span), w)
		}
		for i, j := range c.Span {
			if m.Names[j+1] != w[i] {
				t.Errorf("%s span[%d] = %s, want %s", c.Name, i, m.Names[j+1], w[i])
			}
		}
	}
}

func TestPendulumMassAndGravity(t *testing.T) {
	m := pendulum(t, 1, 0.5, 2)
	ps := m.NewPose()
	q := []float64{math.Pi / 2} // swung forward to horizontal
	m.FK(q, ps)
	M := [][]float64{{0}}
	m.MassMatrix(ps, M)
	if want := 2*0.25 + 1e-6; math.Abs(M[0][0]-want) > 1e-9 {
		t.Errorf("M = %v, want %v", M[0][0], want)
	}
	bias := []float64{0}
	m.RNEA(ps, []float64{0}, []float64{0}, Gravity, bias)
	// Holding torque: weight 2·9.81 N at 0.5 m.
	if math.Abs(bias[0]-9.81) > 1e-9 {
		t.Errorf("holding torque = %v, want 9.81", bias[0])
	}
}

func TestDoublePendulumConservesEnergy(t *testing.T) {
	m := pendulum(t, 2, 0.3, 1)
	n := m.NDof()
	q, qd := []float64{1.0, 0.5}, make([]float64, n)
	ps := m.NewPose()
	M := [][]float64{make([]float64, n), make([]float64, n)}
	bias, zero := make([]float64, n), make([]float64, n)
	energy := func() float64 {
		m.FK(q, ps)
		m.MassMatrix(ps, M)
		e := 0.0
		for i := range qd {
			for j := range qd {
				e += 0.5 * qd[i] * M[i][j] * qd[j]
			}
		}
		for i := range m.Names {
			e -= m.Mass[i] * Gravity.Dot(m.Point(ps, i, m.Com[i]))
		}
		return e
	}
	e0 := energy()
	const dt = 1e-4
	for s := 0; s < 10000; s++ {
		m.FK(q, ps)
		m.RNEA(ps, qd, zero, Gravity, bias)
		m.MassMatrix(ps, M)
		acc := []float64{-bias[0], -bias[1]}
		solveSPD(M, acc)
		for i := range q {
			qd[i] += acc[i] * dt
			q[i] += qd[i] * dt
		}
	}
	if drift := math.Abs(energy()-e0) / math.Abs(e0); drift > 0.01 {
		t.Fatalf("energy drifted %.2f%% over 1 s", 100*drift)
	}
}

func TestWorldStaysFinite(t *testing.T) {
	m := loadBiped(t)
	w := NewWorld(m, false)
	for k := range w.Cmd {
		w.Cmd[k] = math.Sin(float64(k))
	}
	for s := 0; s < 10000; s++ { // 0.5 s
		w.Step(50e-6)
		if !w.Healthy() {
			t.Fatalf("non-finite state at step %d", s)
		}
	}
	for q := range w.Q {
		l := q + 1
		if w.Q[q] < m.Lo[l]-0.3 || w.Q[q] > m.Hi[l]+0.3 {
			t.Errorf("joint %s at %.2f rad, far outside limits [%.2f, %.2f]", m.Names[l], w.Q[q], m.Lo[l], m.Hi[l])
		}
	}
}

func TestReachContainsNeutral(t *testing.T) {
	rows := loadBiped(t).Reach(200)
	if len(rows) == 0 {
		t.Fatal("no reach rows")
	}
	for _, r := range rows {
		if r.Lo > 0 || r.Hi < 0 || r.Lo < r.LimitLo || r.Hi > r.LimitHi {
			t.Errorf("%s/%s: reach [%.3f, %.3f] limits [%.3f, %.3f]", r.Cylinder, r.Joint, r.Lo, r.Hi, r.LimitLo, r.LimitHi)
		}
	}
}

func TestRandomBodiesCompile(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for i := 0; i < 100; i++ {
		b := body.Random(rng)
		if _, err := New(b); err != nil {
			t.Fatalf("random body %d: %v", i, err)
		}
	}
}
