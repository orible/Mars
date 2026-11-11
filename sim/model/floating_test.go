package model

import (
	"math"
	"strings"
	"testing"

	"mars/sim/body"
	"mars/sim/geom"
	"mars/sim/hydraulics"
)

// momentum returns total linear momentum and angular momentum about the
// world origin.
func momentum(w *World) (geom.V3, geom.V3) {
	m, ps := w.M, w.Pose
	var lin, ang geom.V3
	for i := range m.Names {
		c := m.Point(ps, i, m.Com[i])
		v := m.PointVelocity(ps, w.Base.V, w.Base.W, w.Qd, i, c)
		om := w.Base.W
		for j := i; j > 0; j = m.Parent[j] {
			om = om.Add(ps.Z[j].Scale(w.Qd[j-1]))
		}
		p := v.Scale(m.Mass[i])
		lin = lin.Add(p)
		ang = ang.Add(c.Cross(p)).Add(geom.InertiaMul(ps.R[i], m.Inertia[i], om))
	}
	return lin, ang
}

func TestFloatingMassMatrix(t *testing.T) {
	m := loadBiped(t)
	w := NewWorld(m, true)
	for q := range w.Q {
		w.Q[q] = 0.3 * math.Sin(float64(q))
	}
	w.Base.R = geom.AxisAngle(geom.V3{1, 2, 3}, 0.4)
	w.refresh()
	n := m.NDof()
	M := make([][]float64, 6+n)
	for i := range M {
		M[i] = make([]float64, 6+n)
	}
	m.MassMatrixFloating(w.Pose, M)
	fixed := make([][]float64, n)
	for i := range fixed {
		fixed[i] = make([]float64, n)
	}
	m.MassMatrix(w.Pose, fixed)
	_, mass := w.CoG()
	for i := 0; i < 6+n; i++ {
		for j := 0; j < 6+n; j++ {
			if d := math.Abs(M[i][j] - M[j][i]); d > 1e-9 {
				t.Fatalf("M not symmetric at %d,%d: %g vs %g", i, j, M[i][j], M[j][i])
			}
			if i >= 6 && j >= 6 && math.Abs(M[i][j]-fixed[i-6][j-6]) > 1e-12 {
				t.Fatalf("joint block differs from fixed-base M at %d,%d", i, j)
			}
		}
		if i < 3 && math.Abs(M[i][i]-mass) > 1e-9 {
			t.Errorf("M[%d][%d] = %g, want total mass %g", i, i, M[i][i], mass)
		}
	}
}

// With no gravity and no ground, cylinder forces, joint limits and joint
// damping are all internal, so momentum must not change.
func TestFreeFlightConservesMomentum(t *testing.T) {
	g := Gravity
	Gravity = geom.V3{}
	defer func() { Gravity = g }()

	m := loadBiped(t)
	w := NewWorld(m, true)
	w.Base.P = geom.V3{0, 2, 0}
	w.Base.V = geom.V3{0.3, 0.1, -0.2}
	w.Base.W = geom.V3{0.5, -0.4, 0.8}
	for k := range w.Cmd {
		w.Cmd[k] = math.Sin(float64(3 * k))
	}
	w.refresh()
	l0, a0 := momentum(w)
	for s := 0; s < 4000; s++ { // 0.2 s
		w.Step(50e-6)
	}
	if !w.Healthy() {
		t.Fatal("non-finite state")
	}
	l1, a1 := momentum(w)
	t.Logf("linear %v → %v, angular %v → %v, joints %v", l0, l1, a0, a1, w.Q)
	if d := l1.Sub(l0).Norm(); d > 1e-3*l0.Norm() {
		t.Errorf("linear momentum %v → %v", l0, l1)
	}
	// First-order integration of fast joint motion leaves a small drift.
	if d := a1.Sub(a0).Norm(); d > 0.02*a0.Norm() {
		t.Errorf("angular momentum %v → %v", a0, a1)
	}
}

// sled is a flat box on four contact points with a small hinged mast so
// that the spec has a joint and a cylinder.
func sled(t *testing.T) *Model {
	t.Helper()
	b := body.Body{Name: "sled", Supply: hydraulics.DefaultSupply(), Links: []body.Link{
		{Name: "box", Mass: 2, Com: geom.V3{0, 0.02, 0}, Inertia: geom.V3{0.01, 0.01, 0.01},
			Contacts: []geom.V3{{-0.1, 0, -0.1}, {-0.1, 0, 0.1}, {0.1, 0, -0.1}, {0.1, 0, 0.1}}},
		{Name: "mast", Parent: "box", Mass: 0.1, Com: geom.V3{0, 0.05, 0}, Inertia: geom.V3{1e-4, 1e-4, 1e-4},
			Joint: &body.Joint{Origin: geom.V3{0, 0.05, 0}, Axis: geom.V3{0, 0, 1}, Limits: [2]float64{-0.5, 0.5}, Damping: 0.05}},
	}, Cylinders: []body.Cylinder{{Name: "c", Base: body.Anchor{Link: "box", Point: geom.V3{0.05, 0, 0}},
		Rod: body.Anchor{Link: "mast", Point: geom.V3{0, 0.1, 0}}, RetractedLength: 0.13, Params: cylParams()}}}
	m, err := New(b)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// A sled pushed at 1 m/s should slide v²/(2μg) before friction stops it.
func TestSledSlidesToFrictionDistance(t *testing.T) {
	m := sled(t)
	w := NewWorld(m, true)
	w.Base.P[1] -= 2.1 * 9.81 / (4 * w.Ground.K) // start at static sink
	w.refresh()
	for s := 0; s < 2000; s++ { // settle 0.1 s
		w.Step(50e-6)
	}
	x0 := w.Base.P[0]
	w.Base.V = geom.V3{1, 0, 0}
	for s := 0; s < 40000; s++ { // 2 s
		w.Step(50e-6)
	}
	want := 1 / (2 * w.Ground.Mu * 9.81)
	got := w.Base.P[0] - x0
	t.Logf("slid %.4f m (Coulomb predicts %.4f m), final speed %.2e m/s", got, want, w.Base.V.Norm())
	if math.Abs(got-want) > 0.1*want {
		t.Errorf("slid %.4f m, want %.4f m", got, want)
	}
	if v := w.Base.V.Norm(); v > 1e-3 {
		t.Errorf("still moving at %.4f m/s", v)
	}
}

// The harness should carry even a rigid (valves closed) biped along
// upright, its feet dragging.
func TestHarnessCarriesUpright(t *testing.T) {
	m := loadBiped(t)
	w := NewWorld(m, true)
	w.Harness = Harness{Strength: 1, Vel: geom.V3{0.3, 0, 0}, Height: w.Base.P[1], Target: w.Base.P}
	var mid geom.V3
	for s := 0; s < 40000; s++ { // 2 s
		w.Step(50e-6)
		if w.Fallen() || !w.Healthy() {
			t.Fatalf("fell while held at t=%.2f s", w.T)
		}
		if s == 20000 {
			mid = w.Base.P
		}
	}
	// The dragging feet stick and slip, so judge speed over the last second.
	avg := w.Base.P.Sub(mid)
	t.Logf("carried to %v, last-second travel %v, up %v, hands %v N", w.Base.P, avg, w.Up(), w.Harness.Force)
	if w.Base.P[0] < 0.45 || math.Abs(avg[0]-0.3) > 0.05 || math.Abs(avg[2]) > 0.05 {
		t.Errorf("pelvis at x=%.2f m, last second %.2f m forward, %.2f m sideways; want about 0.3 m/s straight", w.Base.P[0], avg[0], avg[2])
	}
	if w.Up()[1] < 0.98 {
		t.Errorf("tilted while held: up %v", w.Up())
	}
}

// Knee lock: with the knees pushed into their hyperextension stops, the
// stops (not the cylinders) carry the knee load.
func TestKneesLockAtExtensionStop(t *testing.T) {
	m := loadBiped(t)
	w := NewWorld(m, true)
	knees := []int{}
	for q := range w.Q {
		if strings.HasPrefix(m.Names[q+1], "shin") {
			knees = append(knees, q)
			w.Q[q] = m.Hi[q+1] // start at the lock
		}
	}
	w.Base.P[1] = w.StandHeight() + 0.002
	w.refresh()
	for k := range m.Cyl { // re-seat cylinders on the new pose
		w.Cyl[k].X = w.Len[k] - m.Cyl[k].Retracted
	}
	// Knee cylinders free-wheel (valve wide open both ways is not
	// modelled), so instead command them to extend the knee hard.
	for k, c := range m.Cyl {
		if strings.HasPrefix(c.Name, "knee") {
			w.Cmd[k] = extendSign(t, w, k)
		}
	}
	for s := 0; s < 20000; s++ {
		w.Step(50e-6)
		if w.Fallen() {
			t.Fatalf("fell at %.2f s", w.T)
		}
	}
	for _, q := range knees {
		over := (w.Q[q] - m.Hi[q+1]) * 180 / math.Pi
		t.Logf("%s at %.2f° (stop %.2f°, %.2f° into it)", m.Names[q+1], w.Q[q]*180/math.Pi, m.Hi[q+1]*180/math.Pi, over)
		if over > 1 || over < -0.5 {
			t.Errorf("%s not held at its stop: %.2f° past it", m.Names[q+1], over)
		}
	}
}

// extendSign returns the valve command sign that straightens the joint
// cylinder k spans (its moment arm tells which way extension turns it).
func extendSign(t *testing.T, w *World, k int) float64 {
	t.Helper()
	if w.DLdq[k][0] > 0 {
		return 1 // lengthening raises q (towards extension)
	}
	return -1
}

// With every valve closed the cylinders lock the joints, so the biped
// should stand where it was put: weight carried by the feet, no drift.
func TestBipedStandsWithValvesClosed(t *testing.T) {
	m := loadBiped(t)
	w := NewWorld(m, true)
	h0 := w.Base.P[1]
	for s := 0; s < 20000; s++ { // 1 s
		w.Step(50e-6)
		if w.Fallen() || !w.Healthy() {
			t.Fatalf("fell at t=%.3f s: pelvis %v, up %v", w.T, w.Base.P, w.Up())
		}
	}
	_, mass := w.CoG()
	load := 0.0
	for _, f := range w.ContactF {
		load += f[1]
	}
	t.Logf("load %.2f N of %.2f N weight, height change %.2f mm, drift %.3f mm, up %v",
		load, mass*9.81, 1000*(w.Base.P[1]-h0), 1000*math.Hypot(w.Base.P[0], w.Base.P[2]), w.Up())
	if math.Abs(load-mass*9.81) > 0.05*mass*9.81 {
		t.Errorf("feet carry %.1f N, weight is %.1f N", load, mass*9.81)
	}
	if d := w.Base.P[1] - h0; d < -0.01 || d > 0.001 {
		t.Errorf("pelvis height changed by %.4f m", d)
	}
	if d := math.Hypot(w.Base.P[0], w.Base.P[2]); d > 0.005 {
		t.Errorf("pelvis drifted %.4f m sideways", d)
	}
	if tilt := math.Acos(math.Min(1, w.Up()[1])); tilt > 0.05 {
		t.Errorf("pelvis tilted %.3f rad", tilt)
	}
}
