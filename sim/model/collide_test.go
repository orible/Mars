package model

import (
	"math"
	"strings"
	"testing"

	"mars/sim/body"
	"mars/sim/geom"
	"mars/sim/hydraulics"
)

// A box whose only collider is a capsule lying along x should rest on
// the floor along its length: no point contacts, the probes carry its
// weight, and it sinks no more than the floor's compliance allows.
func TestCapsuleRestsOnFloor(t *testing.T) {
	b := body.Body{Name: "log", Supply: hydraulics.DefaultSupply(), Links: []body.Link{
		{Name: "box", Mass: 2, Inertia: geom.V3{0.01, 0.01, 0.01},
			Colliders: []body.Capsule{{A: geom.V3{-0.2, 0, 0}, B: geom.V3{0.2, 0, 0}, Radius: 0.05}}},
		{Name: "mast", Parent: "box", Mass: 0.1, Com: geom.V3{0, 0.05, 0}, Inertia: geom.V3{1e-4, 1e-4, 1e-4},
			Joint: &body.Joint{Origin: geom.V3{0, 0.05, 0}, Axis: geom.V3{0, 0, 1}, Limits: [2]float64{-0.5, 0.5}, Damping: 0.05}},
	}, Cylinders: []body.Cylinder{{Name: "c", Base: body.Anchor{Link: "box", Point: geom.V3{0.05, 0, 0}},
		Rod: body.Anchor{Link: "mast", Point: geom.V3{0, 0.1, 0}}, RetractedLength: 0.13, Params: cylParams()}}}
	m, err := New(b)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Probes) < 3 {
		t.Fatalf("a 40 cm capsule of radius 5 cm got %d floor probes", len(m.Probes))
	}
	w := NewWorld(m, true)
	if math.Abs(w.Base.P[1]-0.05) > 1e-9 {
		t.Fatalf("starts with the box at %.4f m, want the capsule's radius 0.05 m", w.Base.P[1])
	}
	w.Base.P[1] += 0.03 // drop it from 3 cm
	w.Refresh()
	for s := 0; s < 20000; s++ { // 1 s
		w.Step(50e-6)
		if !w.Healthy() {
			t.Fatalf("non-finite at %.3f s", w.T)
		}
	}
	load := 0.0
	for _, f := range w.ProbeF {
		load += f[1]
	}
	weight := 2.1 * 9.81
	sink := 0.05 - w.Base.P[1]
	t.Logf("box at %.4f m (sank %.2f mm), probes carry %.2f of %.2f N, up %v", w.Base.P[1], 1000*sink, load, weight, w.Up())
	if math.Abs(load-weight) > 0.02*weight {
		t.Errorf("probes carry %.2f N, want the weight %.2f N", load, weight)
	}
	if sink < 0 || sink > 0.002 {
		t.Errorf("sank %.2f mm into the floor", 1000*sink)
	}
	if w.Up()[1] < 0.999 {
		t.Errorf("rolled or tipped: up %v", w.Up())
	}
}

// swing is two arms hanging from a fixed beam (which has no collider, so
// the arms collide with each other): a held one at x = 0 and a free one at
// x = 0.3 that starts raised away from it, so gravity swings it into the
// held one.
func swing(t *testing.T, colliders bool) *Model {
	t.Helper()
	capsule := func() []body.Capsule {
		if !colliders {
			return nil
		}
		return []body.Capsule{{A: geom.V3{0, -0.05, 0}, B: geom.V3{0, -0.3, 0}, Radius: 0.03}}
	}
	arm := func(name string, x float64, lim float64) body.Link {
		return body.Link{Name: name, Parent: "beam", Mass: 1, Com: geom.V3{0, -0.2, 0}, Inertia: geom.V3{0.005, 0.0005, 0.005},
			Joint:     &body.Joint{Origin: geom.V3{x, 0, 0}, Axis: geom.V3{0, 0, 1}, Limits: [2]float64{-lim, lim}},
			Colliders: capsule()}
	}
	b := body.Body{Name: "swing", Supply: hydraulics.DefaultSupply(), Links: []body.Link{
		{Name: "beam", Mass: 1, Inertia: geom.V3{0.01, 0.01, 0.01}},
		arm("held", 0, 0.01), arm("free", 0.3, 1.5),
	}, Cylinders: []body.Cylinder{{Name: "c", Base: body.Anchor{Link: "beam", Point: geom.V3{0.05, 0, 0}},
		Rod: body.Anchor{Link: "held", Point: geom.V3{0, -0.1, 0}}, RetractedLength: 0.1, Params: cylParams()}}}
	m, err := New(b)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// The free arm should stop against the held one instead of swinging
// through it, and only with colliders.
func TestCapsulesStopEachOther(t *testing.T) {
	closest := func(colliders bool) (float64, int) {
		m := swing(t, colliders)
		w := NewWorld(m, false)
		w.Q[1] = 1.2 // tip out at +x, away from the held arm
		w.Refresh()
		lo, hits := math.Inf(1), 0
		for s := 0; s < 20000; s++ { // 1 s: one swing across
			w.Step(50e-6)
			if !w.Healthy() {
				t.Fatalf("non-finite at %.3f s", w.T)
			}
			held := []geom.V3{m.Point(w.Pose, 1, geom.V3{0, -0.05, 0}), m.Point(w.Pose, 1, geom.V3{0, -0.3, 0})}
			free := []geom.V3{m.Point(w.Pose, 2, geom.V3{0, -0.05, 0}), m.Point(w.Pose, 2, geom.V3{0, -0.3, 0})}
			p, q := geom.SegmentClosest(held[0], held[1], free[0], free[1])
			lo = math.Min(lo, q.Sub(p).Norm())
			hits += len(w.SelfHits)
		}
		return lo, hits
	}
	ghost, hits0 := closest(false)
	solid, hits := closest(true)
	t.Logf("closest axis distance: %.1f mm without colliders, %.1f mm with (contact on %d steps)", 1000*ghost, 1000*solid, hits)
	if ghost > 0.04 || hits0 > 0 {
		t.Fatalf("without colliders the arms should pass within 4 cm (got %.1f mm)", 1000*ghost)
	}
	if hits == 0 || solid < 0.06-0.008 {
		t.Errorf("with 3 cm capsules the axes came within %.1f mm (want no closer than about 60 mm)", 1000*solid)
	}
}

// Legs crossed into each other should be pushed apart on the full human.
func TestHumanLegsDoNotCross(t *testing.T) {
	m := loadBiped(t)
	if len(m.Pairs) == 0 {
		t.Fatal("the human has no self-collision pairs")
	}
	w := NewWorld(m, false)
	idx := map[string]int{}
	for q := 0; q < m.NDof(); q++ {
		idx[m.Names[q+1]] = q
	}
	// Both hips adducted to their limits would cross the thighs by far.
	w.Q[idx["hip_roll_l"]], w.Q[idx["hip_roll_r"]] = m.Lo[idx["hip_roll_l"]+1], m.Hi[idx["hip_roll_r"]+1]
	w.Refresh()
	thigh := func(n string) int {
		for k, c := range m.Capsules {
			if m.Names[c.Link] == n {
				return k
			}
		}
		t.Fatalf("%s has no capsule", n)
		return 0
	}
	l, r := thigh("thigh_l"), thigh("thigh_r")
	gap := func() float64 {
		p, q := geom.SegmentClosest(w.CapA[l], w.CapB[l], w.CapA[r], w.CapB[r])
		return q.Sub(p).Norm() - m.Capsules[l].R - m.Capsules[r].R
	}
	before := gap()
	for s := 0; s < 10000; s++ { // 0.5 s, valves closed
		w.Step(50e-6)
		if !w.Healthy() {
			t.Fatalf("non-finite at %.3f s", w.T)
		}
	}
	t.Logf("thigh gap %.1f mm crossed → %.1f mm after 0.5 s", 1000*before, 1000*gap())
	if before > -0.02 {
		t.Fatalf("test pose should cross the thighs (gap %.1f mm)", 1000*before)
	}
	if g := gap(); g < -0.01 {
		t.Errorf("thighs still overlap by %.1f mm", -1000*g)
	}
}

func TestOverlappingCollidersRejected(t *testing.T) {
	b, err := body.Load("../bodies/biped.json")
	if err != nil {
		t.Fatal(err)
	}
	i := b.LinkIndex("forearm_l")
	b.Links[i].Colliders[0].Radius = 0.08 // into the thigh
	if err := b.Validate(); err == nil || !strings.Contains(err.Error(), "overlap") {
		t.Errorf("fat forearm accepted: %v", err)
	}
}
