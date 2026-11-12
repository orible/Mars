package model

import (
	"math"
	"testing"

	"mars/sim/geom"
)

// The plantar fascia is a windlass: bending the toes up, or flattening
// the arch, stretches it.
func TestWindlass(t *testing.T) {
	m := loadBiped(t)
	w := NewWorld(m, false)
	idx := map[string]int{}
	for q := 0; q < m.NDof(); q++ {
		idx[m.Names[q+1]] = q
	}
	tn := -1
	for i, td := range m.Tendons {
		if td.Name == "plantar_l" {
			tn = i
		}
	}
	if tn < 0 {
		t.Fatal("no plantar_l tendon")
	}
	length := func(joint string, v float64) float64 {
		for q := range w.Q {
			w.Q[q] = 0
		}
		w.Q[idx[joint]] = v
		w.Refresh()
		return w.TendonLen[tn]
	}
	l0 := length("toes_l", 0)
	if lt := length("toes_l", 0.6); lt-l0 < 0.002 {
		t.Errorf("toes up 0.6 rad stretched the fascia only %.2f mm", 1000*(lt-l0))
	}
	if la := length("midfoot_l", 0.15); la-l0 < 0.002 {
		t.Errorf("flattening the arch 0.15 rad stretched the fascia only %.2f mm", 1000*(la-l0))
	}
	if m.Tendons[tn].Rest >= l0 {
		t.Errorf("fascia slack in the neutral pose: rest %.4f, length %.4f", m.Tendons[tn].Rest, l0)
	}
}

// On the test stand, the stirrup's one valve tilts that foot's sole in
// and out and leaves the other foot alone.
func TestStirrupTiltsFoot(t *testing.T) {
	m := loadBiped(t)
	k, heel, other := -1, -1, -1
	for i, c := range m.Cyl {
		if c.Name == "stirrup_l" {
			k = i
		}
	}
	for q := 0; q < m.NDof(); q++ {
		switch m.Names[q+1] {
		case "heel_l":
			heel = q
		case "heel_r":
			other = q
		}
	}
	reached := map[float64]float64{}
	for _, cmd := range []float64{0.6, -0.6} {
		w := NewWorld(m, false)
		w.Cmd[k] = cmd
		for s := 0; s < 10000; s++ { // 0.5 s
			w.Step(50e-6)
		}
		reached[cmd] = w.Q[heel]
		t.Logf("force %+.1f: heel_l %.1f°, heel_r %.1f°", cmd, w.Q[heel]*180/math.Pi, w.Q[other]*180/math.Pi)
		if math.Abs(w.Q[other]) > 0.01 {
			t.Errorf("force %+.1f moved the other foot %.3f rad", cmd, w.Q[other])
		}
	}
	if math.Abs(reached[0.6]-reached[-0.6]) < 0.3 || reached[0.6]*reached[-0.6] >= 0 {
		t.Errorf("stirrup should tilt the foot both ways: %+.2f and %+.2f rad", reached[0.6], reached[-0.6])
	}
}

// Standing on a bumpy, soft, patchy floor: settles without sinking or
// blowing up, and the floor carries the weight.
func TestStandsOnBumpyFloor(t *testing.T) {
	m := loadBiped(t)
	w := NewWorld(m, true)
	w.Ground.Bumps = []Wave{{A: 0.01, Kx: 9, Kz: 4, Phase: 1}, {A: 0.006, Kx: -3, Kz: 11, Phase: 2}}
	w.Ground.MuWaves = []Wave{{A: 0.2, Kx: 2, Kz: 1}}
	w.Ground.K *= 0.4
	w.Place(geom.Identity(), make([]float64, m.NDof()), 0.3, -0.2)
	for s := 0; s < 20000; s++ {
		w.Step(50e-6)
		if !w.Healthy() {
			t.Fatalf("non-finite at %.3f s", w.T)
		}
	}
	load, deepest := 0.0, 0.0
	for i, c := range m.Contacts {
		load += w.ContactF[i][1]
		p := m.Point(w.Pose, c.Link, c.Point)
		deepest = math.Max(deepest, w.Ground.Height(p[0], p[2])-p[1])
	}
	_, mass := w.CoG()
	t.Logf("floor carries %.0f of %.0f N, deepest point %.1f mm into it, upright %v", load, mass*9.81, 1000*deepest, w.Upright())
	if math.Abs(load-mass*9.81) > 0.1*mass*9.81 {
		t.Errorf("floor carries %.0f N of %.0f", load, mass*9.81)
	}
	if deepest > 0.01 {
		t.Errorf("a contact sank %.1f mm into the floor", 1000*deepest)
	}
}
