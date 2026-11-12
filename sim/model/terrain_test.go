package model

import (
	"math"
	"testing"

	"mars/sim/geom"
	"mars/sim/terrain"
)

func mustTerrain(t testing.TB, s terrain.Spec) *terrain.Terrain {
	t.Helper()
	tr, err := terrain.Build(s, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	return tr
}

// On a flat terrain the general contact (normal from the terrain) is the
// plain floor's contact: the robot moves the same either way.
func TestFlatTerrainMatchesPlainFloor(t *testing.T) {
	m := loadBiped(t)
	a, b := NewWorld(m, true), NewWorld(m, true)
	b.Ground.Terrain = mustTerrain(t, terrain.Spec{})
	for _, w := range []*World{a, b} {
		w.Ground.Bumps = []Wave{{A: 0.01, Kx: 3, Kz: 1, Phase: 0.3}}
		w.Place(geom.AxisAngle(geom.V3{0, 1, 0}, 0.4), w.Q, 0.1, 0.2)
		w.Base.V = geom.V3{0.2, 0, -0.1}
	}
	for i := 0; i < 8000; i++ {
		a.Step(50e-6)
		b.Step(50e-6)
	}
	// The bumps tilt the terrain contact's normal a little where the
	// plain floor pushes straight up; nothing else differs.
	if d := a.Base.P.Sub(b.Base.P).Norm(); d > 0.01 {
		t.Errorf("pelvis differs by %.4f m after 0.4 s", d)
	}
	b.Ground.Bumps, a.Ground.Bumps = nil, nil
	a2, b2 := NewWorld(m, true), NewWorld(m, true)
	b2.Ground.Terrain = mustTerrain(t, terrain.Spec{})
	for i := 0; i < 8000; i++ {
		a2.Step(50e-6)
		b2.Step(50e-6)
	}
	if d := a2.Base.P.Sub(b2.Base.P).Norm(); d > 1e-9 {
		t.Errorf("flat: pelvis differs by %.2e m after 0.4 s", d)
	}
}

// deepest is the farthest any contact point or collider sphere has sunk
// into the terrain (m, positive = into it).
func deepest(w *World) float64 {
	m, worst := w.M, 0.0
	for _, c := range m.Contacts {
		d, _ := w.Ground.Terrain.Distance(m.Point(w.Pose, c.Link, c.Point), 0.2, 0, 0, 0)
		worst = math.Max(worst, -d)
	}
	for _, pr := range m.Probes {
		c := m.Capsules[pr.Capsule]
		d, _ := w.Ground.Terrain.Distance(m.Point(w.Pose, c.Link, pr.C), 0.2, 0, 0, 0)
		worst = math.Max(worst, c.R-d)
	}
	return worst
}

// A body put down at rest on a slope, on a stair tread, astride a step
// edge, on rough ground and on a box settles there: it never goes
// unstable, never sinks in, and the gentle cases come to rest standing.
// Its joints are only held by the servos, so on a slope it topples like
// a statue unless held: the slope cases are held up by the training
// harness, and then the feet must grip, not slide down the fall line.
func TestSettlesOnTerrain(t *testing.T) {
	m := loadBiped(t)
	cases := []struct {
		name   string
		spec   terrain.Spec
		x, z   float64
		yawDeg float64
		stand  bool // must still be standing and at rest
		held   bool // by the harness
	}{
		{"slope 5° uphill", terrain.Spec{Slope: &terrain.Slope{AngleDeg: 5}}, 0, 0, 0, true, true},
		{"slope 5° across", terrain.Spec{Slope: &terrain.Slope{AngleDeg: 5, HeadingDeg: 90}}, 0, 0, 0, true, true},
		{"slope 12° uphill", terrain.Spec{Slope: &terrain.Slope{AngleDeg: 12}}, 0, 0, 0, true, true},
		{"slope 12° unheld", terrain.Spec{Slope: &terrain.Slope{AngleDeg: 12}}, 0, 0, 0, false, false},
		{"on a platform", terrain.Spec{Stairs: []terrain.Stairs{{At: [2]float64{0.6, 0}, Rise: 0.17, Run: 0.28, Steps: 4, Width: 1.5, Down: true, Platform: 2}}}, 0, 0, 0, true, false},
		{"on a tread", terrain.Spec{Stairs: []terrain.Stairs{{At: [2]float64{-0.6, 0}, Rise: 0.17, Run: 0.6, Steps: 4, Width: 1.5, Landing: 1.5}}}, 0.9, 0, 0, true, false},
		{"astride a 5 cm step", terrain.Spec{Stairs: []terrain.Stairs{{At: [2]float64{0, 0}, HeadingDeg: 90, Rise: 0.05, Run: 1, Steps: 1, Width: 2}}}, 0, 0, 0, false, false},
		{"toes over a nosing", terrain.Spec{Stairs: []terrain.Stairs{{At: [2]float64{-1, 0}, Rise: 0.17, Run: 1.07, Steps: 1, Width: 2}}}, 0, 0, 0, false, false},
		{"rough field", terrain.Spec{NoisePreset: "rough"}, 0, 0, 30, false, false},
		{"gentle field", terrain.Spec{NoisePreset: "gentle"}, 0, 0, 30, true, true},
		{"on a box", terrain.Spec{Objects: []terrain.Object{{Kind: "box", At: [2]float64{0, 0}, Size: [3]float64{1.2, 0.3, 1.2}}}}, 0, 0, 20, true, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			w := NewWorld(m, true)
			w.Ground.Terrain = mustTerrain(t, c.spec)
			w.Place(geom.AxisAngle(geom.V3{0, 1, 0}, c.yawDeg*math.Pi/180), w.Q, c.x, c.z)
			if c.held {
				w.Harness = Harness{Strength: 1, Height: w.Base.P[1], Target: w.Base.P}
			}
			sink, top := 0.0, 0.0
			var p1 geom.V3
			for i := 0; i < int(1.5/50e-6); i++ {
				w.Step(50e-6)
				if i%20 == 0 {
					if !w.Healthy() {
						t.Fatalf("diverged at %.3f s", w.T)
					}
					sink = math.Max(sink, deepest(w))
					top = math.Max(top, w.Base.V.Norm())
				}
				if i == int(1.0/50e-6) {
					p1 = w.Base.P
				}
			}
			drift := w.Base.P.Sub(p1).Norm()
			t.Logf("deepest %.1f mm, fastest pelvis %.2f m/s, end speed %.3f m/s, drift in the last 0.5 s %.1f mm, up %v", 1000*sink, top, w.Base.V.Norm(), 1000*drift, w.Upright())
			if sink > 0.02 {
				t.Errorf("sank %.1f mm into the terrain", 1000*sink)
			}
			if top > 6 { // a fall from standing reaches about 4.5 m/s
				t.Errorf("pelvis reached %.1f m/s: blew up", top)
			}
			if c.stand && (!w.Upright() || w.Base.V.Norm() > 0.05 || drift > 0.01) {
				t.Errorf("did not settle standing: upright %v, speed %.3f m/s, drift %.1f mm", w.Upright(), w.Base.V.Norm(), 1000*drift)
			}
		})
	}
}

// BenchmarkStandingCourse is BenchmarkStanding on a mixed course
// (noise, stairs and objects), for the cost of the terrain contacts.
func BenchmarkStandingCourse(b *testing.B) {
	bp, err := loadBipedB()
	if err != nil {
		b.Fatal(err)
	}
	w := NewWorld(bp, true)
	w.Ground.Terrain = mustTerrain(b, terrain.Spec{NoisePreset: "gentle",
		Stairs:  []terrain.Stairs{{At: [2]float64{1, 0}, Rise: 0.15, Run: 0.3, Steps: 4, Width: 1.5, Landing: 1, Over: true}},
		Scatter: &terrain.Scatter{Count: 20, Min: [2]float64{-5, -5}, Max: [2]float64{5, 5}, Size: [2]float64{0.2, 0.5}, Height: [2]float64{0.1, 0.5}, Clear: 0.6}})
	w.Place(w.Base.R, w.Q, 0, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for s := 0; s < 20; s++ {
			w.Step(50e-6)
		}
	}
}
