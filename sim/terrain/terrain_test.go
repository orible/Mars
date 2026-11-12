package terrain

import (
	"math"
	"reflect"
	"testing"

	"mars/sim/geom"
)

func build(t *testing.T, s Spec, seed int64, keep ...[2]float64) *Terrain {
	t.Helper()
	tr, err := Build(s, seed, keep)
	if err != nil {
		t.Fatal(err)
	}
	return tr
}

func near(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// The noise slope is exact (matches finite differences), its height stays
// within ±Amplitude, and it really is uneven.
func TestNoiseGradientAndRange(t *testing.T) {
	for name, p := range NoisePresets {
		tr := build(t, Spec{Noise: &p}, 42)
		lo, hi := math.Inf(1), math.Inf(-1)
		for i := 0; i < 2000; i++ {
			x, z := float64(i%50)*0.173-4, float64(i/50)*0.211-4
			h, gx, gz := tr.Field(x, z)
			lo, hi = math.Min(lo, h), math.Max(hi, h)
			const e = 1e-6
			hx, _, _ := tr.Field(x+e, z)
			hz, _, _ := tr.Field(x, z+e)
			if !near((hx-h)/e, gx, 1e-3) || !near((hz-h)/e, gz, 1e-3) {
				t.Fatalf("%s: slope at (%.2f, %.2f) is (%.4f, %.4f), finite difference (%.4f, %.4f)", name, x, z, gx, gz, (hx-h)/e, (hz-h)/e)
			}
		}
		if lo < -p.Amplitude || hi > p.Amplitude {
			t.Errorf("%s: height %.3f..%.3f outside ±%.3f", name, lo, hi, p.Amplitude)
		}
		if hi-lo < 0.5*p.Amplitude {
			t.Errorf("%s: height only spans %.3f m for amplitude %.3f", name, hi-lo, p.Amplitude)
		}
	}
}

// The same seed gives the same ground; another seed a different one; a
// spec's own seed wins over the episode's.
func TestNoiseDeterminism(t *testing.T) {
	s := Spec{NoisePreset: "rough"}
	a, b, c := build(t, s, 7), build(t, s, 7), build(t, s, 8)
	same, diff := true, false
	for i := 0; i < 100; i++ {
		x, z := float64(i)*0.37, float64(i)*-0.19
		if a.Height(x, z) != b.Height(x, z) {
			same = false
		}
		if a.Height(x, z) != c.Height(x, z) {
			diff = true
		}
	}
	if !same || !diff {
		t.Errorf("same seed same ground: %v, different seed different ground: %v", same, diff)
	}
	s.Seed = 99
	if d, e := build(t, s, 1), build(t, s, 2); d.Height(1.3, 0.7) != e.Height(1.3, 0.7) {
		t.Error("a fixed spec seed still changed with the episode seed")
	}
}

// A slope is a plane rising towards its heading; the normal leans back
// down the fall line and the distance is measured along it.
func TestSlopeHeightAndNormal(t *testing.T) {
	for _, hd := range []float64{0, 90, 210} {
		tr := build(t, Spec{Slope: &Slope{AngleDeg: 10, HeadingDeg: hd}}, 1)
		c, s := heading(hd)
		g := math.Tan(rad(10))
		for _, u := range []float64{-2, 0.5, 3} {
			x, z := u*c, -u*s
			if h := tr.Height(x, z); !near(h, g*u, 1e-9) {
				t.Errorf("heading %.0f: height at u=%.1f is %.4f, want %.4f", hd, u, h, g*u)
			}
			// A point 0.1 m above the plane along its normal.
			n0 := geom.V3{-g * c, 1, g * s}.Unit()
			p := geom.V3{x, g * u, z}.Add(n0.Scale(0.1))
			d, n := tr.Distance(p, 1, 0, 0, 0)
			if !near(d, 0.1, 1e-9) || n.Sub(n0).Norm() > 1e-9 {
				t.Errorf("heading %.0f: distance %.4f normal %v, want 0.1 and %v", hd, d, n, n0)
			}
			if n[1] < math.Cos(rad(10))-1e-9 || n.Dot(geom.V3{c, 0, -s}) > 0 {
				t.Errorf("heading %.0f: normal %v does not lean downhill", hd, n)
			}
		}
	}
}

// A ramp rises at its angle to the landing, stays level, and comes down
// again; it is continuous throughout.
func TestRampProfile(t *testing.T) {
	tr := build(t, Spec{Ramps: []Ramp{{At: [2]float64{1, 0}, AngleDeg: 10, Rise: 0.4, Landing: 1.5}}}, 1)
	up := 0.4 / math.Tan(rad(10))
	for _, c := range []struct{ x, h float64 }{{0.5, 0}, {1 + up/2, 0.2}, {1 + up + 0.75, 0.4}, {1 + up + 1.5 + up/2, 0.2}, {1 + 2*up + 1.6, 0}} {
		if h := tr.Height(c.x, 3); !near(h, c.h, 1e-9) {
			t.Errorf("ramp height at x=%.2f is %.4f, want %.4f", c.x, h, c.h)
		}
	}
	prev := tr.Height(0, 0)
	for x := 0.0; x < 8; x += 0.001 {
		h := tr.Height(x, 0)
		if math.Abs(h-prev) > 0.001*math.Tan(rad(10))+1e-9 {
			t.Fatalf("ramp jumps at x=%.3f", x)
		}
		prev = h
	}
}

// Stairs: each tread is at its step's height, the risers stand where the
// runs say, and a point inside a step is pushed out of the nearest open
// face: up through the tread, or back out of the riser it ran into.
func TestStairsGeometry(t *testing.T) {
	for _, hd := range []float64{0, 90, -30} {
		st := Stairs{At: [2]float64{1, 0.5}, HeadingDeg: hd, Rise: 0.17, Run: 0.28, Steps: 5, Width: 1.2, Landing: 1}
		tr := build(t, Spec{Stairs: []Stairs{st}}, 1)
		c, s := heading(hd)
		at := func(u, w float64) (float64, float64) { // flight frame → world x, z
			return st.At[0] + u*c + w*s, st.At[1] - u*s + w*c
		}
		for k := 0; k < st.Steps; k++ {
			x, z := at((float64(k)+0.5)*st.Run, 0.3)
			if h := tr.Height(x, z); !near(h, float64(k+1)*st.Rise, 1e-9) {
				t.Errorf("heading %.0f: tread %d at %.4f, want %.4f", hd, k, h, float64(k+1)*st.Rise)
			}
		}
		if x, z := at(-0.01, 0); tr.Height(x, z) != 0 {
			t.Errorf("heading %.0f: floor before the first riser is not flat", hd)
		}
		if x, z := at(0.5, st.Width/2+0.01); tr.Height(x, z) != 0 {
			t.Errorf("heading %.0f: stairs wider than %.2f m", hd, st.Width)
		}
		if x, z := at(float64(st.Steps)*st.Run+0.9, 0); !near(tr.Height(x, z), 5*0.17, 1e-9) {
			t.Errorf("heading %.0f: landing missing", hd)
		}
		fwd := geom.V3{c, 0, -s}
		// A toe 1 cm into the third riser, 5 cm above the second tread.
		x, z := at(2*st.Run+0.01, 0)
		d, n := tr.Distance(geom.V3{x, 2*st.Rise + 0.05, z}, 0.1, 0, 0, 0)
		if !near(d, -0.01, 1e-9) || n.Sub(fwd.Scale(-1)).Norm() > 1e-9 {
			t.Errorf("heading %.0f: toe in a riser: distance %.4f normal %v, want −0.01 and %v", hd, d, n, fwd.Scale(-1))
		}
		// A heel 3 mm into a tread, 5 cm in front of the riser above it.
		x, z = at(3*st.Run-0.05, 0)
		d, n = tr.Distance(geom.V3{x, 3*st.Rise - 0.003, z}, 0.1, 0, 0, 0)
		if !near(d, -0.003, 1e-9) || n.Sub(geom.V3{0, 1, 0}).Norm() > 1e-9 {
			t.Errorf("heading %.0f: heel on a tread: distance %.4f normal %v", hd, d, n)
		}
		// Deep in a step right beside the riser of the next one: the face
		// shared with the taller step is not an exit, so it goes up.
		x, z = at(2*st.Run-0.001, 0)
		if _, n = tr.Distance(geom.V3{x, 2*st.Rise - 0.02, z}, 0.1, 0, 0, 0); n.Sub(geom.V3{0, 1, 0}).Norm() > 1e-9 {
			t.Errorf("heading %.0f: pushed through the seam between steps: normal %v", hd, n)
		}
		// Above a tread, the distance is the height above it; beyond the
		// nosing, the distance is to the edge.
		x, z = at(0.1, 0)
		if d, _ = tr.Distance(geom.V3{x, st.Rise + 0.04, z}, 0.1, 0, 0, 0); !near(d, 0.04, 1e-9) {
			t.Errorf("heading %.0f: 4 cm above the first tread measured %.4f", hd, d)
		}
		x, z = at(-0.03, 0)
		if d, n = tr.Distance(geom.V3{x, st.Rise + 0.04, z}, 0.1, 0, 0, 0); !near(d, 0.05, 1e-9) || n[1] < 0.7 || n.Dot(fwd) > -0.5 {
			t.Errorf("heading %.0f: in front of the first nosing: distance %.4f normal %v, want 0.05 diagonal", hd, d, n)
		}
	}
}

// Going down, the robot starts on the platform and the steps fall away
// in front of it.
func TestStairsDown(t *testing.T) {
	tr := build(t, Spec{Stairs: []Stairs{{At: [2]float64{0.6, 0}, Rise: 0.15, Run: 0.3, Steps: 4, Width: 1.5, Down: true, Platform: 2}}}, 1)
	want := []struct{ x, h float64 }{{-1, 0.6}, {0, 0.6}, {0.7, 0.45}, {1.0, 0.3}, {1.3, 0.15}, {1.6, 0}, {-1.5, 0}}
	for _, c := range want {
		if h := tr.Height(c.x, 0); !near(h, c.h, 1e-9) {
			t.Errorf("height at x=%.2f is %.3f, want %.3f", c.x, h, c.h)
		}
	}
	// The riser below the first edge faces forward (+x).
	if d, n := tr.Distance(geom.V3{0.89, 0.40, 0}, 0.1, 0, 0, 0); !near(d, -0.01, 1e-9) || n.Sub(geom.V3{-1, 0, 0}).Norm() > 1e-9 && n.Sub(geom.V3{1, 0, 0}).Norm() > 1e-9 {
		t.Errorf("inside the second step near its edge: %.4f %v", d, n)
	}
	if _, n := tr.Distance(geom.V3{0.91, 0.40, 0}, 0.1, 0, 0, 0); n[0] < 0.99 {
		t.Errorf("toe below the second step's edge pushed along %v, want +x out of the riser", n)
	}
}

// Outside every solid the normal is the gradient of the distance, so
// spheres slide off edges and corners the right way.
func TestSolidDistanceGradient(t *testing.T) {
	tr := build(t, Spec{
		Stairs:  []Stairs{{At: [2]float64{1, 0}, HeadingDeg: 20, Rise: 0.17, Run: 0.28, Steps: 3, Width: 1, Landing: 0.5}},
		Objects: []Object{{Kind: "box", At: [2]float64{-1, 1}, YawDeg: 35, Size: [3]float64{0.4, 0.3, 0.2}}, {Kind: "cylinder", At: [2]float64{-1, -1}, Size: [3]float64{0.3, 0.5, 0.3}}},
	}, 1)
	pts := []geom.V3{{-1.3, 0.2, 1.1}, {-0.9, 0.35, 1.05}, {-1.2, 0.3, -1.1}, {-1, 0.6, -1}, {0.95, 0.2, 0.1}, {1.4, 0.4, 0.6}, {1.3, 0.45, 0.2}}
	for _, p := range pts {
		d, n := tr.Distance(p, 1, 0, 0, 0)
		if d <= 0 {
			t.Fatalf("point %v is not outside (d=%.3f)", p, d)
		}
		const e = 1e-6
		var g geom.V3
		for k := 0; k < 3; k++ {
			q := p
			q[k] += e
			dq, _ := tr.Distance(q, 1, 0, 0, 0)
			g[k] = (dq - d) / e
		}
		if g.Sub(n).Norm() > 1e-4 {
			t.Errorf("at %v: normal %v, distance gradient %v", p, n, g)
		}
	}
	// Box and cylinder tops.
	if h := tr.Height(-1, 1); !near(h, 0.3, 1e-9) {
		t.Errorf("box top %.3f", h)
	}
	if h := tr.Height(-1, -1.14); !near(h, 0.5, 1e-9) {
		t.Errorf("cylinder top %.3f", h)
	}
	if h := tr.Height(-1, -1.16); h != 0 {
		t.Errorf("cylinder wider than its radius: %.3f", h)
	}
}

// Scattered objects keep clear of the start, the objectives, the stairs
// and each other, and the same seed scatters them the same way.
func TestScatterKeepsClear(t *testing.T) {
	keep := [][2]float64{{3, 1}, {5, -2}}
	s := Spec{
		Stairs:  []Stairs{{At: [2]float64{1, 2}, Rise: 0.15, Run: 0.3, Steps: 3, Width: 1}},
		Scatter: &Scatter{Count: 40, Min: [2]float64{-4, -4}, Max: [2]float64{6, 4}, Size: [2]float64{0.1, 0.5}, Height: [2]float64{0.1, 0.8}, Clear: 0.8, Gap: 0.1},
	}
	for seed := int64(1); seed <= 20; seed++ {
		tr := build(t, s, seed, keep...)
		objs := tr.Spec.Objects
		if len(objs) < 30 {
			t.Errorf("seed %d: only %d of 40 objects placed", seed, len(objs))
		}
		for i, o := range objs {
			r := 0.5 * math.Hypot(o.Size[0], o.Size[2])
			for _, k := range append(keep, [2]float64{0, 0}) {
				if math.Hypot(o.At[0]-k[0], o.At[1]-k[1]) < 0.8+r {
					t.Errorf("seed %d: object %d at %v within the clear radius of %v", seed, i, o.At, k)
				}
			}
			if math.Abs(o.At[1]-2) < 0.5+r && o.At[0] > 1-r && o.At[0] < 1.9+r {
				t.Errorf("seed %d: object %d on the stairs", seed, i)
			}
			for j := 0; j < i; j++ {
				q := objs[j]
				if footprintDist(q, o.At[0], o.At[1]) < r+0.1-1e-9 {
					t.Errorf("seed %d: objects %d and %d overlap", seed, i, j)
				}
			}
			if o.Seed == 0 {
				t.Errorf("seed %d: object %d has no appearance seed", seed, i)
			}
		}
	}
	a, b := build(t, s, 5, keep...), build(t, s, 5, keep...)
	if !reflect.DeepEqual(a.Primitives(), b.Primitives()) {
		t.Error("the same seed scattered different objects")
	}
	if c := build(t, s, 6, keep...); reflect.DeepEqual(a.Primitives(), c.Primitives()) {
		t.Error("another seed scattered the same objects")
	}
}

// Every ladder builds at every level, gets harder as the level rises,
// and Scale shrinks it.
func TestLadders(t *testing.T) {
	for _, k := range LadderNames() {
		var prev *Terrain
		for _, l := range []float64{0, 0.5, 1} {
			tr := build(t, Spec{Ladder: k, Level: l}, 3, [2]float64{5, 0})
			if tr.Spec.Goal == [2]float64{} {
				t.Errorf("%s: no goal", k)
			}
			if prev != nil && reflect.DeepEqual(prev.Spec, tr.Spec) {
				t.Errorf("%s: level %.1f is the same world as the level below", k, l)
			}
			prev = tr
		}
	}
	full := build(t, Spec{Ladder: "stairs-up", Level: 1}, 1)
	half := build(t, Spec{Ladder: "stairs-up", Level: 1, Scale: 0.5}, 1)
	if !near(half.Height(0.5+0.13*2.5, 0), full.Height(1+0.26*2.5, 0)/2, 1e-9) || half.Spec.Goal[0] != full.Spec.Goal[0]/2 {
		t.Errorf("scale 0.5: tread %.3f vs %.3f, goal %v vs %v", half.Height(0.5+0.13*2.5, 0), full.Height(1+0.26*2.5, 0), half.Spec.Goal, full.Spec.Goal)
	}
	if _, err := Build(Spec{Ladder: "lava"}, 1, nil); err == nil {
		t.Error("unknown ladder accepted")
	}
}

func BenchmarkDistanceCourse(b *testing.B) {
	tr, _ := Build(Spec{NoisePreset: "gentle", Stairs: []Stairs{{At: [2]float64{2, 0}, Rise: 0.15, Run: 0.3, Steps: 4, Width: 1.5, Landing: 1, Over: true}},
		Scatter: &Scatter{Count: 20, Min: [2]float64{-5, -5}, Max: [2]float64{5, 5}, Size: [2]float64{0.2, 0.5}, Height: [2]float64{0.1, 0.5}, Clear: 0.5}}, 1, nil)
	for i := 0; i < b.N; i++ {
		tr.Distance(geom.V3{float64(i%100) * 0.05, 0.01, 0.2}, 0.11, 0, 0, 0)
	}
}
