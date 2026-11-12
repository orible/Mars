package optic

import (
	"math"
	"math/rand"
	"testing"

	"mars/sim/body"
	"mars/sim/geom"
	"mars/sim/model"
)

// lookPose is a head 1.6 m up at (x, z), turned yaw and pitched down.
func lookPose(x, z, yawA, pitch float64) (geom.M3, geom.V3) {
	r := geom.AxisAngle(geom.V3{0, 1, 0}, yawA).Mul(geom.AxisAngle(geom.V3{0, 0, 1}, -pitch))
	return r, geom.V3{x, 1.6, z}
}

// twoFrames runs the pipeline (no camera stage) for two steps with the
// head moving forward, sideways and turning, while the scene animates.
func twoFrames(t *testing.T) (*Pipeline, *StereoFrame, *StereoFrame) {
	t.Helper()
	s := DefaultScene(3)
	p := NewPipeline(s, DefaultRig(), 3)
	p.Camera = nil
	r, e := lookPose(0, 0, 0, 0.15)
	p.Step(r, e, 0)
	r, e = lookPose(0.04, 0.015, 0.02, 0.16)
	f := p.Step(r, e, 0.05)
	return p, p.Prev(), f
}

func TestRenderSeesSceneAndDepth(t *testing.T) {
	_, _, f := twoFrames(t)
	ids := map[uint16]int{}
	for _, id := range f.L.ID {
		ids[id]++
	}
	if ids[0] == 0 || ids[1] == 0 || len(ids) < 8 {
		t.Fatalf("expected sky, floor and several objects, got %v", ids)
	}
	// Depth agrees with the world position of every pixel.
	for k := range f.L.ID {
		wp, ok := f.L.World(k)
		if !ok {
			continue
		}
		u, v, d, ok := f.L.Cam.Project(wp)
		x, y := float64(k%f.L.W)+0.5, float64(k/f.L.W)+0.5
		if !ok || math.Abs(d-float64(f.L.Depth[k])) > 1e-3*d || math.Abs(u-x) > 1e-3 || math.Abs(v-y) > 1e-3 {
			t.Fatalf("pixel %d: reprojects to (%.4f,%.4f,%.4f), want (%.1f,%.1f,%.4f)", k, u, v, d, x, y, f.L.Depth[k])
		}
	}
}

func TestStereoCorrespondenceIsDisparity(t *testing.T) {
	p, _, f := twoFrames(t)
	n := 0
	for k, c := range f.LR {
		if !c.Valid || f.L.ID[k] == 0 {
			continue
		}
		want := -p.Rig.Disparity(float64(f.L.Depth[k]))
		if math.Abs(float64(c.DX)-want) > 1e-3 || math.Abs(float64(c.DY)) > 1e-3 {
			t.Fatalf("pixel %d: stereo flow (%.4f, %.4f), want (%.4f, 0)", k, c.DX, c.DY, want)
		}
		n++
	}
	if n < len(f.LR)/2 {
		t.Fatalf("only %d of %d pixels have a stereo match", n, len(f.LR))
	}
}

func TestFlowWarpsCleanRenderExactly(t *testing.T) {
	_, prev, f := twoFrames(t)
	cur, old := Display(f.L.RGB, 0.45), Display(prev.L.RGB, 0.45)
	exact := Flicker(cur, old, f.BackL, f.L)
	zero := make([]Flow, len(f.BackL))
	for k := range zero {
		zero[k] = Flow{Valid: f.BackL[k].Valid}
	}
	none := Flicker(cur, old, zero, f.L)
	t.Logf("warped difference: exact flow %.2f, no flow %.2f", exact, none)
	if exact > 2.5 || none < 2*exact {
		t.Fatalf("exact flow should explain the frame change: exact %.2f, none %.2f", exact, none)
	}
	// Moving objects: the sliding box must have flow that differs from the
	// ego-motion of the floor around it.
	valid := 0
	for _, fl := range f.BackL {
		if fl.Valid {
			valid++
		}
	}
	if valid < len(f.BackL)*3/4 {
		t.Fatalf("only %d of %d pixels tracked", valid, len(f.BackL))
	}
}

func TestRealismIsConsistent(t *testing.T) {
	p, prev, f := twoFrames(t)
	w, h := p.Rig.W, p.Rig.H
	clean := Flicker(Display(f.L.RGB, 0.45), Display(prev.L.RGB, 0.45), f.BackL, f.L)
	real := Flicker(Display(f.RadL, 0.45), Display(prev.RadL, 0.45), f.BackL, f.L)
	rng := rand.New(rand.NewSource(1))
	naiveA := append([]RGB(nil), prev.RadL...)
	naiveB := append([]RGB(nil), f.RadL...)
	ScreenNoise(naiveA, w, h, 0.12, rng)
	ScreenNoise(naiveB, w, h, 0.12, rng)
	naive := Flicker(Display(naiveB, 0.45), Display(naiveA, 0.45), f.BackL, f.L)
	st := &Stabiliser{Alpha: 0.8}
	st.Apply(naiveA, prev.BackL, w, h)
	st.Apply(naiveB, f.BackL, w, h)
	stab := Flicker(Display(naiveB, 0.45), Display(naiveA, 0.45), f.BackL, f.L)
	t.Logf("flicker: clean %.2f realism %.2f naive %.2f naive+stabiliser %.2f", clean, real, naive, stab)
	if real > 1.6*clean+0.5 {
		t.Errorf("realism flickers: %.2f vs clean %.2f", real, clean)
	}
	if naive < 3*real {
		t.Errorf("the metric should catch per-frame noise: naive %.2f vs realism %.2f", naive, real)
	}
	if stab > naive/2 {
		t.Errorf("stabiliser should halve flicker: %.2f vs %.2f", stab, naive)
	}

	// Stereo: without the view-dependent specular term, both eyes give a
	// surface point the same radiance.
	p.Realism.Specular = false
	l, r := make([]RGB, w*h), make([]RGB, w*h)
	p.Realism.Translate(f.L, l)
	p.Realism.Translate(f.R, r)
	var diff, n float64
	m := interior(f.L, f.LR)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			k := y*w + x
			if !m[k] || f.L.ID[k] <= 1 { // the floor's grime is fine-grained; objects carry the test
				continue
			}
			q := bilinearRGB(r, w, h, float32(x)+0.5+f.LR[k].DX, float32(y)+0.5+f.LR[k].DY)
			diff += math.Abs(float64(luma(l[k]) - luma(q)))
			n++
		}
	}
	if d := diff / n; d > 0.03 {
		t.Errorf("left and right disagree by %.3f (linear) at matching points", d)
	}
}

func TestEdgeDriftMeasuresShift(t *testing.T) {
	_, _, f := twoFrames(t)
	w, h := f.L.W, f.L.H
	l := Luma(Display(f.L.RGB, 0.45))
	shifted := make([]float32, len(l))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			shifted[y*w+x] = l[y*w+clampi(x-1, 0, w-1)]
		}
	}
	edges := GeometricEdges(f.L)
	same, n := EdgeDrift(l, l, edges, w, h, 0)
	moved, _ := EdgeDrift(l, shifted, edges, w, h, 0)
	t.Logf("edge drift: identical %.3f, 1 px shift %.3f over %d edge pixels", same, moved, n)
	if n < 100 || same > 0.01 || moved < 0.4 || moved > 1.1 {
		t.Fatalf("edge drift metric off: identical %.3f, shifted %.3f (n %d)", same, moved, n)
	}
}

func TestRealismKeepsEdges(t *testing.T) {
	p, _, f := twoFrames(t)
	w, h := f.L.W, f.L.H
	l := Luma(Display(f.L.RGB, 0.45))
	e := GeometricEdges(f.L)
	out := make([]RGB, w*h)
	drift := func() float64 {
		p.Realism.Translate(f.L, out)
		d, _ := EdgeDrift(l, Luma(Display(out, 0.45)), e, w, h, 0)
		return d
	}
	all := drift()
	p.Realism.Shadows = false
	noShadow := drift()
	t.Logf("edge drift: realism %.3f px, without cast shadows %.3f px", all, noShadow)
	// Material and shading terms never resample the image; cast shadows
	// add new edges beside contact edges, which the metric partly reads
	// as movement.
	if noShadow > 0.03 || all > 0.3 {
		t.Fatalf("realism moves edges: %.3f px (%.3f without shadows)", all, noShadow)
	}
}

func TestBlockMatchOnCleanPair(t *testing.T) {
	p, _, f := twoFrames(t)
	maxD := int(math.Ceil(p.Rig.Disparity(0.6)))
	lc := Luma(Display(f.L.RGB, 0.45))
	mask := StereoMask(f.L, f.LR, lc, maxD, 0.02)
	est := BlockMatch(lc, Luma(Display(f.R.RGB, 0.45)), f.L.W, f.L.H, maxD)
	mae, bad := DisparityError(est, p.Rig.DisparityMap(f.L), mask)
	t.Logf("clean pair: MAE %.2f px, %.0f%% off by >1 px", mae, 100*bad)
	if mae > 0.8 {
		t.Fatalf("block matching on the clean pair is too poor to judge translation: MAE %.2f", mae)
	}
}

func TestCameraOutputRange(t *testing.T) {
	s := DefaultScene(5)
	p := NewPipeline(s, DefaultRig(), 5)
	var f *StereoFrame
	for i := 0; i < 3; i++ {
		r, e := lookPose(0.03*float64(i), 0, 0, 0.15)
		f = p.Step(r, e, 0.05*float64(i))
	}
	var mean float64
	for _, c := range f.OutL {
		for _, v := range c {
			if v < 0 || v > 1 || math.IsNaN(float64(v)) {
				t.Fatalf("output %v out of range", c)
			}
		}
		mean += float64(luma(c))
	}
	mean /= float64(len(f.OutL))
	if mean < 0.2 || mean > 0.8 {
		t.Fatalf("auto-exposure left mean %.2f", mean)
	}
}

func TestJPEG(t *testing.T) {
	w, h := 32, 24
	img := make([]RGB, w*h)
	rng := rand.New(rand.NewSource(1))
	for k := range img {
		x := float32(k%w) / float32(w)
		img[k] = RGB{x, 0.5 + 0.1*float32(rng.Float64()), 1 - x}
	}
	errAt := func(q int) float64 {
		c := append([]RGB(nil), img...)
		JPEG(c, w, h, q)
		var e float64
		for k := range c {
			for ch := 0; ch < 3; ch++ {
				e += math.Abs(float64(c[k][ch] - img[k][ch]))
			}
		}
		return e / float64(3*len(c))
	}
	hi, lo := errAt(100), errAt(20)
	if hi > 0.02 || lo <= hi {
		t.Fatalf("JPEG error: quality 100 %.4f, quality 20 %.4f", hi, lo)
	}
}

func TestAppearanceVectors(t *testing.T) {
	s := DefaultScene(1)
	for i, a := range s.Objects {
		for j := i + 1; j < len(s.Objects); j++ {
			b := s.Objects[j]
			c := Cosine(a.Appearance, b.Appearance)
			if a.Kind == b.Kind && c < 0.8 {
				t.Errorf("objects %d and %d of one kind: cosine %.2f", i, j, c)
			}
			if a.Kind != b.Kind && math.Abs(c) > 0.6 {
				t.Errorf("objects %d and %d of different kinds: cosine %.2f", i, j, c)
			}
		}
	}
}

func TestHeadOnBody(t *testing.T) {
	m, err := model.New(body.Human(body.DefaultHuman()))
	if err != nil {
		t.Fatal(err)
	}
	w := model.NewWorld(m, true)
	hm := FindHead(m)
	_, eye := hm.HeadPose(w)
	if eye[1] < 1.55 || eye[1] > 1.75 {
		t.Fatalf("eye height %.3f m on a 1.78 m body", eye[1])
	}
	pitch, yaw, roll := hm.HeadAngles(w)
	if math.Abs(pitch)+math.Abs(yaw)+math.Abs(roll) > 1e-6 {
		t.Fatalf("standing head angles %.3f %.3f %.3f", pitch, yaw, roll)
	}
	// The angles rebuild the head orientation in the heading frame.
	rng := rand.New(rand.NewSource(2))
	for i := 0; i < 50; i++ {
		head := geom.AxisAngle(geom.V3{rng.NormFloat64(), rng.NormFloat64(), rng.NormFloat64()}, rng.Float64()*1.2)
		hy := 2 * math.Pi * rng.Float64()
		p, y, r := Angles(head, hy)
		got := HeadingRotation(p, y, r)
		want := geom.AxisAngle(geom.V3{0, 1, 0}, -hy).Mul(head)
		for a := 0; a < 3; a++ {
			for b := 0; b < 3; b++ {
				if math.Abs(got[a][b]-want[a][b]) > 1e-9 {
					t.Fatalf("angles (%.3f %.3f %.3f) rebuild %v, want %v", p, y, r, got, want)
				}
			}
		}
	}
}

func BenchmarkStereoFrame(b *testing.B) {
	s := DefaultScene(1)
	p := NewPipeline(s, DefaultRig(), 1)
	for i := 0; i < b.N; i++ {
		r, e := lookPose(0.02*float64(i%50), 0, 0.01*float64(i%50), 0.15)
		p.Step(r, e, 0.05*float64(i))
	}
	tm := p.Timing
	n := float64(tm.Frames)
	b.ReportMetric(tm.Render.Seconds()*1000/n, "render-ms")
	b.ReportMetric(tm.Truth.Seconds()*1000/n, "truth-ms")
	b.ReportMetric(tm.Realism.Seconds()*1000/n, "realism-ms")
	b.ReportMetric(tm.Camera.Seconds()*1000/n, "camera-ms")
}

func BenchmarkStereoFrameSmall(b *testing.B) {
	s := DefaultScene(1)
	rig := DefaultRig()
	rig.W, rig.H = 64, 48
	p := NewPipeline(s, rig, 1)
	for i := 0; i < b.N; i++ {
		r, e := lookPose(0.02*float64(i%50), 0, 0.01*float64(i%50), 0.15)
		p.Step(r, e, 0.05*float64(i))
	}
	tm := p.Timing
	n := float64(tm.Frames)
	b.ReportMetric(tm.Render.Seconds()*1000/n, "render-ms")
	b.ReportMetric(tm.Truth.Seconds()*1000/n, "truth-ms")
	b.ReportMetric(tm.Realism.Seconds()*1000/n, "realism-ms")
	b.ReportMetric(tm.Camera.Seconds()*1000/n, "camera-ms")
}

func BenchmarkEyesFast(b *testing.B) {
	m, _ := model.New(body.Human(body.DefaultHuman()))
	w := model.NewWorld(m, true)
	e := NewEyes(DefaultScene(1), m, FastProfile(), 1)
	for i := 0; i < b.N; i++ {
		t := 0.05 * float64(i)
		ScriptedWalk(w, t, 0.5+0.5*math.Sin(0.7*t))
		e.Step(w, t)
	}
	tm := e.Pipe.Timing
	n := float64(tm.Frames)
	b.ReportMetric(tm.Render.Seconds()*1000/n, "render-ms")
	b.ReportMetric(tm.Truth.Seconds()*1000/n, "truth-ms")
	b.ReportMetric(tm.Realism.Seconds()*1000/n, "realism-ms")
	b.ReportMetric(tm.Camera.Seconds()*1000/n, "camera-ms")
}

func humanWorld(t testing.TB, height float64) (*model.Model, *model.World) {
	spec := body.DefaultHuman()
	k := height / spec.Height
	spec.Height, spec.Mass = height, spec.Mass*k*k*k
	m, err := model.New(body.Human(spec))
	if err != nil {
		t.Fatal(err)
	}
	return m, model.NewWorld(m, true)
}

func TestSeesOwnBody(t *testing.T) {
	m, w := humanWorld(t, 1.778)
	e := NewEyes(DefaultScene(1), m, FastProfile(), 1)
	var f *StereoFrame
	for k := 0; k < 2; k++ {
		ScriptedWalk(w, 0.6+0.05*float64(k), 1)
		f = e.Step(w, 0.6+0.05*float64(k))
	}
	links := map[int16]int{}
	tracked := 0
	for k, p := range f.L.Part {
		id := f.L.ID[k]
		self := id > 0 && e.Scene.Objects[id-1].Self
		if (p >= 0) != self {
			t.Fatalf("pixel %d: part %d but object %d self=%v", k, p, id, self)
		}
		if p >= 0 {
			links[p]++
			if e.Scene.Objects[id-1].Link != int(p) {
				t.Fatalf("pixel %d: part %d on object riding link %d", k, p, e.Scene.Objects[id-1].Link)
			}
			if f.BackL[k].Valid {
				tracked++
			}
		}
	}
	t.Logf("own links in view (link: pixels): %v, %d tracked by flow", links, tracked)
	if len(links) < 2 {
		t.Fatalf("looking down should show several own links, saw %v", links)
	}
	if tracked == 0 {
		t.Fatal("no flow on the body")
	}
}

func TestEyeTokenScalesWithBody(t *testing.T) {
	tok := func(height float64) EyeToken {
		m, w := humanWorld(t, height)
		r := FastProfile().Rig
		r.Baseline *= height / 1.778 // a scaled-down robot gets scaled-down eyes
		return FindHead(m).Token(r, w)
	}
	a, b := tok(1.778), tok(0.9)
	va, vb := a.Vector(), b.Vector()
	for i := range va {
		if math.Abs(va[i]-vb[i]) > 0.03*math.Max(1, math.Abs(va[i])) {
			t.Fatalf("token feature %d differs with body size: %.4f vs %.4f\n%+v\n%+v", i, va[i], vb[i], a, b)
		}
	}
	if a.Height < 0.8 || a.MaxDepth <= 0 {
		t.Fatalf("token %+v", a)
	}
}

func TestBatchThreadsAndReset(t *testing.T) {
	m, _ := humanWorld(t, 1.778)
	ws := make([]*model.World, 4)
	ts := make([]float64, 4)
	for i := range ws {
		ws[i] = model.NewWorld(m, true)
	}
	run := func(threads int) [][]RGB {
		b := NewBatch(4, DefaultScene(1), m, FastProfile(), threads, 7)
		var out [][]RGB
		for k := 0; k < 3; k++ {
			for i := range ws {
				ts[i] = 0.05*float64(k) + 0.3*float64(i)
				ScriptedWalk(ws[i], ts[i], 0.5)
			}
			fs := b.Step(ws, ts)
			if k == 2 {
				for _, f := range fs {
					out = append(out, append([]RGB(nil), f.OutL...))
				}
				b.Envs[0].Reset()
				if f := b.Envs[0].Step(ws[0], ts[0]+0.05); !f.First {
					t.Fatal("frame after Reset is not marked first")
				}
			}
		}
		return out
	}
	a, c := run(1), run(3)
	for i := range a {
		for k := range a[i] {
			if a[i][k] != c[i][k] {
				t.Fatalf("env %d differs between 1 and 3 threads at pixel %d", i, k)
			}
		}
	}
}
