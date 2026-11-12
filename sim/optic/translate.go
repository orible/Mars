package optic

import (
	"math"
	"math/rand"

	"mars/sim/geom"
)

// Realism is the img2img translator's scene stage: it turns a clean
// render into plausible scene radiance by re-lighting and re-texturing it
// from the frame's G-buffer (albedo, object id, surface point in the
// object's frame, normal), the way G-buffer-conditioned photorealism
// enhancers do, but procedurally, so it runs in a few milliseconds on a
// CPU and needs no training data.
//
// Every stochastic detail it adds (albedo blotches, grime, tint drift,
// gloss variation) is a function of the surface point in its object's
// frame, and every lighting term (soft sun shadows from fixed sample
// directions, contact occlusion, sky and floor bounce) is a function of
// world geometry. So the same surface point gets the same value in the
// next frame and in the other eye: temporal and stereo consistency hold
// by construction instead of being enforced afterwards, and nothing it
// adds can move an edge, because it never resamples the image. Only the
// specular term depends on the viewpoint, as it does for a real camera.
type Realism struct {
	Scene *Scene
	// Grime scales all material variation (0 = clean render's textures).
	Grime float32
	// Shadows, Occlusion and Specular switch the lighting terms.
	Shadows, Occlusion, Specular bool
	// Fast trades softness for speed (for vision inside the RL loop):
	// one hard sun shadow ray instead of a soft cone, one noise octave,
	// one tint sample.
	Fast bool

	boxes []obb
	feet  []int // boxes standing on the floor
	sun1  [1]geom.V3
	grid  shadowGrid
}

// shadowGrid bins shadow casters by where they fall on a plane facing
// the sun, so a pixel tests only the casters whose shadow can reach it
// (Fast mode: one hard shadow ray, no cone).
type shadowGrid struct {
	u, v       geom.V3 // plane axes, perpendicular to the sun
	u0, v0, cs float64 // origin and cell size
	n          int
	cells      [][]int16
	ext        [][4]float64
}

const gridN = 64

func (g *shadowGrid) build(sun geom.V3, boxes []obb) {
	g.u = geom.V3{0, 1, 0}.Cross(sun).Unit()
	g.v = sun.Cross(g.u)
	lo := [2]float64{math.Inf(1), math.Inf(1)}
	hi := [2]float64{math.Inf(-1), math.Inf(-1)}
	g.ext = g.ext[:0]
	for _, b := range boxes {
		// The box's outline on the plane: the extent of its 8 corners.
		e := [4]float64{math.Inf(1), math.Inf(-1), math.Inf(1), math.Inf(-1)}
		if !b.static {
			for c := 0; c < 8; c++ {
				q := geom.V3{b.lo[0], b.lo[1], b.lo[2]}
				for a := 0; a < 3; a++ {
					if c>>a&1 == 1 {
						q[a] = b.hi[a]
					}
				}
				w := b.r.MulV(q).Add(b.p)
				pu, pv := w.Dot(g.u), w.Dot(g.v)
				e = [4]float64{math.Min(e[0], pu), math.Max(e[1], pu), math.Min(e[2], pv), math.Max(e[3], pv)}
			}
			lo = [2]float64{math.Min(lo[0], e[0]), math.Min(lo[1], e[2])}
			hi = [2]float64{math.Max(hi[0], e[1]), math.Max(hi[1], e[3])}
		}
		g.ext = append(g.ext, e)
	}
	g.n = gridN
	g.u0, g.v0 = lo[0], lo[1]
	g.cs = math.Max(hi[0]-lo[0], hi[1]-lo[1])/gridN + 1e-9
	if len(g.cells) != gridN*gridN {
		g.cells = make([][]int16, gridN*gridN)
	}
	for i := range g.cells {
		g.cells[i] = g.cells[i][:0]
	}
	for j, b := range boxes {
		if b.static {
			continue
		}
		e := g.ext[j]
		i0, i1 := g.cell(e[0], g.u0), g.cell(e[1], g.u0)
		j0, j1 := g.cell(e[2], g.v0), g.cell(e[3], g.v0)
		for a := i0; a <= i1; a++ {
			for c := j0; c <= j1; c++ {
				g.cells[c*gridN+a] = append(g.cells[c*gridN+a], int16(j))
			}
		}
	}
}

func (g *shadowGrid) cell(x, x0 float64) int { return clampi(int((x-x0)/g.cs), 0, g.n-1) }

// casters lists the casters whose shadow may cover point p (nil outside
// the grid: nothing there can be shadowed).
func (g *shadowGrid) casters(p geom.V3) []int16 {
	pu, pv := (p.Dot(g.u)-g.u0)/g.cs, (p.Dot(g.v)-g.v0)/g.cs
	if pu < 0 || pv < 0 || pu >= float64(g.n) || pv >= float64(g.n) {
		return nil
	}
	return g.cells[int(pv)*g.n+int(pu)]
}

// sunlitFast is sunlit for one hard ray along the sun, using the grid.
func (t *Realism) sunlitFast(origin geom.V3, skip int) float64 {
	for _, j := range t.grid.casters(origin) {
		b := &t.boxes[j]
		if int(j) == skip {
			continue
		}
		if slab(b.rt.MulV(origin.Sub(b.p)), b.sun, b.lo, b.hi) {
			return 0
		}
	}
	return 1
}

type obb struct {
	r, rt  geom.M3
	p      geom.V3
	lo, hi geom.V3
	floor  bool // stands on the floor (gets a contact shadow)
	static bool
	// Bounding sphere (world) and the sun sample directions in the box
	// frame, for the shadow test.
	c    geom.V3
	rad  float64
	sun  geom.V3
	dirs []geom.V3
}

// NewRealism is the translator with every term on.
func NewRealism(s *Scene) *Realism {
	return &Realism{Scene: s, Grime: 1, Shadows: true, Occlusion: true, Specular: true}
}

// Translate writes the realistic radiance of frame f into out.
func (t *Realism) Translate(f *Frame, out []RGB) {
	s := t.Scene
	t.sun1[0] = s.Sun
	t.boxes = t.boxes[:0]
	for i, o := range s.Objects {
		p := f.Poses[i]
		var dirs []geom.V3
		if i < cap(t.boxes) {
			dirs = t.boxes[:i+1][i].dirs[:0]
		}
		b := obb{r: p.R, rt: p.R.T(), p: p.P, lo: o.Mesh.Lo, hi: o.Mesh.Hi, dirs: dirs,
			floor: !o.Self && math.Abs(p.P[1]+o.Mesh.Lo[1]) < 0.02, static: o.Static}
		b.c = p.R.MulV(o.Mesh.Lo.Add(o.Mesh.Hi).Scale(0.5)).Add(p.P)
		b.rad = o.Mesh.Hi.Sub(o.Mesh.Lo).Norm() / 2
		b.sun = b.rt.MulV(s.Sun)
		for _, d := range t.sunDirs() {
			b.dirs = append(b.dirs, b.rt.MulV(d))
		}
		t.boxes = append(t.boxes, b)
	}
	t.feet = t.feet[:0]
	for j, b := range t.boxes {
		if b.floor && !b.static {
			t.feet = append(t.feet, j)
		}
	}
	cone := 0.0 // widest angle of the sun samples from the sun direction
	for _, d := range t.sunDirs() {
		cone = math.Max(cone, math.Acos(math.Min(1, d.Dot(s.Sun))))
	}
	cone = math.Tan(cone)*1.01 + 1e-9
	oct := 2
	if t.Fast {
		oct = 1
		if t.Shadows {
			t.grid.build(s.Sun, t.boxes)
		}
	}
	sun := s.Sun
	bounce := RGB{0.22, 0.2, 0.17}
	for k := range out {
		id := f.ID[k]
		if id == 0 {
			out[k] = f.RGB[k]
			continue
		}
		oi := int(id - 1)
		o := s.Objects[oi]
		l := f.Local[k]
		nn := f.Normal[k]
		n := geom.V3{float64(nn[0]), float64(nn[1]), float64(nn[2])}
		wp, _ := f.World(k)

		// Material: detail anchored to the surface point.
		alb := f.Albedo[k]
		g := o.Grime * t.Grime
		var fine, blot float32
		if g > 0 {
			nz := noise3{o.Seed}
			fine = nz.fbm(l[0]*6, l[1]*6, l[2]*6, oct)
			blot = noise3{o.Seed ^ 0x9e3779b9}.fbm(l[0]*0.9+3.1, l[1]*0.9, l[2]*0.9, oct)
			tint := noise3{o.Seed ^ 0x85ebca6b}
			alb = alb.scale(1 + 0.45*g*fine)
			if t.Fast { // one sample shifts warm against cool
				d := 0.12 * g * (tint.at(l[0]*1.7, l[1]*1.7, l[2]*1.7) - 0.5)
				alb[0] *= 1 + d
				alb[2] *= 1 - d
			} else {
				for c := 0; c < 3; c++ {
					alb[c] *= 1 + 0.12*g*(tint.at(l[0]*1.7+float32(c)*7.3, l[1]*1.7, l[2]*1.7)-0.5)
				}
			}
			dirt := clamp01(2.2 * g * blot)
			alb = alb.scale(1 - 0.45*dirt).add(RGB{0.3, 0.25, 0.18}.scale(0.45 * dirt * 0.5))
		}

		// Occlusion: darker near the floor on objects, and around every
		// object's footprint on the floor.
		ao := float32(1)
		if t.Occlusion {
			if !o.Static && t.boxes[oi].floor {
				ao *= 0.6 + 0.4*smoothstep(0, 0.3, l[1]-float32(o.Mesh.Lo[1]))
			}
			if o.Static {
				for _, j := range t.feet {
					b := &t.boxes[j]
					if dx, dz := wp[0]-b.c[0], wp[2]-b.c[2]; dx*dx+dz*dz > (b.rad+0.6)*(b.rad+0.6) {
						continue
					}
					if d := footprintDist(*b, wp); d < 0.6 {
						ao *= 1 - 0.55*expNeg(float32(d/0.12))
					}
				}
			}
		}

		// Sun with soft shadows from fixed directions.
		ndl := math.Max(0, n.Dot(sun))
		sh := 1.0
		if t.Shadows && ndl > 0 {
			if t.Fast {
				sh = t.sunlitFast(wp.Add(n.Scale(0.01)), oi)
			} else {
				sh = t.sunlit(wp.Add(n.Scale(0.01)), oi, cone)
			}
		}
		up := float32(n[1])
		light := s.SkyCol.scale((0.5 + 0.5*up) * ao).
			add(bounce.scale((0.5 - 0.5*up) * 0.6 * ao)).
			add(s.SunCol.scale(float32(ndl * sh)))
		c := alb.mul(light)
		if t.Specular && o.Spec > 0 && ndl > 0 && sh > 0 {
			v := f.Cam.P.Sub(wp).Unit()
			h := sun.Add(v).Unit()
			shine := float64(o.Shine) * (1 + 0.6*float64(fine))
			gloss := float64(o.Spec) * (1 - float64(clamp01(2*g*blot)))
			nh := math.Max(0, n.Dot(h))
			var lobe float64
			if t.Fast { // x^s ≈ e^(−s(1−x)) near the peak, where it matters
				lobe = float64(expNeg(float32(shine * (1 - nh))))
			} else {
				lobe = math.Pow(nh, shine)
			}
			sp := gloss * lobe * (shine + 8) / (8 * math.Pi) * ndl * sh
			c = c.add(s.SunCol.scale(float32(sp)))
		}
		out[k] = c
	}
}

// sunlit is the fraction of the sun sample directions along which the
// ray from origin escapes every shadow caster but object skip (a convex
// object cannot shadow itself beyond its own Lambert term). Casters whose
// bounding sphere is outside the cone of sample directions are skipped
// whole.
func (t *Realism) sunlit(origin geom.V3, skip int, cone float64) float64 {
	sun := t.Scene.Sun
	nd := len(t.sunDirs())
	var hit uint64 // bit j: direction j blocked
	for j := range t.boxes {
		b := &t.boxes[j]
		if j == skip || b.static {
			continue
		}
		oc := b.c.Sub(origin)
		along := oc.Dot(sun)
		if along < -b.rad {
			continue
		}
		perp2 := oc.Dot(oc) - along*along
		reach := b.rad + cone*math.Max(0, along)
		if perp2 > reach*reach {
			continue
		}
		o := b.rt.MulV(origin.Sub(b.p))
		// One test against the box grown by the cone's spread rejects most
		// casters (walls, whose bounding spheres are huge) before the
		// per-direction tests.
		m := cone * (math.Sqrt(oc.Dot(oc)) + b.rad)
		grow := geom.V3{m, m, m}
		if !slab(o, b.sun, b.lo.Sub(grow), b.hi.Add(grow)) {
			continue
		}
		for k, dd := range b.dirs {
			if hit&(1<<k) == 0 && slab(o, dd, b.lo, b.hi) {
				hit |= 1 << k
			}
		}
	}
	blocked := 0
	for k := 0; k < nd; k++ {
		if hit&(1<<k) != 0 {
			blocked++
		}
	}
	return 1 - float64(blocked)/float64(nd)
}

// slab reports whether the ray o + t·d (t > 0) meets the box [lo, hi].
func slab(o, d, lo, hi geom.V3) bool {
	t0, t1 := 0.0, math.Inf(1)
	for a := 0; a < 3; a++ {
		if math.Abs(d[a]) < 1e-12 {
			if o[a] < lo[a] || o[a] > hi[a] {
				return false
			}
			continue
		}
		inv := 1 / d[a]
		ta, tb := (lo[a]-o[a])*inv, (hi[a]-o[a])*inv
		if ta > tb {
			ta, tb = tb, ta
		}
		t0, t1 = math.Max(t0, ta), math.Min(t1, tb)
		if t0 > t1 {
			return false
		}
	}
	return true
}

// sunDirs are the shadow ray directions: the soft-shadow samples, or the
// sun alone when Fast.
func (t *Realism) sunDirs() []geom.V3 {
	if t.Fast || len(t.Scene.SunJitter) == 0 {
		return t.sun1[:]
	}
	return t.Scene.SunJitter
}

// footprintDist is the horizontal distance from world point p to the
// footprint rectangle of box b.
func footprintDist(b obb, p geom.V3) float64 {
	q := b.rt.MulV(p.Sub(b.p))
	dx := math.Max(0, math.Max(b.lo[0]-q[0], q[0]-b.hi[0]))
	dz := math.Max(0, math.Max(b.lo[2]-q[2], q[2]-b.hi[2]))
	return math.Hypot(dx, dz)
}

// expNeg is a fast e^−x for x ≥ 0 (rational approximation, error < 0.3 %).
func expNeg(x float32) float32 {
	x = x / 8
	y := 1 / (1 + x + x*x*(0.5+x*(1.0/6+x/24)))
	y *= y
	y *= y
	return y * y
}

func clamp01(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

func smoothstep(a, b, x float32) float32 {
	t := clamp01((x - a) / (b - a))
	return t * t * (3 - 2*t)
}

// Stabiliser conditions a translator on its own previous output: each
// pixel blends towards the previous output warped along the exact
// backward flow (frame t → t−1), where the surface was visible. It is
// what keeps a translator whose output has frame-to-frame randomness (a
// learned net, a screen-space detail generator) from flickering; the
// Realism stage does not need it but tolerates it.
type Stabiliser struct {
	Alpha float32 // weight of the warped history, 0..1
	prev  []RGB
}

// Apply stabilises img in place given the backward flow of its frame.
func (s *Stabiliser) Apply(img []RGB, back []Flow, w, h int) {
	if s.prev != nil && s.Alpha > 0 {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				k := y*w + x
				fl := back[k]
				if !fl.Valid {
					continue
				}
				p := bilinearRGB(s.prev, w, h, float32(x)+0.5+fl.DX, float32(y)+0.5+fl.DY)
				img[k] = img[k].scale(1 - s.Alpha).add(p.scale(s.Alpha))
			}
		}
	}
	s.prev = append(s.prev[:0], img...)
}

// Reset forgets the history (a cut).
func (s *Stabiliser) Reset() { s.prev = nil }

// ScreenNoise is the naive baseline the consistency metrics are checked
// against: a translator that hallucinates screen-space detail afresh for
// every frame and every eye (smoothed multiplicative noise of amplitude
// amp), the failure mode of an unconditioned per-frame img2img net.
func ScreenNoise(img []RGB, w, h int, amp float32, rng *rand.Rand) {
	n := make([]float32, w*h)
	for i := range n {
		n[i] = float32(rng.NormFloat64())
	}
	m := make([]float32, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var s, c float32
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					xx, yy := x+dx, y+dy
					if xx >= 0 && yy >= 0 && xx < w && yy < h {
						s += n[yy*w+xx]
						c++
					}
				}
			}
			m[y*w+x] = s / c * 3 // back to unit variance
		}
	}
	for i := range img {
		img[i] = img[i].scale(1 + amp*m[i])
	}
}

// bilinearRGB samples img at continuous position (x, y) (pixel centres at
// +0.5), clamping at the border.
func bilinearRGB(img []RGB, w, h int, x, y float32) RGB {
	x -= 0.5
	y -= 0.5
	fx, fy := floorf(x), floorf(y)
	ax, ay := x-fx, y-fy
	x0, y0 := clampi(int(fx), 0, w-1), clampi(int(fy), 0, h-1)
	x1, y1 := clampi(int(fx)+1, 0, w-1), clampi(int(fy)+1, 0, h-1)
	a, b := img[y0*w+x0], img[y0*w+x1]
	c, d := img[y1*w+x0], img[y1*w+x1]
	return a.scale(1 - ax).add(b.scale(ax)).scale(1 - ay).add(c.scale(1 - ax).add(d.scale(ax)).scale(ay))
}

func clampi(x, lo, hi int) int {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}
