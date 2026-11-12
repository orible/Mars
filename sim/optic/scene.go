package optic

import (
	"math"
	"math/rand"

	"mars/sim/geom"
)

// Vertex is a mesh corner in the object frame.
type Vertex struct {
	P  geom.V3    // position
	N  geom.V3    // normal
	UV [2]float64 // texture coordinates in repeats
}

// Mesh is an indexed triangle list. Box and cylinder meshes have their
// origin at the centre of their base, so an object stands on the floor at
// height 0.
type Mesh struct {
	V   []Vertex
	Tri [][3]int32
	Lo  geom.V3 // object-frame bounding box
	Hi  geom.V3
}

func (m *Mesh) add(vs ...Vertex) int32 {
	i := int32(len(m.V))
	m.V = append(m.V, vs...)
	return i
}

func (m *Mesh) bounds() {
	m.Lo = geom.V3{math.Inf(1), math.Inf(1), math.Inf(1)}
	m.Hi = geom.V3{math.Inf(-1), math.Inf(-1), math.Inf(-1)}
	for _, v := range m.V {
		for k := 0; k < 3; k++ {
			m.Lo[k], m.Hi[k] = math.Min(m.Lo[k], v.P[k]), math.Max(m.Hi[k], v.P[k])
		}
	}
}

// quad adds a planar quad a,b,c,d (counter-clockwise seen from outside)
// with normal n, textured at rep repeats per metre.
func (m *Mesh) quad(a, b, c, d, n geom.V3, rep float64) {
	ab, ad := b.Sub(a), d.Sub(a)
	la, lb := ab.Norm(), ad.Norm()
	i := m.add(
		Vertex{a, n, [2]float64{0, 0}},
		Vertex{b, n, [2]float64{la * rep, 0}},
		Vertex{c, n, [2]float64{la * rep, lb * rep}},
		Vertex{d, n, [2]float64{0, lb * rep}})
	m.Tri = append(m.Tri, [3]int32{i, i + 1, i + 2}, [3]int32{i, i + 2, i + 3})
}

// BoxMesh is an sx×sy×sz box (X depth, Y height, Z width) standing on
// its base.
func BoxMesh(sx, sy, sz, rep float64) *Mesh {
	m := &Mesh{}
	x, z := sx/2, sz/2
	p := func(a, b, c float64) geom.V3 { return geom.V3{a, b, c} }
	m.quad(p(x, 0, -z), p(x, 0, z), p(x, sy, z), p(x, sy, -z), geom.V3{1, 0, 0}, rep)      // +X
	m.quad(p(-x, 0, z), p(-x, 0, -z), p(-x, sy, -z), p(-x, sy, z), geom.V3{-1, 0, 0}, rep) // −X
	m.quad(p(-x, 0, z), p(x, 0, z), p(x, sy, z), p(-x, sy, z), geom.V3{0, 0, 1}, rep)      // +Z
	m.quad(p(x, 0, -z), p(-x, 0, -z), p(-x, sy, -z), p(x, sy, -z), geom.V3{0, 0, -1}, rep) // −Z
	m.quad(p(-x, sy, -z), p(-x, sy, z), p(x, sy, z), p(x, sy, -z), geom.V3{0, 1, 0}, rep)  // top
	m.quad(p(-x, 0, -z), p(x, 0, -z), p(x, 0, z), p(-x, 0, z), geom.V3{0, -1, 0}, rep)     // bottom
	m.bounds()
	return m
}

// CylinderMesh is an upright n-sided cylinder of radius r and height h
// with smooth side normals.
func CylinderMesh(r, h float64, n int, rep float64) *Mesh {
	m := &Mesh{}
	circ := 2 * math.Pi * r
	for i := 0; i <= n; i++ { // seam vertices duplicated for the wrap
		a := 2 * math.Pi * float64(i) / float64(n)
		c, s := math.Cos(a), math.Sin(a)
		nv := geom.V3{c, 0, s}
		u := float64(i) / float64(n) * circ * rep
		m.add(Vertex{geom.V3{r * c, 0, r * s}, nv, [2]float64{u, 0}}, Vertex{geom.V3{r * c, h, r * s}, nv, [2]float64{u, h * rep}})
	}
	for i := 0; i < n; i++ {
		a, b := int32(2*i), int32(2*i+2)
		m.Tri = append(m.Tri, [3]int32{a, b, b + 1}, [3]int32{a, b + 1, a + 1})
	}
	for _, y := range []float64{0, h} {
		ny := 1.0
		if y == 0 {
			ny = -1
		}
		c := m.add(Vertex{geom.V3{0, y, 0}, geom.V3{0, ny, 0}, [2]float64{0, 0}})
		first := int32(len(m.V))
		for i := 0; i < n; i++ {
			a := 2 * math.Pi * float64(i) / float64(n)
			m.add(Vertex{geom.V3{r * math.Cos(a), y, r * math.Sin(a)}, geom.V3{0, ny, 0}, [2]float64{r * math.Cos(a) * rep, r * math.Sin(a) * rep}})
		}
		for i := 0; i < n; i++ {
			m.Tri = append(m.Tri, [3]int32{c, first + int32(i), first + int32((i+1)%n)})
		}
	}
	m.bounds()
	return m
}

// FloorMesh is a size×size horizontal square at y = 0 centred on the
// origin, cut into cells so near-plane clipping stays well conditioned.
func FloorMesh(size float64, cells int, rep float64) *Mesh {
	m := &Mesh{}
	d := size / float64(cells)
	up := geom.V3{0, 1, 0}
	for i := 0; i < cells; i++ {
		for j := 0; j < cells; j++ {
			x0, z0 := -size/2+float64(i)*d, -size/2+float64(j)*d
			a, b, c, e := geom.V3{x0, 0, z0}, geom.V3{x0, 0, z0 + d}, geom.V3{x0 + d, 0, z0 + d}, geom.V3{x0 + d, 0, z0}
			k := m.add(
				Vertex{a, up, [2]float64{a[0] * rep, a[2] * rep}}, Vertex{b, up, [2]float64{b[0] * rep, b[2] * rep}},
				Vertex{c, up, [2]float64{c[0] * rep, c[2] * rep}}, Vertex{e, up, [2]float64{e[0] * rep, e[2] * rep}})
			m.Tri = append(m.Tri, [3]int32{k, k + 1, k + 2}, [3]int32{k, k + 2, k + 3})
		}
	}
	m.bounds()
	return m
}

// Object is one rigid thing in the scene. Its id in segmentation buffers
// is its index + 1 (0 is the sky).
type Object struct {
	Mesh *Mesh
	Tex  *Texture
	Tint RGB
	R    geom.M3 // object frame to world
	P    geom.V3
	// Kind groups objects that look alike; Appearance is the fixed
	// vector the optic bus would carry for this object (unit length,
	// close to the other objects of its kind).
	Kind       int
	Appearance []float32
	// Material realism for the translator: how strongly the surface
	// varies, how shiny it is, and the seed that anchors its detail.
	Grime, Spec, Shine float32
	Seed               uint32
	// Motion, if set, gives the pose at time t (s); Scene.At applies it.
	Motion func(t float64) (geom.M3, geom.V3)
	Static bool // never casts shadows or occludes the sun (the floor)
	// Self marks the robot's own body; Link is then the model link the
	// object rides on (its frame is the link frame).
	Self bool
	Link int
}

// Scene is the floor, walls and objects, plus lighting.
type Scene struct {
	Objects []*Object
	Sun     geom.V3 // unit, pointing towards the sun
	SunCol  RGB
	SkyCol  RGB // hemisphere light from above
	Horizon RGB // sky colour at the horizon (background)
	Zenith  RGB
	AppDim  int
	// SunJitter are the fixed directions the translator's soft shadows
	// sample (deterministic, so shadows never flicker).
	SunJitter []geom.V3
}

// Clone copies the scene so another environment can move its objects
// independently; meshes, textures and appearance vectors are shared.
func (s *Scene) Clone() *Scene {
	c := *s
	c.Objects = make([]*Object, len(s.Objects))
	for i, o := range s.Objects {
		oc := *o
		c.Objects[i] = &oc
	}
	return &c
}

// At moves every animated object to its pose at time t.
func (s *Scene) At(t float64) {
	for _, o := range s.Objects {
		if o.Motion != nil {
			o.R, o.P = o.Motion(t)
		}
	}
}

// AppearanceVectors draws a unit prototype per kind and gives each object
// that has no vector yet its prototype plus a small private perturbation (cosine ≈ 0.95 within a
// kind, ≈ 0 across kinds).
func (s *Scene) AppearanceVectors(rng *rand.Rand, spread float64) {
	protos := map[int][]float64{}
	unit := func(v []float64) []float64 {
		n := 0.0
		for _, x := range v {
			n += x * x
		}
		n = math.Sqrt(n)
		for i := range v {
			v[i] /= n
		}
		return v
	}
	gauss := func() []float64 {
		v := make([]float64, s.AppDim)
		for i := range v {
			v[i] = rng.NormFloat64()
		}
		return v
	}
	for _, o := range s.Objects {
		if o.Appearance != nil {
			continue
		}
		p, ok := protos[o.Kind]
		if !ok {
			p = unit(gauss())
			protos[o.Kind] = p
		}
		g := gauss()
		v := make([]float64, s.AppDim)
		for i := range v {
			v[i] = p[i] + spread*g[i]/math.Sqrt(float64(s.AppDim))
		}
		v = unit(v)
		o.Appearance = make([]float32, s.AppDim)
		for i := range v {
			o.Appearance[i] = float32(v[i])
		}
	}
}

// Cosine is the cosine similarity of two appearance vectors.
func Cosine(a, b []float32) float64 {
	var d, na, nb float64
	for i := range a {
		d += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return d / math.Sqrt(na*nb)
}

func yaw(a float64) geom.M3 { return geom.AxisAngle(geom.V3{0, 1, 0}, a) }

// DefaultScene is a room about 12 m × 10 m with a tiled floor, brick
// walls and a dozen low-poly objects in front of a robot standing at the
// origin facing +X: pairs of near-identical crates, barrels and pillars
// (identity is ambiguous within a pair), a door that swings and a box
// that slides, so frames carry object motion as well as ego-motion.
func DefaultScene(seed int64) *Scene {
	rng := rand.New(rand.NewSource(seed))
	s := &Scene{
		Sun:    geom.V3{-0.45, 0.8, 0.4}.Unit(),
		SunCol: RGB{2.6, 2.45, 2.2}, SkyCol: RGB{0.55, 0.62, 0.75},
		Horizon: RGB{0.75, 0.8, 0.85}, Zenith: RGB{0.35, 0.5, 0.8},
		AppDim: 32,
	}
	// Fixed soft-shadow directions: a small disc around the sun.
	for k := 0; k < 6; k++ {
		a := 2 * math.Pi * float64(k) / 6
		t := geom.V3{0, 1, 0}.Cross(s.Sun).Unit()
		b := s.Sun.Cross(t)
		r := 0.06
		s.SunJitter = append(s.SunJitter, s.Sun.Add(t.Scale(r*math.Cos(a))).Add(b.Scale(r*math.Sin(a))).Unit())
	}
	const ts = 64
	type kind struct {
		pat  Pattern
		a, b RGB
		tex  *Texture
	}
	kinds := []kind{
		{pat: PatTiles, a: RGB{0.55, 0.53, 0.5}, b: RGB{0.25, 0.24, 0.23}},     // 0 floor
		{pat: PatBrick, a: RGB{0.55, 0.28, 0.2}, b: RGB{0.7, 0.68, 0.62}},      // 1 walls
		{pat: PatCardboard, a: RGB{0.62, 0.45, 0.28}, b: RGB{0.45, 0.3, 0.18}}, // 2 crates
		{pat: PatStripes, a: RGB{0.15, 0.3, 0.55}, b: RGB{0.75, 0.75, 0.7}},    // 3 barrels
		{pat: PatWood, a: RGB{0.5, 0.33, 0.18}, b: RGB{0.32, 0.2, 0.1}},        // 4 door
		{pat: PatMarble, a: RGB{0.8, 0.8, 0.78}, b: RGB{0.45, 0.47, 0.5}},      // 5 pillars
		{pat: PatDots, a: RGB{0.2, 0.5, 0.25}, b: RGB{0.85, 0.8, 0.3}},         // 6 dotted drum
		{pat: PatStripes, a: RGB{0.6, 0.15, 0.12}, b: RGB{0.85, 0.82, 0.75}},   // 7 sliding box
	}
	for i := range kinds {
		size := ts
		if i == 0 {
			size = 128
		}
		kinds[i].tex = Bake(kinds[i].pat, size, kinds[i].a, kinds[i].b, seed*31+int64(i))
	}
	add := func(k int, m *Mesh, x, z, rot float64, grime, spec, shine float32) *Object {
		jit := func() float32 { return float32(1 + 0.04*(2*rng.Float64()-1)) }
		o := &Object{Mesh: m, Tex: kinds[k].tex, Tint: RGB{jit(), jit(), jit()}, R: yaw(rot), P: geom.V3{x, 0, z},
			Kind: k, Grime: grime, Spec: spec, Shine: shine, Seed: rng.Uint32()}
		s.Objects = append(s.Objects, o)
		return o
	}
	fl := add(0, FloorMesh(40, 10, 1.5), 0, 0, 0, 0.6, 0.15, 30)
	fl.Tint, fl.Static = RGB{1, 1, 1}, true
	// Walls: three sides of the room.
	add(1, BoxMesh(0.2, 3, 10, 1), 8, 0, 0, 0.5, 0.02, 8)
	add(1, BoxMesh(12, 3, 0.2, 1), 2, -5, 0, 0.5, 0.02, 8)
	add(1, BoxMesh(12, 3, 0.2, 1), 2, 5, 0, 0.5, 0.02, 8)
	// Near-identical pairs.
	add(2, BoxMesh(0.5, 0.45, 0.6, 2), 2.2, -0.9, 0.2, 0.7, 0.03, 6)
	add(2, BoxMesh(0.5, 0.45, 0.6, 2), 3.0, 1.1, -0.35, 0.7, 0.03, 6)
	add(3, CylinderMesh(0.28, 0.9, 14, 2), 3.6, -1.8, 0, 0.5, 0.35, 40)
	add(3, CylinderMesh(0.28, 0.9, 14, 2), 4.4, 0.4, 0, 0.5, 0.35, 40)
	add(5, CylinderMesh(0.18, 2.2, 10, 1.5), 5.5, -2.6, 0, 0.3, 0.25, 60)
	add(5, CylinderMesh(0.18, 2.2, 10, 1.5), 5.5, 2.6, 0, 0.3, 0.25, 60)
	add(6, CylinderMesh(0.22, 0.6, 12, 3), 1.6, 1.6, 0, 0.5, 0.2, 30)
	// A door hinged at its left edge, swinging open and shut.
	door := add(4, BoxMesh(0.05, 2.0, 0.9, 1.5), 6.5, -0.5, 0, 0.4, 0.1, 20)
	hinge := geom.V3{6.5, 0, -0.95}
	door.Motion = func(t float64) (geom.M3, geom.V3) {
		a := 0.6 * (1 - math.Cos(1.3*t))
		r := yaw(a)
		return r, hinge.Add(r.MulV(geom.V3{0, 0, 0.45}))
	}
	door.R, door.P = door.Motion(0)
	// A box sliding left and right across the robot's path.
	slider := add(7, BoxMesh(0.4, 0.4, 0.4, 2.5), 3.7, 0, 0, 0.6, 0.05, 10)
	slider.Motion = func(t float64) (geom.M3, geom.V3) {
		return yaw(0.4 * t), geom.V3{3.7, 0, 0.9 * math.Sin(0.9*t)}
	}
	slider.R, slider.P = slider.Motion(0)
	s.AppearanceVectors(rng, 0.3)
	return s
}
