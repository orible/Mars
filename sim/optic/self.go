package optic

import (
	"math"
	"math/rand"

	"mars/sim/geom"
	"mars/sim/model"
)

// Body is the robot's own geometry in a scene: one low-poly capsule per
// collision capsule of the model, riding on its link, so the eyes see
// their own feet, knees, arms and hands. Pixels showing it carry the link
// index in Frame.Part, which the encoder can match to the policy's part
// tokens (obs.Parts lists links heavier than obs.PartMinMass, in link
// order).
type Body struct {
	Objects []int // scene object index of each body capsule
}

// AddBody appends the body of m to scene s. Capsules on the head link
// that the eyes sit inside are left out (the eyes would see the inside of
// the head).
func AddBody(s *Scene, m *model.Model, hm HeadMount, seed int64) *Body {
	rng := rand.New(rand.NewSource(seed))
	tex := Bake(PatMarble, 32, RGB{0.32, 0.33, 0.35}, RGB{0.2, 0.2, 0.22}, seed)
	b := &Body{}
	for _, c := range m.Capsules {
		if c.Link == hm.Link && segDist(hm.Eye, c.A, c.B) < c.R+0.03 {
			continue
		}
		o := &Object{Mesh: CapsuleMesh(c.A, c.B, c.R, 6, 4), Tex: tex, Tint: RGB{1, 1, 1}, R: geom.Identity(),
			Kind: 1000 + c.Link, Grime: 0.4, Spec: 0.3, Shine: 25, Seed: rng.Uint32(), Self: true, Link: c.Link}
		b.Objects = append(b.Objects, len(s.Objects))
		s.Objects = append(s.Objects, o)
	}
	// Appearance vectors for the new capsules: one prototype per link.
	s.AppearanceVectors(rand.New(rand.NewSource(seed+1)), 0.3)
	return b
}

// Update moves the body capsules to the links' poses in w.
func (b *Body) Update(s *Scene, w *model.World) {
	for _, i := range b.Objects {
		o := s.Objects[i]
		o.R, o.P = w.Pose.R[o.Link], w.Pose.P[o.Link]
	}
}

func segDist(p, a, b geom.V3) float64 {
	ab := b.Sub(a)
	t := 0.0
	if l2 := ab.Dot(ab); l2 > 0 {
		t = math.Max(0, math.Min(1, p.Sub(a).Dot(ab)/l2))
	}
	return p.Sub(a.Add(ab.Scale(t))).Norm()
}

// CapsuleMesh is a low-poly capsule around segment a–b of radius r: an
// n-sided prism with a cone over each end.
func CapsuleMesh(a, b geom.V3, r float64, n int, rep float64) *Mesh {
	d := b.Sub(a)
	l := d.Norm()
	if l < 1e-9 {
		d = geom.V3{0, 1, 0}
	} else {
		d = d.Scale(1 / l)
	}
	u := geom.V3{1, 0, 0}
	if math.Abs(d[0]) > 0.9 {
		u = geom.V3{0, 0, 1}
	}
	u = u.Sub(d.Scale(u.Dot(d))).Unit()
	v := d.Cross(u)
	m := &Mesh{}
	circ := 2 * math.Pi * r
	for i := 0; i <= n; i++ {
		ang := 2 * math.Pi * float64(i) / float64(n)
		nv := u.Scale(math.Cos(ang)).Add(v.Scale(math.Sin(ang)))
		uu := float64(i) / float64(n) * circ * rep
		m.add(Vertex{a.Add(nv.Scale(r)), nv, [2]float64{uu, 0}}, Vertex{b.Add(nv.Scale(r)), nv, [2]float64{uu, l * rep}})
	}
	for i := 0; i < n; i++ {
		p, q := int32(2*i), int32(2*i+2)
		m.Tri = append(m.Tri, [3]int32{p, q, q + 1}, [3]int32{p, q + 1, p + 1})
	}
	ta := m.add(Vertex{a.Sub(d.Scale(r)), d.Scale(-1), [2]float64{0, -r * rep}})
	tb := m.add(Vertex{b.Add(d.Scale(r)), d, [2]float64{0, (l + r) * rep}})
	for i := 0; i < n; i++ {
		p, q := int32(2*i), int32(2*i+2)
		m.Tri = append(m.Tri, [3]int32{ta, q, p}, [3]int32{tb, p + 1, q + 1})
	}
	m.bounds()
	return m
}

// EyeToken is the camera's sense of its own geometry, for the encoder to
// read beside the frames the way the policy reads part and sensor
// tokens. Lengths are divided by the body's standing pelvis height (as
// in obs), so a scaled-down robot with a shorter baseline reads the
// same; MaxDepth shows how the stereo range shrinks with the baseline.
type EyeToken struct {
	Baseline   float64    // B / L
	HFOV, VFOV float64    // radians
	Focal      float64    // focal length / image width (= ½·cot(HFOV/2))
	W, H       float64    // resolution / 100
	Mount      [3]float64 // eye centre in the head link frame / L
	Pitch      float64    // nose up +, against gravity
	Yaw        float64    // left +, against the pelvis heading
	Roll       float64    // right ear down +
	// Up is world up in the eye frame (forward, up, right): the same
	// information as pitch and roll, without their singularity when
	// looking straight down, where yaw and roll are ill-defined.
	Up       [3]float64
	Height   float64 // eye height above the floor below it / L
	MaxDepth float64 // depth at 1 px disparity (F·B) / L
}

// Vector is the token as a flat feature vector.
func (t EyeToken) Vector() []float64 {
	return []float64{t.Baseline, t.HFOV, t.VFOV, t.Focal, t.W, t.H, t.Mount[0], t.Mount[1], t.Mount[2],
		t.Pitch, t.Yaw, t.Roll, t.Up[0], t.Up[1], t.Up[2], t.Height, t.MaxDepth}
}

// Token builds the eye token for rig r on world w.
func (hm HeadMount) Token(r Rig, w *model.World) EyeToken {
	L := w.StandH()
	if L <= 0 {
		L = 1
	}
	p, y, ro := hm.HeadAngles(w)
	hr, eye := hm.HeadPose(w)
	up := hr.T().MulV(geom.V3{0, 1, 0})
	return EyeToken{
		Baseline: r.Baseline / L, HFOV: r.HFOV, VFOV: 2 * math.Atan(float64(r.H)/2/r.F()),
		Focal: r.F() / float64(r.W), W: float64(r.W) / 100, H: float64(r.H) / 100,
		Mount: [3]float64{hm.Eye[0] / L, hm.Eye[1] / L, hm.Eye[2] / L},
		Pitch: p, Yaw: y, Roll: ro, Up: up,
		Height:   (eye[1] - w.Ground.Height(eye[0], eye[2])) / L,
		MaxDepth: r.F() * r.Baseline / L,
	}
}
