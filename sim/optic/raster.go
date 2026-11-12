package optic

import (
	"math"

	"mars/sim/geom"
)

// Pose is an object's placement when a frame was rendered.
type Pose struct {
	R geom.M3
	P geom.V3
}

// Frame is one eye's render with its ground truth. All buffers are W×H,
// row-major from the top-left.
type Frame struct {
	W, H int
	T    float64 // seconds
	Cam  Camera
	// RGB is the clean render (linear light): texture × (sky + sun
	// Lambert), no shadows. Albedo is the texture × tint alone.
	RGB, Albedo []RGB
	// Depth is distance along the viewing direction (+Inf for sky); ID is
	// 0 for sky, object index + 1 otherwise.
	Depth []float32
	ID    []uint16
	// Local is the visible surface point in its object's frame (for the
	// sky: the world ray direction). With Poses, it gives exact flow and
	// stereo correspondence for every pixel.
	Local  [][3]float32
	Normal [][3]float32 // world surface normal
	// Part is the robot's own link seen at each pixel, −1 for anything
	// that is not the robot.
	Part  []int16
	UV    [][2]float32
	LOD   []float32
	Poses []Pose // every object's pose at render time
}

// NewFrame allocates a w×h frame.
func NewFrame(w, h int) *Frame {
	n := w * h
	return &Frame{W: w, H: h, RGB: make([]RGB, n), Albedo: make([]RGB, n), Depth: make([]float32, n),
		ID: make([]uint16, n), Local: make([][3]float32, n), Normal: make([][3]float32, n),
		UV: make([][2]float32, n), LOD: make([]float32, n), Part: make([]int16, n)}
}

// cv is a clip-space vertex: camera coordinates plus attributes.
type cv struct {
	c, loc, n geom.V3
	uv        [2]float64
}

func lerpCV(a, b cv, t float64) cv {
	l := func(x, y geom.V3) geom.V3 { return x.Add(y.Sub(x).Scale(t)) }
	return cv{c: l(a.c, b.c), loc: l(a.loc, b.loc), n: l(a.n, b.n),
		uv: [2]float64{a.uv[0] + (b.uv[0]-a.uv[0])*t, a.uv[1] + (b.uv[1]-a.uv[1])*t}}
}

// Renderer holds scratch space; one per goroutine.
type Renderer struct {
	cvs  []cv
	poly []cv
	tmp  []cv
}

// Render draws scene s from camera cam into f (which must match cam's
// size) at time t. The scene's objects must already be at their poses
// for t (Scene.At).
func (r *Renderer) Render(s *Scene, cam Camera, t float64, f *Frame) {
	f.T, f.Cam = t, cam
	inf := float32(math.Inf(1))
	for i := range f.Depth {
		f.Depth[i], f.ID[i] = inf, 0
	}
	f.Poses = f.Poses[:0]
	ct := cam.R.T()
	for oi, o := range s.Objects {
		f.Poses = append(f.Poses, Pose{o.R, o.P})
		m := ct.Mul(o.R)
		off := ct.MulV(o.P.Sub(cam.P))
		r.cvs = r.cvs[:0]
		for _, v := range o.Mesh.V {
			r.cvs = append(r.cvs, cv{c: m.MulV(v.P).Add(off), loc: v.P, n: o.R.MulV(v.N), uv: v.UV})
		}
		ts := float64(o.Tex.Size)
		for _, tri := range o.Mesh.Tri {
			a, b, c := r.cvs[tri[0]], r.cvs[tri[1]], r.cvs[tri[2]]
			if a.c[0] < Near && b.c[0] < Near && c.c[0] < Near {
				continue
			}
			if outside(&cam, a.c, b.c, c.c) {
				continue
			}
			if a.c[0] >= Near && b.c[0] >= Near && c.c[0] >= Near {
				r.tri(f, &cam, uint16(oi+1), ts, a, b, c)
				continue
			}
			// Clip against the near plane (Sutherland–Hodgman, one plane).
			r.poly = append(r.poly[:0], a, b, c)
			r.tmp = r.tmp[:0]
			for i := range r.poly {
				p, q := r.poly[i], r.poly[(i+1)%len(r.poly)]
				pin, qin := p.c[0] >= Near, q.c[0] >= Near
				if pin {
					r.tmp = append(r.tmp, p)
				}
				if pin != qin {
					r.tmp = append(r.tmp, lerpCV(p, q, (Near-p.c[0])/(q.c[0]-p.c[0])))
				}
			}
			for i := 1; i+1 < len(r.tmp); i++ {
				r.tri(f, &cam, uint16(oi+1), ts, r.tmp[0], r.tmp[i], r.tmp[i+1])
			}
		}
	}
	shade(s, f)
}

// outside reports a triangle wholly beyond one side of the view frustum.
func outside(cam *Camera, a, b, c geom.V3) bool {
	kx := float64(cam.W) / 2 / cam.F
	ky := float64(cam.H) / 2 / cam.F
	side := func(p geom.V3) int {
		s := 0
		if p[2] > kx*p[0] {
			s |= 1
		}
		if p[2] < -kx*p[0] {
			s |= 2
		}
		if p[1] > ky*p[0] {
			s |= 4
		}
		if p[1] < -ky*p[0] {
			s |= 8
		}
		return s
	}
	return side(a)&side(b)&side(c) != 0
}

func (r *Renderer) tri(f *Frame, cam *Camera, id uint16, texSize float64, a, b, c cv) {
	var x, y, iw [3]float64
	vs := [3]*cv{&a, &b, &c}
	for i, v := range vs {
		iw[i] = 1 / v.c[0]
		x[i] = cam.Cx + cam.F*v.c[2]*iw[i]
		y[i] = cam.Cy - cam.F*v.c[1]*iw[i]
	}
	area := (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0])
	if math.Abs(area) < 1e-12 {
		return
	}
	minX := int(math.Max(0, math.Floor(math.Min(x[0], math.Min(x[1], x[2]))-0.5)))
	maxX := int(math.Min(float64(f.W-1), math.Ceil(math.Max(x[0], math.Max(x[1], x[2]))-0.5)))
	minY := int(math.Max(0, math.Floor(math.Min(y[0], math.Min(y[1], y[2]))-0.5)))
	maxY := int(math.Min(float64(f.H-1), math.Ceil(math.Max(y[0], math.Max(y[1], y[2]))-0.5)))
	if minX > maxX || minY > maxY {
		return
	}
	ia := 1 / area
	// Barycentric derivatives (constant over the triangle).
	dx := [3]float64{(y[1] - y[2]) * ia, (y[2] - y[0]) * ia, (y[0] - y[1]) * ia}
	dy := [3]float64{(x[2] - x[1]) * ia, (x[0] - x[2]) * ia, (x[1] - x[0]) * ia}
	var dIx, dIy, dUx, dUy, dVx, dVy float64
	for i, v := range vs {
		dIx += dx[i] * iw[i]
		dIy += dy[i] * iw[i]
		dUx += dx[i] * v.uv[0] * iw[i]
		dUy += dy[i] * v.uv[0] * iw[i]
		dVx += dx[i] * v.uv[1] * iw[i]
		dVy += dy[i] * v.uv[1] * iw[i]
	}
	const eps = -1e-9
	for py := minY; py <= maxY; py++ {
		fy := float64(py) + 0.5
		for px := minX; px <= maxX; px++ {
			fx := float64(px) + 0.5
			l0 := ((x[1]-fx)*(y[2]-fy) - (x[2]-fx)*(y[1]-fy)) * ia
			l1 := ((x[2]-fx)*(y[0]-fy) - (x[0]-fx)*(y[2]-fy)) * ia
			l2 := 1 - l0 - l1
			if l0 < eps || l1 < eps || l2 < eps {
				continue
			}
			inv := l0*iw[0] + l1*iw[1] + l2*iw[2]
			d := 1 / inv
			k := py*f.W + px
			if float32(d) >= f.Depth[k] {
				continue
			}
			w0, w1, w2 := l0*iw[0]*d, l1*iw[1]*d, l2*iw[2]*d
			f.Depth[k], f.ID[k] = float32(d), id
			u := w0*a.uv[0] + w1*b.uv[0] + w2*c.uv[0]
			v := w0*a.uv[1] + w1*b.uv[1] + w2*c.uv[1]
			f.UV[k] = [2]float32{float32(u), float32(v)}
			// Screen-space texture footprint → mip level.
			ux, uy := (dUx-u*dIx)*d, (dUy-u*dIy)*d
			vx, vy := (dVx-v*dIx)*d, (dVy-v*dIy)*d
			rho2 := math.Max(ux*ux+vx*vx, uy*uy+vy*vy) * texSize * texSize
			f.LOD[k] = float32(0.5 * math.Log2(math.Max(rho2, 1e-12)))
			for j := 0; j < 3; j++ {
				f.Local[k][j] = float32(w0*a.loc[j] + w1*b.loc[j] + w2*c.loc[j])
				f.Normal[k][j] = float32(w0*a.n[j] + w1*b.n[j] + w2*c.n[j])
			}
		}
	}
}

// shade fills Albedo, RGB, unit normals and the sky's ray directions.
func shade(s *Scene, f *Frame) {
	cam := &f.Cam
	sun := s.Sun
	for py := 0; py < f.H; py++ {
		for px := 0; px < f.W; px++ {
			k := py*f.W + px
			id := f.ID[k]
			f.Part[k] = -1
			if id == 0 {
				d := cam.Ray(float64(px)+0.5, float64(py)+0.5).Unit()
				f.Local[k] = [3]float32{float32(d[0]), float32(d[1]), float32(d[2])}
				f.Normal[k] = [3]float32{}
				c := SkyColour(s, d)
				f.Albedo[k], f.RGB[k] = c, c
				continue
			}
			o := s.Objects[id-1]
			if o.Self {
				f.Part[k] = int16(o.Link)
			}
			n := f.Normal[k]
			l := float32(math.Sqrt(float64(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])))
			if l > 0 {
				n = [3]float32{n[0] / l, n[1] / l, n[2] / l}
			}
			f.Normal[k] = n
			alb := o.Tex.Sample(f.UV[k][0], f.UV[k][1], f.LOD[k]).mul(o.Tint)
			lam := float32(math.Max(0, float64(n[0])*sun[0]+float64(n[1])*sun[1]+float64(n[2])*sun[2]))
			light := s.SkyCol.scale(0.5 + 0.5*n[1]).add(s.SunCol.scale(lam))
			f.Albedo[k], f.RGB[k] = alb, alb.mul(light)
		}
	}
}

// SkyColour is the background radiance in world direction d.
func SkyColour(s *Scene, d geom.V3) RGB {
	t := float32(math.Max(0, math.Min(1, 1.5*d[1])))
	return s.Horizon.scale(1 - t).add(s.Zenith.scale(t))
}

// World returns the world position of pixel k (false for sky).
func (f *Frame) World(k int) (geom.V3, bool) {
	id := f.ID[k]
	if id == 0 {
		return geom.V3{}, false
	}
	l := f.Local[k]
	p := f.Poses[id-1]
	return p.R.MulV(geom.V3{float64(l[0]), float64(l[1]), float64(l[2])}).Add(p.P), true
}
