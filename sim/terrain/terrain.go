// Package terrain builds test worlds for the robot: uneven ground from
// octave noise, planar slopes, ramps, flights of stairs and scattered
// static objects. A Spec is the data (a scenario's "terrain" block); Build
// turns it, a seed and the places to keep clear into a Terrain that the
// contact model queries (model.Ground.Terrain).
//
// A Terrain is two things:
//
//   - a smooth height field (noise + slope + ramps), whose contact normal
//     is the true surface normal from its exact gradient, and
//   - solids standing on it (stair flights, boxes, cylinders) that do not
//     overlap each other. Their contacts use a signed distance: a point is
//     pushed out through the nearest open face, so a toe that runs into a
//     riser is pushed back horizontally and one that lands on a tread is
//     pushed up. Faces shared with a neighbour (inside a flight) and faces
//     buried in the ground are never used as exits, so nothing falls
//     through the seams.
//
// Headings follow the robot's yaw: 0° is +x, positive turns left (towards
// −z); "right" of a heading is (sin, 0, cos) in (x, y, z).
//
// A Terrain is immutable once built and safe to share between worlds and
// goroutines.
package terrain

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"mars/sim/geom"
)

// Spec describes a terrain. Every part is optional; an empty Spec is a
// flat floor at y = 0.
type Spec struct {
	// Seed fixes the noise and the scattered objects. 0 draws a new seed
	// for every episode (the scenario's random stream decides it).
	Seed int64 `json:"seed,omitempty"`
	// Ladder names a world from Ladders at difficulty Level (0 easy … 1
	// hard); it replaces every other part of the spec except Seed and
	// Scale. This is the knob for a curriculum.
	Ladder string  `json:"ladder,omitempty"`
	Level  float64 `json:"level,omitempty"`
	// Scale multiplies every length (not the angles): 1 (or 0) is sized
	// for a 1.8 m robot.
	Scale float64 `json:"scale,omitempty"`
	// Goal is where a walk across this world should end (x, z); scenarios
	// use it as the objective. Zero leaves the scenario's own objectives.
	Goal [2]float64 `json:"goal,omitempty"`
	// Noise is octave-noise ground; NoisePreset names one of NoisePresets
	// instead (Noise wins if both are set). NoiseScale multiplies the
	// amplitude of either, for sweeping difficulty (0 = 1).
	Noise       *Noise   `json:"noise,omitempty"`
	NoisePreset string   `json:"noisePreset,omitempty"`
	NoiseScale  float64  `json:"noiseScale,omitempty"`
	Slope       *Slope   `json:"slope,omitempty"`
	Ramps       []Ramp   `json:"ramps,omitempty"`
	Stairs      []Stairs `json:"stairs,omitempty"`
	// Objects are placed as given; Scatter adds random ones.
	Objects []Object `json:"objects,omitempty"`
	Scatter *Scatter `json:"scatter,omitempty"`
}

// Slope tilts the whole floor: a plane through the origin rising at
// AngleDeg towards HeadingDeg.
type Slope struct {
	AngleDeg   float64 `json:"angleDeg"`
	HeadingDeg float64 `json:"headingDeg"`
}

// Ramp is a band across the whole floor (unlimited width) that starts At
// a point, rises at AngleDeg towards HeadingDeg to Rise metres, runs flat
// for Landing metres, then (unless Plateau) goes down again at the same
// angle.
type Ramp struct {
	At         [2]float64 `json:"at"` // x, z of the foot of the ramp
	HeadingDeg float64    `json:"headingDeg"`
	AngleDeg   float64    `json:"angleDeg"`
	Rise       float64    `json:"rise"`
	Landing    float64    `json:"landing"`
	Plateau    bool       `json:"plateau,omitempty"` // stay up after the landing
}

// Stairs is a flight of Steps steps, each Rise high and Run deep, Width
// wide, whose first riser stands At a point facing back along
// HeadingDeg. Going up, the top step is lengthened by Landing (and, with
// Over, the same steps lead down again on the far side). Going Down, the
// robot starts on a platform Steps·Rise high that reaches Platform metres
// back from the first edge, and the steps lead down from it.
type Stairs struct {
	At         [2]float64 `json:"at"`
	HeadingDeg float64    `json:"headingDeg"`
	Rise       float64    `json:"rise"`
	Run        float64    `json:"run"`
	Steps      int        `json:"steps"`
	Width      float64    `json:"width"`
	Down       bool       `json:"down,omitempty"`
	Landing    float64    `json:"landing,omitempty"`
	Over       bool       `json:"over,omitempty"`
	Platform   float64    `json:"platform,omitempty"`
}

// Object is one static primitive standing on the floor.
type Object struct {
	Kind   string     `json:"kind"` // "box" or "cylinder"
	At     [2]float64 `json:"at"`   // x, z of its centre
	YawDeg float64    `json:"yawDeg,omitempty"`
	// Size is the box's length (along its yaw), height and width; a
	// cylinder's diameter, height and diameter.
	Size [3]float64 `json:"size"`
	// Seed is a stable appearance seed for renderers (colour, texture).
	Seed uint32 `json:"seed,omitempty"`
}

// Scatter places Count random objects inside a rectangle, keeping Clear
// metres from the start, every objective and the stairs, and Gap metres
// from each other.
type Scatter struct {
	Count  int        `json:"count"`
	Min    [2]float64 `json:"min"` // x, z corner of the area
	Max    [2]float64 `json:"max"`
	Size   [2]float64 `json:"size"`            // m, footprint range
	Height [2]float64 `json:"height"`          // m
	Kinds  []string   `json:"kinds,omitempty"` // default box and cylinder
	Clear  float64    `json:"clear"`
	Gap    float64    `json:"gap,omitempty"`
}

// Primitive is a solid in world coordinates for drawing and rendering.
type Primitive struct {
	Kind   string  `json:"kind"` // "box" or "cylinder"
	Role   string  `json:"role"` // "object", "step" or "platform"
	Center geom.V3 `json:"center"`
	YawDeg float64 `json:"yawDeg"`
	// Size is the full extent along the primitive's own axes: length
	// (along yaw), height, width; for a cylinder diameter, height, diameter.
	Size geom.V3 `json:"size"`
	Seed uint32  `json:"seed"`
}

// Terrain is a built Spec.
type Terrain struct {
	Spec Spec  // as built (scattered objects listed in Spec.Objects)
	Seed int64 // the seed used

	noise  *noiseField
	slope  [2]float64 // ∂h/∂x, ∂h/∂z
	ramps  []ramp
	solids []solid
	prims  []Primitive
	top    float64 // highest solid top
}

// Validate rejects specs that cannot be built.
func (s Spec) Validate() error {
	if s.Ladder != "" {
		if _, ok := Ladders[s.Ladder]; !ok {
			return fmt.Errorf("terrain: unknown ladder %q (%s)", s.Ladder, strings.Join(LadderNames(), ", "))
		}
		if s.Level < 0 || s.Level > 1 {
			return fmt.Errorf("terrain: level must be within 0–1")
		}
	}
	if s.Scale < 0 {
		return fmt.Errorf("terrain: scale must not be negative")
	}
	if s.Noise != nil && (s.Noise.Wavelength <= 0 || s.Noise.Amplitude < 0) {
		return fmt.Errorf("terrain: noise needs a positive wavelength")
	}
	if s.NoisePreset != "" {
		if _, ok := NoisePresets[s.NoisePreset]; !ok {
			return fmt.Errorf("terrain: unknown noise preset %q", s.NoisePreset)
		}
	}
	if s.Slope != nil && math.Abs(s.Slope.AngleDeg) >= 60 {
		return fmt.Errorf("terrain: slope must be under 60°")
	}
	for i, r := range s.Ramps {
		if r.AngleDeg <= 0 || r.AngleDeg >= 60 || r.Rise <= 0 || r.Landing < 0 {
			return fmt.Errorf("terrain: ramp %d needs 0 < angle < 60°, rise > 0, landing ≥ 0", i)
		}
	}
	for i, st := range s.Stairs {
		if st.Rise <= 0 || st.Run <= 0 || st.Steps < 1 || st.Width <= 0 || st.Landing < 0 || st.Platform < 0 {
			return fmt.Errorf("terrain: stairs %d need positive rise, run, steps and width", i)
		}
	}
	for i, o := range s.Objects {
		if (o.Kind != "box" && o.Kind != "cylinder") || o.Size[0] <= 0 || o.Size[1] <= 0 || o.Size[2] <= 0 {
			return fmt.Errorf("terrain: object %d needs kind box or cylinder and a positive size", i)
		}
	}
	if sc := s.Scatter; sc != nil {
		if sc.Count < 0 || sc.Max[0] <= sc.Min[0] || sc.Max[1] <= sc.Min[1] || sc.Size[0] <= 0 || sc.Size[1] < sc.Size[0] || sc.Height[0] <= 0 || sc.Height[1] < sc.Height[0] {
			return fmt.Errorf("terrain: scatter needs an area, a size range and a height range")
		}
		for _, k := range sc.Kinds {
			if k != "box" && k != "cylinder" {
				return fmt.Errorf("terrain: scatter kind %q (box or cylinder)", k)
			}
		}
	}
	return nil
}

// Resolve validates s and expands its ladder and scale, giving the plain
// spec that Build uses (its Goal is in scaled world coordinates).
func (s Spec) Resolve() (Spec, error) {
	if err := s.Validate(); err != nil {
		return Spec{}, err
	}
	if s.Ladder != "" {
		l, _ := Ladder(s.Ladder, s.Level)
		l.Seed, l.Scale = s.Seed, s.Scale
		s = l
	}
	k := s.Scale
	if k == 0 {
		k = 1
	}
	s = s.scaled(k)
	return s, s.Validate()
}

// Build makes the terrain. seed is used when the spec has none; keep
// lists the floor points scattered objects must stay Clear of (the start
// is always kept clear).
func Build(s Spec, seed int64, keep [][2]float64) (*Terrain, error) {
	s, err := s.Resolve()
	if err != nil {
		return nil, err
	}
	if s.Seed != 0 {
		seed = s.Seed
	}
	t := &Terrain{Seed: seed, top: math.Inf(-1)}
	n := s.Noise
	if n == nil && s.NoisePreset != "" {
		p := NoisePresets[s.NoisePreset]
		n = &p
	}
	if n != nil {
		nn := *n
		if s.NoiseScale > 0 {
			nn.Amplitude *= s.NoiseScale
		}
		t.noise = newNoiseField(nn, seed)
	}
	if sl := s.Slope; sl != nil {
		g := math.Tan(rad(sl.AngleDeg))
		c, si := heading(sl.HeadingDeg)
		t.slope = [2]float64{g * c, -g * si} // h = g·u, u = x·cos − z·sin
	}
	for _, r := range s.Ramps {
		t.ramps = append(t.ramps, newRamp(r))
	}
	// Solids sit on the field: their bottoms are buried a little below it.
	for _, st := range s.Stairs {
		f := t.newFlight(st)
		t.solids = append(t.solids, f)
		t.prims = append(t.prims, f.prims()...)
	}
	objs := append([]Object(nil), s.Objects...)
	if sc := s.Scatter; sc != nil {
		objs = append(objs, t.scatter(*sc, rand.New(rand.NewSource(seed+7919)), keep, objs)...)
	}
	for i := range objs {
		if objs[i].Seed == 0 {
			objs[i].Seed = uint32(seed)*2654435761 + uint32(i+1)*40503
		}
	}
	built := s
	built.Objects = objs
	built.Scatter = nil
	t.Spec = built
	for _, o := range objs {
		sd, p := t.newObject(o)
		t.solids = append(t.solids, sd)
		t.prims = append(t.prims, p)
	}
	for _, sd := range t.solids {
		t.top = math.Max(t.top, sd.bound().top)
	}
	return t, nil
}

// Primitives lists every solid (stair steps and platforms as boxes, then
// objects) in world coordinates, for observers and renderers.
func (t *Terrain) Primitives() []Primitive { return append([]Primitive(nil), t.prims...) }

// HasField reports whether the floor between the solids is anything but
// flat at y = 0.
func (t *Terrain) HasField() bool {
	return t.noise != nil || t.slope != [2]float64{} || len(t.ramps) > 0
}

// Field is the smooth floor's height under (x, z) and its slope, without
// the solids.
func (t *Terrain) Field(x, z float64) (h, gx, gz float64) {
	if t.noise != nil {
		h, gx, gz = t.noise.eval(x, z)
	}
	h += t.slope[0]*x + t.slope[1]*z
	gx += t.slope[0]
	gz += t.slope[1]
	for i := range t.ramps {
		rh, rx, rz := t.ramps[i].eval(x, z)
		h, gx, gz = h+rh, gx+rx, gz+rz
	}
	return
}

// Height is the top surface under (x, z): the field, or the top of a
// solid standing there.
func (t *Terrain) Height(x, z float64) float64 {
	h, _, _ := t.Field(x, z)
	for _, s := range t.solids {
		if top, ok := s.topAt(x, z); ok && top > h {
			h = top
		}
	}
	return h
}

// Distance is the signed distance from p to the terrain (negative inside)
// and the outward surface normal there. bh, bgx and bgz add a further
// height (and its slope) at p's x, z to the field, such as the floor's
// sine-wave bumps. Solids farther than reach are skipped, so a result
// above reach may overestimate.
func (t *Terrain) Distance(p geom.V3, reach, bh, bgx, bgz float64) (float64, geom.V3) {
	h, gx, gz := t.Field(p[0], p[2])
	h, gx, gz = h+bh, gx+bgx, gz+bgz
	inv := 1 / math.Sqrt(1+gx*gx+gz*gz)
	d, n := (p[1]-h)*inv, geom.V3{-gx * inv, inv, -gz * inv}
	if p[1]-reach > t.top {
		return d, n
	}
	for _, s := range t.solids {
		b := s.bound()
		if p[1]-reach > b.top {
			continue
		}
		if dx, dz := p[0]-b.x, p[2]-b.z; dx*dx+dz*dz > (b.r+reach)*(b.r+reach) {
			continue
		}
		if ds, ns := s.distance(p); ds < d {
			d, n = ds, ns
		}
	}
	return d, n
}

// ---------------------------------------------------------------- field

type ramp struct {
	x0, z0, c, s float64
	g            float64 // slope
	up, flat     float64 // u where the rise ends and where the landing ends
	rise         float64
	plateau      bool
}

func newRamp(r Ramp) ramp {
	c, s := heading(r.HeadingDeg)
	g := math.Tan(rad(r.AngleDeg))
	up := r.Rise / g
	return ramp{x0: r.At[0], z0: r.At[1], c: c, s: s, g: g, up: up, flat: up + r.Landing, rise: r.Rise, plateau: r.Plateau}
}

// eval is the ramp's height profile along its heading and its slope.
func (r *ramp) eval(x, z float64) (h, gx, gz float64) {
	u := (x-r.x0)*r.c - (z-r.z0)*r.s
	var dh float64 // dh/du
	switch {
	case u <= 0:
		return 0, 0, 0
	case u < r.up:
		h, dh = r.g*u, r.g
	case u <= r.flat || r.plateau:
		h = r.rise
	case u < r.flat+r.up:
		h, dh = r.rise-r.g*(u-r.flat), -r.g
	default:
		return 0, 0, 0
	}
	return h, dh * r.c, -dh * r.s
}

// --------------------------------------------------------------- solids

type bounds struct{ x, z, r, top float64 }

type solid interface {
	distance(p geom.V3) (float64, geom.V3)
	topAt(x, z float64) (float64, bool)
	bound() bounds
}

// frame is a solid's local frame: origin, and heading (u forward, w right).
type frame struct {
	o    geom.V3
	c, s float64
}

func (f frame) local(p geom.V3) (u, y, w float64) {
	dx, dz := p[0]-f.o[0], p[2]-f.o[2]
	return dx*f.c - dz*f.s, p[1] - f.o[1], dx*f.s + dz*f.c
}

func (f frame) world(nu, ny, nw float64) geom.V3 {
	return geom.V3{nu*f.c + nw*f.s, ny, -nu*f.s + nw*f.c}
}

// boxSDF is the exact outside distance to an axis-aligned box in local
// coordinates (half sizes hu, hy, hw about the local origin) and its
// outward direction. Inside it returns the (negative) distance to the
// nearest face except the bottom, which is buried.
func boxSDF(u, y, w, hu, hy, hw float64) (float64, [3]float64) {
	q := [3]float64{math.Abs(u) - hu, math.Abs(y) - hy, math.Abs(w) - hw}
	sg := [3]float64{sign(u), sign(y), sign(w)}
	if q[0] > 0 || q[1] > 0 || q[2] > 0 {
		var o [3]float64
		l := 0.0
		for k := range q {
			if q[k] > 0 {
				o[k] = q[k] * sg[k]
				l += q[k] * q[k]
			}
		}
		l = math.Sqrt(l)
		return l, [3]float64{o[0] / l, o[1] / l, o[2] / l}
	}
	// Inside: the top and the four sides are exits.
	best, n := hy-y, [3]float64{0, 1, 0}
	if d := -q[0]; d < best {
		best, n = d, [3]float64{sg[0], 0, 0}
	}
	if d := -q[2]; d < best {
		best, n = d, [3]float64{0, 0, sg[2]}
	}
	return -best, n
}

func sign(v float64) float64 {
	if v < 0 {
		return -1
	}
	return 1
}

// box is a yawed box from bottom (buried) to top.
type box struct {
	f          frame // origin at the centre of the box
	hu, hy, hw float64
	b          bounds
}

func (b *box) distance(p geom.V3) (float64, geom.V3) {
	u, y, w := b.f.local(p)
	d, n := boxSDF(u, y, w, b.hu, b.hy, b.hw)
	return d, b.f.world(n[0], n[1], n[2])
}

func (b *box) topAt(x, z float64) (float64, bool) {
	u, _, w := b.f.local(geom.V3{x, 0, z})
	return b.f.o[1] + b.hy, math.Abs(u) <= b.hu && math.Abs(w) <= b.hw
}

func (b *box) bound() bounds { return b.b }

// cylinder is an upright cylinder from bottom (buried) to top.
type cylinder struct {
	o     geom.V3 // centre
	r, hy float64
	b     bounds
}

func (c *cylinder) distance(p geom.V3) (float64, geom.V3) {
	dx, dy, dz := p[0]-c.o[0], p[1]-c.o[1], p[2]-c.o[2]
	rho := math.Hypot(dx, dz)
	radial := geom.V3{1, 0, 0}
	if rho > 1e-12 {
		radial = geom.V3{dx / rho, 0, dz / rho}
	}
	qr, qy := rho-c.r, math.Abs(dy)-c.hy
	switch {
	case qr > 0 && qy > 0:
		l := math.Hypot(qr, qy)
		return l, radial.Scale(qr / l).Add(geom.V3{0, sign(dy) * qy / l, 0})
	case qr > 0:
		return qr, radial
	case qy > 0:
		return qy, geom.V3{0, sign(dy), 0}
	}
	if top := c.hy - dy; top < -qr { // inside: out through the top or the side
		return -top, geom.V3{0, 1, 0}
	}
	return qr, radial
}

func (c *cylinder) topAt(x, z float64) (float64, bool) {
	return c.o[1] + c.hy, math.Hypot(x-c.o[0], z-c.o[2]) <= c.r
}

func (c *cylinder) bound() bounds { return c.b }

// flight is a run of adjacent columns sharing one heading and width: the
// steps of a staircase and its landing or platform.
type flight struct {
	f    frame // origin at the first riser's foot, y = base (buried)
	hw   float64
	segs []seg
	b    bounds
}

type seg struct {
	u0, u1, top float64 // top above f.o[1]
	role        string  // "step" or "platform"
}

// buried is how far below the floor a solid's bottom reaches, so it joins
// the ground without a seam.
const buried = 0.3

func (t *Terrain) newFlight(st Stairs) *flight {
	c, s := heading(st.HeadingDeg)
	y0, _, _ := t.Field(st.At[0], st.At[1])
	f := &flight{f: frame{o: geom.V3{st.At[0], y0 - buried, st.At[1]}, c: c, s: s}, hw: st.Width / 2}
	add := func(u0, u1, top float64) {
		if u1 > u0 && top > 0 {
			role := "step"
			if u1-u0 > st.Run*1.01 {
				role = "platform"
			}
			f.segs = append(f.segs, seg{u0, u1, top + buried, role})
		}
	}
	n := st.Steps
	if st.Down {
		add(-st.Platform, 0, float64(n)*st.Rise)
		for i := 0; i < n; i++ {
			add(float64(i)*st.Run, float64(i+1)*st.Run, float64(n-1-i)*st.Rise)
		}
	} else {
		for i := 0; i < n; i++ {
			u1 := float64(i+1) * st.Run
			if i == n-1 {
				u1 += st.Landing
			}
			add(float64(i)*st.Run, u1, float64(i+1)*st.Rise)
		}
		if st.Over {
			u := float64(n)*st.Run + st.Landing
			for i := 0; i < n-1; i++ {
				add(u+float64(i)*st.Run, u+float64(i+1)*st.Run, float64(n-1-i)*st.Rise)
			}
		}
	}
	lo, hi := f.segs[0].u0, f.segs[len(f.segs)-1].u1
	mid := (lo + hi) / 2
	cw := f.f.world(mid, 0, 0)
	top := 0.0
	for _, sg := range f.segs {
		top = math.Max(top, sg.top)
	}
	f.b = bounds{x: f.f.o[0] + cw[0], z: f.f.o[2] + cw[2], r: math.Hypot((hi-lo)/2, f.hw), top: f.f.o[1] + top}
	return f
}

func (f *flight) bound() bounds { return f.b }

// at returns the segment index under u, or −1.
func (f *flight) at(u float64) int {
	for i, sg := range f.segs {
		if u >= sg.u0 && u < sg.u1 {
			return i
		}
	}
	return -1
}

func (f *flight) topAt(x, z float64) (float64, bool) {
	u, _, w := f.f.local(geom.V3{x, 0, z})
	if math.Abs(w) > f.hw {
		return 0, false
	}
	if i := f.at(u); i >= 0 {
		return f.f.o[1] + f.segs[i].top, true
	}
	return 0, false
}

func (f *flight) distance(p geom.V3) (float64, geom.V3) {
	u, y, w := f.f.local(p)
	if i := f.at(u); i >= 0 && y > 0 && y < f.segs[i].top && math.Abs(w) < f.hw {
		// Inside column i: out through the tread, a side, or the nearest
		// riser face that is open at this height (towards a lower column
		// or off the end of the flight); faces shared with a taller
		// neighbour are not exits.
		best, n := f.segs[i].top-y, [3]float64{0, 1, 0}
		if d := f.hw - math.Abs(w); d < best {
			best, n = d, [3]float64{0, 0, sign(w)}
		}
		for j := i; j >= 0; j-- {
			if j == 0 || f.segs[j-1].top <= y || f.segs[j-1].u1 < f.segs[j].u0 {
				if d := u - f.segs[j].u0; d < best {
					best, n = d, [3]float64{-1, 0, 0}
				}
				break
			}
		}
		for j := i; j < len(f.segs); j++ {
			if j == len(f.segs)-1 || f.segs[j+1].top <= y || f.segs[j+1].u0 > f.segs[j].u1 {
				if d := f.segs[j].u1 - u; d < best {
					best, n = d, [3]float64{1, 0, 0}
				}
				break
			}
		}
		return -best, f.f.world(n[0], n[1], n[2])
	}
	// Outside: the union of the columns, exactly (the nearest column).
	best, n := math.Inf(1), [3]float64{0, 1, 0}
	for _, sg := range f.segs {
		hu, hy := (sg.u1-sg.u0)/2, sg.top/2
		d, dn := boxSDF(u-(sg.u0+hu), y-hy, w, hu, hy, f.hw)
		if d < best {
			best, n = d, dn
		}
	}
	return best, f.f.world(n[0], n[1], n[2])
}

func (f *flight) prims() []Primitive {
	var out []Primitive
	for i, sg := range f.segs {
		hu := (sg.u1 - sg.u0) / 2
		c := f.f.world(sg.u0+hu, 0, 0)
		out = append(out, Primitive{Kind: "box", Role: sg.role,
			Center: geom.V3{f.f.o[0] + c[0], f.f.o[1] + sg.top/2, f.f.o[2] + c[2]},
			YawDeg: math.Atan2(f.f.s, f.f.c) * 180 / math.Pi, Size: geom.V3{2 * hu, sg.top, 2 * f.hw}, Seed: uint32(i)})
	}
	return out
}

// newObject stands an object on the field, its bottom buried below the
// lowest point of the field under its footprint.
func (t *Terrain) newObject(o Object) (solid, Primitive) {
	c, s := heading(o.YawDeg)
	r := 0.5 * math.Hypot(o.Size[0], o.Size[2])
	lo := math.Inf(1)
	for k := 0; k < 9; k++ { // centre and eight points round the footprint
		x, z := o.At[0], o.At[1]
		if k > 0 {
			a := float64(k) * math.Pi / 4
			x, z = x+r*math.Cos(a), z+r*math.Sin(a)
		}
		h, _, _ := t.Field(x, z)
		lo = math.Min(lo, h)
	}
	hc, _, _ := t.Field(o.At[0], o.At[1])
	bottom, top := lo-buried, hc+o.Size[1] // its height stands above the ground at its centre
	cy, hy := (bottom+top)/2, (top-bottom)/2
	b := bounds{x: o.At[0], z: o.At[1], r: r, top: top}
	p := Primitive{Kind: o.Kind, Role: "object", Center: geom.V3{o.At[0], cy, o.At[1]}, YawDeg: o.YawDeg,
		Size: geom.V3{o.Size[0], 2 * hy, o.Size[2]}, Seed: o.Seed}
	if o.Kind == "cylinder" {
		return &cylinder{o: geom.V3{o.At[0], cy, o.At[1]}, r: o.Size[0] / 2, hy: hy, b: bounds{x: o.At[0], z: o.At[1], r: o.Size[0] / 2, top: top}}, p
	}
	return &box{f: frame{o: geom.V3{o.At[0], cy, o.At[1]}, c: c, s: s}, hu: o.Size[0] / 2, hy: hy, hw: o.Size[2] / 2, b: b}, p
}

// scatter draws random objects that keep clear of the start, keep, the
// stairs and each other (and of the objects already placed).
func (t *Terrain) scatter(sc Scatter, rng *rand.Rand, keep [][2]float64, placed []Object) []Object {
	kinds := sc.Kinds
	if len(kinds) == 0 {
		kinds = []string{"box", "cylinder"}
	}
	u := func(lo, hi float64) float64 { return lo + rng.Float64()*(hi-lo) }
	clearOf := append([][2]float64{{0, 0}}, keep...)
	var taken []Object
	taken = append(taken, placed...)
	var out []Object
	for tries := 0; len(out) < sc.Count && tries < 200*max(1, sc.Count); tries++ {
		o := Object{Kind: kinds[rng.Intn(len(kinds))], At: [2]float64{u(sc.Min[0], sc.Max[0]), u(sc.Min[1], sc.Max[1])}, YawDeg: u(0, 360)}
		a, h := u(sc.Size[0], sc.Size[1]), u(sc.Height[0], sc.Height[1])
		if o.Kind == "cylinder" {
			o.Size, o.YawDeg = [3]float64{a, h, a}, 0
		} else {
			o.Size = [3]float64{a, h, u(sc.Size[0], sc.Size[1])}
		}
		o.Seed = rng.Uint32() | 1
		r := 0.5 * math.Hypot(o.Size[0], o.Size[2])
		ok := true
		for _, k := range clearOf {
			if math.Hypot(o.At[0]-k[0], o.At[1]-k[1]) < sc.Clear+r {
				ok = false
			}
		}
		for _, q := range taken {
			if footprintDist(q, o.At[0], o.At[1]) < r+sc.Gap {
				ok = false
			}
		}
		for _, s := range t.solids {
			if f, isFlight := s.(*flight); isFlight && f.near(o.At[0], o.At[1], r+sc.Gap) {
				ok = false
			}
		}
		if !ok {
			continue
		}
		taken = append(taken, o)
		out = append(out, o)
	}
	return out
}

// FootprintDistance is how far (x, z) is from the primitive's footprint
// on the floor (0 inside it).
func (p Primitive) FootprintDistance(x, z float64) float64 {
	return footprintDist(Object{Kind: p.Kind, At: [2]float64{p.Center[0], p.Center[2]}, YawDeg: p.YawDeg, Size: [3]float64{p.Size[0], p.Size[1], p.Size[2]}}, x, z)
}

// footprintDist is how far (x, z) is from object o's footprint on the
// floor (0 inside it): a long wall keeps things off its length only.
func footprintDist(o Object, x, z float64) float64 {
	dx, dz := x-o.At[0], z-o.At[1]
	if o.Kind == "cylinder" {
		return math.Max(0, math.Hypot(dx, dz)-o.Size[0]/2)
	}
	c, s := heading(o.YawDeg)
	u, w := dx*c-dz*s, dx*s+dz*c
	return math.Hypot(math.Max(0, math.Abs(u)-o.Size[0]/2), math.Max(0, math.Abs(w)-o.Size[2]/2))
}

// near reports whether (x, z) is within m of the flight's footprint.
func (f *flight) near(x, z, m float64) bool {
	u, _, w := f.f.local(geom.V3{x, 0, z})
	lo, hi := f.segs[0].u0, f.segs[len(f.segs)-1].u1
	return u > lo-m && u < hi+m && math.Abs(w) < f.hw+m
}

// ------------------------------------------------------------- helpers

func rad(d float64) float64 { return d * math.Pi / 180 }

// heading returns cos and sin of a heading in degrees; the direction is
// (cos, 0, −sin).
func heading(deg float64) (c, s float64) { return math.Cos(rad(deg)), math.Sin(rad(deg)) }
