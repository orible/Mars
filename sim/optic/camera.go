package optic

import (
	"math"

	"mars/sim/geom"
	"mars/sim/model"
)

// Camera is a pinhole camera. R maps camera axes to the world (column 0
// is the viewing direction, 1 up, 2 right); P is the optical centre.
type Camera struct {
	R      geom.M3
	P      geom.V3
	W, H   int
	F      float64 // focal length in pixels (square pixels)
	Cx, Cy float64 // principal point in pixels
}

// Near is the near clipping distance in metres.
const Near = 0.05

// ToCam maps a world point to camera coordinates (forward, up, right).
func (c *Camera) ToCam(p geom.V3) geom.V3 { return c.R.T().MulV(p.Sub(c.P)) }

// Project maps a world point to pixel coordinates and depth (distance
// along the viewing direction). ok is false behind the near plane.
func (c *Camera) Project(p geom.V3) (u, v, depth float64, ok bool) {
	q := c.ToCam(p)
	return c.projectCam(q)
}

func (c *Camera) projectCam(q geom.V3) (u, v, depth float64, ok bool) {
	if q[0] < Near {
		return 0, 0, q[0], false
	}
	return c.Cx + c.F*q[2]/q[0], c.Cy - c.F*q[1]/q[0], q[0], true
}

// Ray is the world direction through pixel position (u, v), with unit
// forward component (so a point at depth d is P + d·Ray).
func (c *Camera) Ray(u, v float64) geom.V3 {
	return c.R.MulV(geom.V3{1, (c.Cy - v) / c.F, (u - c.Cx) / c.F})
}

// Rig is the stereo pair: two parallel (rectified) cameras Baseline apart
// along the head's right axis.
type Rig struct {
	W, H     int
	HFOV     float64 // horizontal field of view, radians
	Baseline float64 // metres
}

// DefaultRig is 128×96 per eye, 90° horizontal field of view (74°
// vertical) and a 6.5 cm baseline, about human eye spacing.
func DefaultRig() Rig { return Rig{W: 128, H: 96, HFOV: 90 * math.Pi / 180, Baseline: 0.065} }

// F is the focal length in pixels.
func (r Rig) F() float64 { return float64(r.W) / 2 / math.Tan(r.HFOV/2) }

// Eyes places the left and right cameras for a head at orientation h and
// cyclopean eye centre p.
func (r Rig) Eyes(h geom.M3, p geom.V3) (left, right Camera) {
	right3 := geom.V3{h[0][2], h[1][2], h[2][2]}
	mk := func(s float64) Camera {
		return Camera{R: h, P: p.Add(right3.Scale(s * r.Baseline / 2)), W: r.W, H: r.H, F: r.F(),
			Cx: float64(r.W) / 2, Cy: float64(r.H) / 2}
	}
	return mk(-1), mk(1)
}

// Disparity is the exact stereo disparity in pixels (left u minus right
// u) of a point at depth d. Depth error from a disparity error δ is
// d²·δ/(F·B): the sense sharpens with the square of approach.
func (r Rig) Disparity(d float64) float64 { return r.F() * r.Baseline / d }

// HeadMount says where the eyes sit on the body: the link that carries
// the head (the chest: the body has no neck, so gaze is the thorax
// orientation) and the cyclopean eye point in that link's frame.
type HeadMount struct {
	Link int
	Eye  geom.V3
}

// FindHead locates the head on any body: the link with the highest
// ground-contact point in the neutral pose (the top of the head), with
// the eyes 6.4 % of standing height below it and 5 % forward, the adult
// proportions. The camera axes are the link's axes, which in the neutral
// pose are the pelvis axes (forward, up, right).
func FindHead(m *model.Model) HeadMount {
	ps := m.NewPose()
	m.FK(make([]float64, m.NDof()), ps)
	best, link := math.Inf(-1), 0
	var top geom.V3
	lo := math.Inf(1)
	for _, c := range m.Contacts {
		p := m.Point(ps, c.Link, c.Point)
		lo = math.Min(lo, p[1])
		if p[1] > best {
			best, link, top = p[1], c.Link, p
		}
	}
	h := best - lo
	if len(m.Contacts) == 0 { // no contacts: the last link's origin
		link, top, h = len(m.Names)-1, ps.P[len(m.Names)-1], 1
	}
	eye := top.Add(geom.V3{0.05 * h, -0.064 * h, 0})
	return HeadMount{Link: link, Eye: ps.R[link].T().MulV(eye.Sub(ps.P[link]))}
}

// HeadPose is the head's orientation and cyclopean eye position in w.
func (hm HeadMount) HeadPose(w *model.World) (geom.M3, geom.V3) {
	return w.Pose.R[hm.Link], w.M.Point(w.Pose, hm.Link, hm.Eye)
}

// HeadAngles are the middleware inputs that turn eye-frame directions
// into body-frame ones: pitch (nose up +) and roll (right ear down +)
// against gravity, and yaw (left +) relative to the pelvis heading. With
// these, HeadingRotation maps an eye-frame direction to the heading frame
// (pelvis heading, gravity up).
func (hm HeadMount) HeadAngles(w *model.World) (pitch, yaw, roll float64) {
	r := w.Pose.R[hm.Link]
	return Angles(r, headingYaw(w.Base.R))
}

// Angles splits head orientation r into pitch, yaw relative to a heading
// of headYaw, and roll (see HeadAngles).
func Angles(r geom.M3, headYaw float64) (pitch, yaw, roll float64) {
	f := geom.V3{r[0][0], r[1][0], r[2][0]}
	u := geom.V3{r[0][1], r[1][1], r[2][1]}
	rt := geom.V3{r[0][2], r[1][2], r[2][2]}
	pitch = math.Asin(math.Max(-1, math.Min(1, f[1])))
	yaw = wrap(math.Atan2(-f[2], f[0]) - headYaw)
	roll = math.Atan2(-rt[1], u[1])
	return
}

func headingYaw(r geom.M3) float64 { return math.Atan2(-r[2][0], r[0][0]) }

func wrap(a float64) float64 {
	for a > math.Pi {
		a -= 2 * math.Pi
	}
	for a < -math.Pi {
		a += 2 * math.Pi
	}
	return a
}

// HeadingRotation is the eye-to-heading-frame rotation for the angles of
// HeadAngles: yaw about up, then pitch about right, then roll about
// forward.
func HeadingRotation(pitch, yaw, roll float64) geom.M3 {
	return geom.AxisAngle(geom.V3{0, 1, 0}, yaw).
		Mul(geom.AxisAngle(geom.V3{0, 0, 1}, pitch)).
		Mul(geom.AxisAngle(geom.V3{1, 0, 0}, roll))
}
