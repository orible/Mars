// Package geom has the small amount of 3D vector algebra the simulator needs.
//
// World frame: X forward, Y up, Z to the robot's right (right-handed).
package geom

import "math"

// V3 is a 3D vector. It marshals to JSON as [x, y, z].
type V3 [3]float64

func (a V3) Add(b V3) V3        { return V3{a[0] + b[0], a[1] + b[1], a[2] + b[2]} }
func (a V3) Sub(b V3) V3        { return V3{a[0] - b[0], a[1] - b[1], a[2] - b[2]} }
func (a V3) Scale(s float64) V3 { return V3{a[0] * s, a[1] * s, a[2] * s} }
func (a V3) Dot(b V3) float64   { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
func (a V3) Norm() float64      { return math.Sqrt(a.Dot(a)) }

func (a V3) Cross(b V3) V3 {
	return V3{a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]}
}

// Unit returns a scaled to length 1 (or a unchanged if it is zero).
func (a V3) Unit() V3 {
	n := a.Norm()
	if n == 0 {
		return a
	}
	return a.Scale(1 / n)
}

// M3 is a row-major 3×3 matrix, used for rotations.
type M3 [3][3]float64

func Identity() M3 { return M3{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}} }

func (m M3) MulV(v V3) V3 {
	var r V3
	for i := 0; i < 3; i++ {
		r[i] = m[i][0]*v[0] + m[i][1]*v[1] + m[i][2]*v[2]
	}
	return r
}

func (m M3) Mul(n M3) M3 {
	var r M3
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			r[i][j] = m[i][0]*n[0][j] + m[i][1]*n[1][j] + m[i][2]*n[2][j]
		}
	}
	return r
}

func (m M3) T() M3 {
	var r M3
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			r[i][j] = m[j][i]
		}
	}
	return r
}

// AxisAngle is the rotation by ang radians about axis (Rodrigues' formula).
func AxisAngle(axis V3, ang float64) M3 {
	k := axis.Unit()
	c, s := math.Cos(ang), math.Sin(ang)
	K := M3{{0, -k[2], k[1]}, {k[2], 0, -k[0]}, {-k[1], k[0], 0}}
	var r M3
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			d := 0.0
			if i == j {
				d = 1
			}
			r[i][j] = c*d + (1-c)*k[i]*k[j] + s*K[i][j]
		}
	}
	return r
}

// SegmentClosest returns the closest points c1 on segment p1–q1 and c2 on
// segment p2–q2 (Ericson, Real-Time Collision Detection, 5.1.9). Either
// segment may be a single point.
func SegmentClosest(p1, q1, p2, q2 V3) (c1, c2 V3) {
	d1, d2, r := q1.Sub(p1), q2.Sub(p2), p1.Sub(p2)
	a, e, f := d1.Dot(d1), d2.Dot(d2), d2.Dot(r)
	clamp := func(x float64) float64 { return math.Max(0, math.Min(1, x)) }
	const eps = 1e-12
	var s, t float64
	switch {
	case a <= eps && e <= eps:
		return p1, p2
	case a <= eps:
		t = clamp(f / e)
	default:
		c := d1.Dot(r)
		if e <= eps {
			s = clamp(-c / a)
		} else {
			b := d1.Dot(d2)
			if den := a*e - b*b; den > eps {
				s = clamp((b*f - c*e) / den)
			}
			t = (b*s + f) / e
			if t < 0 {
				t, s = 0, clamp(-c/a)
			} else if t > 1 {
				t, s = 1, clamp((b-c)/a)
			}
		}
	}
	return p1.Add(d1.Scale(s)), p2.Add(d2.Scale(t))
}

// InertiaMul returns (R·diag(d)·Rᵀ)·v: a body-frame principal inertia d,
// rotated into the world by R, applied to the world vector v.
func InertiaMul(r M3, d, v V3) V3 {
	l := r.T().MulV(v)
	return r.MulV(V3{d[0] * l[0], d[1] * l[1], d[2] * l[2]})
}
