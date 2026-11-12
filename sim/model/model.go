// Package model compiles a body spec into a 3D articulated model:
// forward kinematics, cylinder lengths and their gradients with respect to
// every joint, and inverse dynamics (recursive Newton–Euler).
//
// Joint i (i ≥ 1) moves link i relative to its parent; its coordinate is
// q[i-1]. Link 0 is the pelvis, which is either bolted to the world (test
// stand) or free (floating base). With a floating base the generalised
// velocity is ν = [v, ω, q̇]: the world velocity of the pelvis frame
// origin, the pelvis angular velocity in the world, then the joint rates.
// The matching generalised forces are [force, moment about the pelvis
// origin, joint torques].
package model

import (
	"math"

	"mars/sim/body"
	"mars/sim/geom"
	"mars/sim/hydraulics"
)

// route is a compiled signed polyline (a cylinder's pins, or a cable).
type route struct {
	Sign  float64
	Links []int
	Pts   []geom.V3
}

// Cyl is a compiled cylinder.
type Cyl struct {
	Name              string
	BaseLink, RodLink int
	Base, Rod         geom.V3 // anchor points in their link frames (the cylinder body)
	Retracted         float64
	P                 hydraulics.Params
	Routes            []route // what its length is measured along: the pins, or its cables
	// Span lists the joints (as q indices) whose motion changes this
	// cylinder's length: those on the path between its two links. A
	// cylinder from pelvis to thigh spans both hip joints; one from
	// pelvis to shin would span hip and knee.
	Span []int
}

// Tendon is a compiled passive cable.
type Tendon struct {
	Name         string
	Route        route
	Rest, K, C   float64
	Span         []int
}

// Sensor is a pressure pad at a contact point. Root is the link the pad
// belongs to as a whole (the foot it is on), and Local is the pad's place
// on it in the neutral pose.
type Sensor struct {
	Contact int
	Root    int
	Local   geom.V3
}

// limitOmega sets each joint stop's stiffness from the inertia it stops:
// k = ω²·I, so a toe and a hip both bounce off their stops at about
// 95 Hz: stiff enough to hold a cylinder pushing into them, stable for the
// lightest toe at 50 µs steps (ω·dt = 0.03), and gentle enough on the
// reused mass matrix (MassEvery) to keep momentum drift small.
const limitOmega = 600 // rad/s

// Model is an immutable compiled body plus scratch space for dynamics.
// A Model must not be used from several goroutines at once.
type Model struct {
	Body     body.Body
	Names    []string
	Parent   []int
	Origin   []geom.V3
	Axis     []geom.V3
	Com      []geom.V3
	Inertia  []geom.V3
	Mass     []float64
	Lo, Hi   []float64
	Damping  []float64
	// Stiffness is each joint's passive spring towards 0; LimitK and
	// LimitC are its end stops.
	Stiffness, LimitK, LimitC []float64
	Cyl                       []Cyl
	Tendons                   []Tendon
	Sensors                   []Sensor
	Supply   hydraulics.Supply
	Contacts []Contact

	moves [][]bool // moves[l][j]: rotating joint j moves link l

	om, al, ao, f, n []geom.V3
	zero, unit, col  []float64
	comp             []spatialInertia
}

// Contact is a point on a link that collides with the ground.
type Contact struct {
	Link  int
	Point geom.V3 // in the link frame
}

// New validates and compiles a body.
func New(b body.Body) (*Model, error) {
	if err := b.Validate(); err != nil {
		return nil, err
	}
	n := len(b.Links)
	m := &Model{
		Body: b, Supply: b.Supply,
		Names: make([]string, n), Parent: make([]int, n),
		Origin: make([]geom.V3, n), Axis: make([]geom.V3, n), Com: make([]geom.V3, n), Inertia: make([]geom.V3, n),
		Mass: make([]float64, n), Lo: make([]float64, n), Hi: make([]float64, n), Damping: make([]float64, n),
		Stiffness: make([]float64, n), LimitK: make([]float64, n), LimitC: make([]float64, n),
		moves: make([][]bool, n),
		om:    make([]geom.V3, n), al: make([]geom.V3, n), ao: make([]geom.V3, n), f: make([]geom.V3, n), n: make([]geom.V3, n),
		zero: make([]float64, n-1), unit: make([]float64, n-1), col: make([]float64, n-1),
	}
	for i, l := range b.Links {
		m.Names[i], m.Mass[i], m.Com[i], m.Inertia[i] = l.Name, l.Mass, l.Com, l.Inertia
		m.Parent[i] = b.LinkIndex(l.Parent)
		if l.Joint != nil {
			m.Origin[i], m.Axis[i] = l.Joint.Origin, l.Joint.Axis.Unit()
			m.Lo[i], m.Hi[i], m.Damping[i], m.Stiffness[i] = l.Joint.Limits[0], l.Joint.Limits[1], l.Joint.Damping, l.Joint.Stiffness
		}
		m.moves[i] = make([]bool, n)
		for k := i; k > 0; k = m.Parent[k] {
			m.moves[i][k] = true
		}
		for _, p := range l.Contacts {
			m.Contacts = append(m.Contacts, Contact{Link: i, Point: p})
		}
	}
	compile := func(c body.Cable) route {
		r := route{Sign: c.Sign}
		for _, a := range c.Path {
			r.Links, r.Pts = append(r.Links, b.LinkIndex(a.Link)), append(r.Pts, a.Point)
		}
		return r
	}
	// A joint is spanned if it moves some anchor of a route relative to
	// that route's first anchor.
	span := func(rs []route) []int {
		var out []int
		for j := 1; j < n; j++ {
			for _, r := range rs {
				moved := false
				for _, l := range r.Links[1:] {
					moved = moved || m.moves[l][j] != m.moves[r.Links[0]][j]
				}
				if moved {
					out = append(out, j-1)
					break
				}
			}
		}
		return out
	}
	for _, c := range b.Cylinders {
		cy := Cyl{
			Name: c.Name, BaseLink: b.LinkIndex(c.Base.Link), RodLink: b.LinkIndex(c.Rod.Link),
			Base: c.Base.Point, Rod: c.Rod.Point, Retracted: c.RetractedLength, P: c.Params,
		}
		for _, r := range c.Routes() {
			cy.Routes = append(cy.Routes, compile(r))
		}
		cy.Span = span(cy.Routes)
		m.Cyl = append(m.Cyl, cy)
	}
	for _, t := range b.Tendons {
		r := compile(body.Cable{Sign: 1, Path: t.Path})
		m.Tendons = append(m.Tendons, Tendon{Name: t.Name, Route: r, Rest: t.RestLength, K: t.Stiffness, C: t.Damping, Span: span([]route{r})})
	}

	ps := m.NewPose()
	m.FK(make([]float64, n-1), ps)
	// Joint stops sized to the inertia each one stops (the subtree about
	// the joint axis, neutral pose).
	for l := 1; l < n; l++ {
		a := ps.Z[l]
		in := 0.0
		for i := 1; i < n; i++ {
			if !m.moves[i][l] {
				continue
			}
			c := m.Point(ps, i, m.Com[i]).Sub(ps.P[l])
			r := c.Cross(a)
			in += geom.InertiaMul(ps.R[i], m.Inertia[i], a).Dot(a) + m.Mass[i]*r.Dot(r)
		}
		m.LimitK[l], m.LimitC[l] = limitOmega*limitOmega*in, limitOmega*in
	}
	// Sensors: a pad's root is found by climbing from its link while the
	// parent carries no plain (sensorless) contacts: heel → talus bracket,
	// stopping below the shin.
	plain := make([]bool, n)
	first := make([]int, n) // index of each link's first contact
	for i, l := range b.Links {
		first[i] = -1
		sensed := map[int]bool{}
		for _, k := range l.Sensors {
			sensed[k] = true
		}
		plain[i] = len(l.Contacts) > len(sensed)
	}
	for c, ct := range m.Contacts {
		if first[ct.Link] < 0 {
			first[ct.Link] = c
		}
	}
	for i, l := range b.Links {
		root := i
		for root > 0 && !plain[m.Parent[root]] && m.Parent[root] > 0 {
			root = m.Parent[root]
		}
		for _, k := range l.Sensors {
			c := first[i] + k
			m.Sensors = append(m.Sensors, Sensor{Contact: c, Root: root, Local: m.Point(ps, i, m.Contacts[c].Point).Sub(ps.P[root])})
		}
	}
	return m, nil
}

// NDof is the number of joint coordinates.
func (m *Model) NDof() int { return len(m.Names) - 1 }

// Pose is the world placement of every link for one q.
type Pose struct {
	R []geom.M3 // link orientation
	P []geom.V3 // link frame origin (its joint centre)
	Z []geom.V3 // joint axis in the world (Z[0] unused)
}

func (m *Model) NewPose() *Pose {
	n := len(m.Names)
	return &Pose{R: make([]geom.M3, n), P: make([]geom.V3, n), Z: make([]geom.V3, n)}
}

// FK fills ps for joint coordinates q, with the pelvis at the world origin.
func (m *Model) FK(q []float64, ps *Pose) { m.FKBase(geom.Identity(), geom.V3{}, q, ps) }

// FKBase fills ps with the pelvis at orientation r0 and position p0.
func (m *Model) FKBase(r0 geom.M3, p0 geom.V3, q []float64, ps *Pose) {
	ps.R[0], ps.P[0] = r0, p0
	for i := 1; i < len(m.Names); i++ {
		p := m.Parent[i]
		ps.Z[i] = ps.R[p].MulV(m.Axis[i])
		ps.P[i] = ps.P[p].Add(ps.R[p].MulV(m.Origin[i]))
		ps.R[i] = ps.R[p].Mul(geom.AxisAngle(m.Axis[i], q[i-1]))
	}
}

// Point maps a point in a link frame to the world.
func (m *Model) Point(ps *Pose, link int, pt geom.V3) geom.V3 {
	return ps.P[link].Add(ps.R[link].MulV(pt))
}

// CylLength returns cylinder k's pin-to-pin length. If dLdq is non-nil it
// receives ∂L/∂q for each joint in the cylinder's Span; a cylinder force F
// (positive pushing the pins apart) produces joint torque F·∂L/∂q, so these
// are the cylinder's signed moment arms in the current pose.
func (m *Model) CylLength(ps *Pose, k int, dLdq []float64) float64 {
	return m.routeLength(ps, m.Cyl[k].Routes, m.Cyl[k].Span, dLdq)
}

// TendonLength is CylLength for passive tendon t.
func (m *Model) TendonLength(ps *Pose, t int, dLdq []float64) float64 {
	return m.routeLength(ps, []route{m.Tendons[t].Route}, m.Tendons[t].Span, dLdq)
}

// RoutePoints returns the world polylines of cylinder k's cables (nil for
// a plain cylinder) or, with tendon set, of tendon k.
func (m *Model) RoutePoints(ps *Pose, k int, tendon bool) [][]geom.V3 {
	rs := []route{}
	if tendon {
		rs = append(rs, m.Tendons[k].Route)
	} else if len(m.Body.Cylinders[k].Cables) > 0 {
		rs = m.Cyl[k].Routes
	}
	var out [][]geom.V3
	for _, r := range rs {
		var line []geom.V3
		for i := range r.Pts {
			line = append(line, m.Point(ps, r.Links[i], r.Pts[i]))
		}
		out = append(out, line)
	}
	return out
}

// routeLength sums signed polyline lengths and, if dLdq is non-nil, their
// derivatives with respect to the joints in span.
func (m *Model) routeLength(ps *Pose, rs []route, span []int, dLdq []float64) float64 {
	for i := range dLdq {
		dLdq[i] = 0
	}
	l := 0.0
	for _, r := range rs {
		prev := m.Point(ps, r.Links[0], r.Pts[0])
		for i := 1; i < len(r.Pts); i++ {
			p := m.Point(ps, r.Links[i], r.Pts[i])
			d := p.Sub(prev)
			n := d.Norm()
			l += r.Sign * n
			if dLdq != nil && n > 0 {
				u := d.Scale(1 / n)
				for si, q := range span {
					j := q + 1
					var v geom.V3
					if m.moves[r.Links[i]][j] {
						v = v.Add(ps.Z[j].Cross(p.Sub(ps.P[j])))
					}
					if m.moves[r.Links[i-1]][j] {
						v = v.Sub(ps.Z[j].Cross(prev.Sub(ps.P[j])))
					}
					dLdq[si] += r.Sign * u.Dot(v)
				}
			}
			prev = p
		}
	}
	return l
}

// RNEA computes the joint torques needed to produce accelerations qdd at
// velocities qd under gravity g (recursive Newton–Euler, world frame),
// with the pelvis fixed.
func (m *Model) RNEA(ps *Pose, qd, qdd []float64, g geom.V3, tau []float64) {
	m.rnea(ps, geom.V3{}, geom.V3{}, geom.V3{}, qd, qdd, g, tau)
}

// RNEAFloating is RNEA for a free pelvis: nu and nud are generalised
// velocity and acceleration ([v, ω, q̇] and its derivative) and gen
// receives the 6+NDof generalised forces that produce them.
func (m *Model) RNEAFloating(ps *Pose, nu, nud []float64, g geom.V3, gen []float64) {
	w0 := geom.V3{nu[3], nu[4], nu[5]}
	a0 := geom.V3{nud[0], nud[1], nud[2]}
	al0 := geom.V3{nud[3], nud[4], nud[5]}
	f0, n0 := m.rnea(ps, w0, al0, a0, nu[6:], nud[6:], g, gen[6:])
	copy(gen[0:3], f0[:])
	copy(gen[3:6], n0[:])
}

// rnea runs Newton–Euler with the pelvis moving at angular velocity w0,
// angular acceleration al0 and origin acceleration a0. It writes joint
// torques to tau and returns the force and the moment about the pelvis
// origin that the pelvis must receive.
func (m *Model) rnea(ps *Pose, w0, al0, a0 geom.V3, qd, qdd []float64, g geom.V3, tau []float64) (geom.V3, geom.V3) {
	n := len(m.Names)
	m.om[0], m.al[0], m.ao[0] = w0, al0, a0.Sub(g)
	for i := 1; i < n; i++ {
		p := m.Parent[i]
		z := ps.Z[i]
		d := ps.P[i].Sub(ps.P[p])
		wq := z.Scale(qd[i-1])
		m.om[i] = m.om[p].Add(wq)
		m.al[i] = m.al[p].Add(z.Scale(qdd[i-1])).Add(m.om[p].Cross(wq))
		m.ao[i] = m.ao[p].Add(m.al[p].Cross(d)).Add(m.om[p].Cross(m.om[p].Cross(d)))
	}
	for i := 0; i < n; i++ {
		c := ps.R[i].MulV(m.Com[i])
		om, al := m.om[i], m.al[i]
		acom := m.ao[i].Add(al.Cross(c)).Add(om.Cross(om.Cross(c)))
		F := acom.Scale(m.Mass[i])
		N := geom.InertiaMul(ps.R[i], m.Inertia[i], al).Add(om.Cross(geom.InertiaMul(ps.R[i], m.Inertia[i], om)))
		m.f[i] = F
		m.n[i] = N.Add(c.Cross(F))
	}
	for i := n - 1; i >= 1; i-- {
		tau[i-1] = ps.Z[i].Dot(m.n[i])
		p := m.Parent[i]
		m.f[p] = m.f[p].Add(m.f[i])
		m.n[p] = m.n[p].Add(m.n[i]).Add(ps.P[i].Sub(ps.P[p]).Cross(m.f[i]))
	}
	return m.f[0], m.n[0]
}

// MassMatrixFloating fills M ((6+NDof)×(6+NDof)) for a free pelvis.
func (m *Model) MassMatrixFloating(ps *Pose, M [][]float64) { m.crba(ps, M, 6) }

// MassMatrix fills M (NDof×NDof) for pose ps with the pelvis fixed.
func (m *Model) MassMatrix(ps *Pose, M [][]float64) { m.crba(ps, M, 0) }

// spatialInertia is a rigid body (or a composite of several) about the
// pelvis origin, in world axes: mass, first moment h = m·c and the
// rotational inertia J about the pelvis origin.
type spatialInertia struct {
	m float64
	h geom.V3
	J geom.M3
}

// apply returns the momentum (linear, and angular about the pelvis origin)
// of the body moving with angular velocity w while its point at the
// pelvis origin moves at v.
func (a *spatialInertia) apply(w, v geom.V3) (geom.V3, geom.V3) {
	return v.Scale(a.m).Add(w.Cross(a.h)), a.J.MulV(w).Add(a.h.Cross(v))
}

// crba fills the mass matrix with the composite rigid body algorithm:
// every entry is one joint's motion dotted with the momentum of the
// subtree below the other joint, so it costs O(links × depth) instead of
// one Newton–Euler pass per column. nb is 6 for a free pelvis (rows and
// columns 0–5 are the pelvis origin's linear and angular motion), else 0.
func (m *Model) crba(ps *Pose, M [][]float64, nb int) {
	n := len(m.Names)
	if len(m.comp) != n {
		m.comp = make([]spatialInertia, n)
	}
	for i := range M {
		for j := range M[i] {
			M[i][j] = 0
		}
	}
	p0 := ps.P[0]
	for i := 0; i < n; i++ {
		r := ps.P[i].Add(ps.R[i].MulV(m.Com[i])).Sub(p0)
		R, d, mi := ps.R[i], m.Inertia[i], m.Mass[i]
		var J geom.M3
		for a := 0; a < 3; a++ {
			for b := 0; b < 3; b++ {
				v := R[a][0]*d[0]*R[b][0] + R[a][1]*d[1]*R[b][1] + R[a][2]*d[2]*R[b][2] - mi*r[a]*r[b]
				if a == b {
					v += mi * r.Dot(r)
				}
				J[a][b] = v
			}
		}
		m.comp[i] = spatialInertia{mi, r.Scale(mi), J}
	}
	for i := n - 1; i >= 1; i-- {
		c, ch := &m.comp[m.Parent[i]], &m.comp[i]
		c.m += ch.m
		c.h = c.h.Add(ch.h)
		for a := 0; a < 3; a++ {
			for b := 0; b < 3; b++ {
				c.J[a][b] += ch.J[a][b]
			}
		}
	}
	motion := func(i int) (geom.V3, geom.V3) { // joint i's unit motion, as seen at the pelvis origin
		z := ps.Z[i]
		return z, ps.P[i].Sub(p0).Cross(z)
	}
	for i := 1; i < n; i++ {
		wi, vi := motion(i)
		f, mo := m.comp[i].apply(wi, vi)
		col := nb + i - 1
		for a := i; a >= 1; a = m.Parent[a] {
			wa, va := motion(a)
			x := mo.Dot(wa) + f.Dot(va)
			M[nb+a-1][col], M[col][nb+a-1] = x, x
		}
		if nb == 6 {
			for k := 0; k < 3; k++ {
				M[k][col], M[col][k] = f[k], f[k]
				M[3+k][col], M[col][3+k] = mo[k], mo[k]
			}
		}
	}
	if nb == 6 {
		for j := 0; j < 6; j++ {
			var w, v geom.V3
			if j < 3 {
				v[j] = 1
			} else {
				w[j-3] = 1
			}
			f, mo := m.comp[0].apply(w, v)
			for k := 0; k < 3; k++ {
				M[k][j], M[3+k][j] = f[k], mo[k]
			}
		}
	}
}

// massMatrixFloatingRNEA is MassMatrixFloating by one Newton–Euler pass
// per column: slow, kept as the reference the tests check crba against.
func (m *Model) massMatrixFloatingRNEA(ps *Pose, M [][]float64) {
	nd := m.NDof()
	for j := 0; j < 6+nd; j++ {
		var a0, al0 geom.V3
		for i := range m.unit {
			m.unit[i] = 0
		}
		switch {
		case j < 3:
			a0[j] = 1
		case j < 6:
			al0[j-3] = 1
		default:
			m.unit[j-6] = 1
		}
		f0, n0 := m.rnea(ps, geom.V3{}, al0, a0, m.zero, m.unit, geom.V3{}, m.col)
		for i := 0; i < 3; i++ {
			M[i][j], M[3+i][j] = f0[i], n0[i]
		}
		for i := range m.col {
			M[6+i][j] = m.col[i]
		}
	}
}

// PointVelocity is the world velocity of world point p fixed to link l,
// for pelvis origin velocity v0, pelvis angular velocity w0 and joint
// rates qd.
func (m *Model) PointVelocity(ps *Pose, v0, w0 geom.V3, qd []float64, l int, p geom.V3) geom.V3 {
	v := v0.Add(w0.Cross(p.Sub(ps.P[0])))
	for j := l; j > 0; j = m.Parent[j] {
		v = v.Add(ps.Z[j].Scale(qd[j-1]).Cross(p.Sub(ps.P[j])))
	}
	return v
}

// AddPointForce adds the generalised force of world force f applied at
// world point p on link l. gen holds 6 base entries first when floating
// is true, then one per joint.
func (m *Model) AddPointForce(ps *Pose, l int, p, f geom.V3, floating bool, gen []float64) {
	off := 0
	if floating {
		mo := p.Sub(ps.P[0]).Cross(f)
		for i := 0; i < 3; i++ {
			gen[i] += f[i]
			gen[3+i] += mo[i]
		}
		off = 6
	}
	for j := l; j > 0; j = m.Parent[j] {
		gen[off+j-1] += ps.Z[j].Dot(p.Sub(ps.P[j]).Cross(f))
	}
}

// massMatrixRNEA is MassMatrix by one RNEA call per column (test reference).
func (m *Model) massMatrixRNEA(ps *Pose, M [][]float64) {
	for j := range m.unit {
		for i := range m.unit {
			m.unit[i] = 0
		}
		m.unit[j] = 1
		m.RNEA(ps, m.zero, m.unit, geom.V3{}, m.col)
		for i := range m.col {
			M[i][j] = m.col[i]
		}
	}
}

// CoG returns the whole-body centre of gravity and total mass.
func (m *Model) CoG(ps *Pose) (geom.V3, float64) {
	var s geom.V3
	var mass float64
	for i := range m.Names {
		s = s.Add(m.Point(ps, i, m.Com[i]).Scale(m.Mass[i]))
		mass += m.Mass[i]
	}
	return s.Scale(1 / mass), mass
}

// solveSPD solves A·x = b in place (b becomes x) by Cholesky; A is overwritten.
func solveSPD(A [][]float64, b []float64) {
	cholesky(A)
	cholSolve(A, b)
}

// cholesky overwrites the lower triangle of A with L, where A = L·Lᵀ.
func cholesky(A [][]float64) {
	n := len(A)
	for j := 0; j < n; j++ {
		s := A[j][j]
		for k := 0; k < j; k++ {
			s -= A[j][k] * A[j][k]
		}
		s = math.Sqrt(math.Max(s, 1e-12))
		A[j][j] = s
		for i := j + 1; i < n; i++ {
			t := A[i][j]
			for k := 0; k < j; k++ {
				t -= A[i][k] * A[j][k]
			}
			A[i][j] = t / s
		}
	}
}

// cholSolve solves L·Lᵀ·x = b in place for a factor from cholesky.
func cholSolve(A [][]float64, b []float64) {
	n := len(b)
	for i := 0; i < n; i++ {
		t := b[i]
		for k := 0; k < i; k++ {
			t -= A[i][k] * b[k]
		}
		b[i] = t / A[i][i]
	}
	for i := n - 1; i >= 0; i-- {
		t := b[i]
		for k := i + 1; k < n; k++ {
			t -= A[k][i] * b[k]
		}
		b[i] = t / A[i][i]
	}
}

// TotalMass is the sum of the link masses.
func (m *Model) TotalMass() float64 {
	s := 0.0
	for _, x := range m.Mass {
		s += x
	}
	return s
}
