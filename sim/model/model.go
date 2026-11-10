// Package model compiles a body spec into a 3D articulated model:
// forward kinematics, cylinder lengths and their gradients with respect to
// every joint, and inverse dynamics (recursive Newton–Euler).
//
// Joint i (i ≥ 1) moves link i relative to its parent; its coordinate is
// q[i-1]. Link 0 is the pelvis. In the current fixed-base (test stand)
// mode the pelvis is bolted to the world; a floating base with foot
// contacts is the next step.
package model

import (
	"math"

	"mars/sim/body"
	"mars/sim/geom"
	"mars/sim/hydraulics"
)

// Cyl is a compiled cylinder.
type Cyl struct {
	Name              string
	BaseLink, RodLink int
	Base, Rod         geom.V3 // anchor points in their link frames
	Retracted         float64
	P                 hydraulics.Params
	// Span lists the joints (as q indices) whose motion changes this
	// cylinder's length: those on the path between its two links. A
	// cylinder from pelvis to thigh spans both hip joints; one from
	// pelvis to shin would span hip and knee.
	Span []int
}

// Model is an immutable compiled body plus scratch space for dynamics.
// A Model must not be used from several goroutines at once.
type Model struct {
	Body    body.Body
	Names   []string
	Parent  []int
	Origin  []geom.V3
	Axis    []geom.V3
	Com     []geom.V3
	Inertia []geom.V3
	Mass    []float64
	Lo, Hi  []float64
	Damping []float64
	Cyl     []Cyl
	Supply  hydraulics.Supply

	moves [][]bool // moves[l][j]: rotating joint j moves link l

	om, al, ao, f, n []geom.V3
	zero, unit, col  []float64
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
		moves: make([][]bool, n),
		om:    make([]geom.V3, n), al: make([]geom.V3, n), ao: make([]geom.V3, n), f: make([]geom.V3, n), n: make([]geom.V3, n),
		zero: make([]float64, n-1), unit: make([]float64, n-1), col: make([]float64, n-1),
	}
	for i, l := range b.Links {
		m.Names[i], m.Mass[i], m.Com[i], m.Inertia[i] = l.Name, l.Mass, l.Com, l.Inertia
		m.Parent[i] = b.LinkIndex(l.Parent)
		if l.Joint != nil {
			m.Origin[i], m.Axis[i] = l.Joint.Origin, l.Joint.Axis.Unit()
			m.Lo[i], m.Hi[i], m.Damping[i] = l.Joint.Limits[0], l.Joint.Limits[1], l.Joint.Damping
		}
		m.moves[i] = make([]bool, n)
		for k := i; k > 0; k = m.Parent[k] {
			m.moves[i][k] = true
		}
	}
	for _, c := range b.Cylinders {
		cy := Cyl{
			Name: c.Name, BaseLink: b.LinkIndex(c.Base.Link), RodLink: b.LinkIndex(c.Rod.Link),
			Base: c.Base.Point, Rod: c.Rod.Point, Retracted: c.RetractedLength, P: c.Params,
		}
		for j := 1; j < n; j++ {
			if m.moves[cy.BaseLink][j] != m.moves[cy.RodLink][j] {
				cy.Span = append(cy.Span, j-1)
			}
		}
		m.Cyl = append(m.Cyl, cy)
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
func (m *Model) FK(q []float64, ps *Pose) {
	ps.R[0], ps.P[0] = geom.Identity(), geom.V3{}
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
	c := &m.Cyl[k]
	b := m.Point(ps, c.BaseLink, c.Base)
	r := m.Point(ps, c.RodLink, c.Rod)
	d := r.Sub(b)
	l := d.Norm()
	if dLdq != nil {
		u := d.Scale(1 / l)
		for i, q := range c.Span {
			j := q + 1
			var vel geom.V3
			if m.moves[c.RodLink][j] {
				vel = ps.Z[j].Cross(r.Sub(ps.P[j]))
			} else {
				vel = ps.Z[j].Cross(b.Sub(ps.P[j])).Scale(-1)
			}
			dLdq[i] = u.Dot(vel)
		}
	}
	return l
}

// RNEA computes the joint torques needed to produce accelerations qdd at
// velocities qd under gravity g (recursive Newton–Euler, world frame).
func (m *Model) RNEA(ps *Pose, qd, qdd []float64, g geom.V3, tau []float64) {
	n := len(m.Names)
	m.om[0], m.al[0], m.ao[0] = geom.V3{}, geom.V3{}, g.Scale(-1)
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
}

// MassMatrix fills M (NDof×NDof) for pose ps, one RNEA call per column.
// CRBA would be faster; this is simple and exact.
func (m *Model) MassMatrix(ps *Pose, M [][]float64) {
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
	n := len(b)
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
