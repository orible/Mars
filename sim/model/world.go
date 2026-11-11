package model

import (
	"math"

	"mars/sim/geom"
	"mars/sim/hydraulics"
)

// Gravity in the world frame.
var Gravity = geom.V3{0, -9.81, 0}

const (
	endStopK = 2e5 // N/m, cylinder end stop stiffness
	endStopC = 500 // N·s/m
	// Joint limits are stiff mechanical stops, so a knee at its extension
	// stop (a few degrees past straight, as in a human knee) locks: the
	// weight line falls in front of the joint and pushes it into the stop,
	// and standing needs no cylinder force at the knee.
	limitK = 2000 // N·m/rad
	limitC = 10   // N·m·s/rad
)

// Ground is a flat floor at y = 0. Each contact point is a compliant
// spring-damper normally and a stick–slip spring tangentially: while the
// tangential spring force stays inside the friction cone the point sticks
// to where it touched down, and beyond it the anchor slides so the force
// equals μ·N. These are environment parameters, so randomise them (above
// all Mu) when training for the real floor.
type Ground struct {
	K, C   float64 // N/m, N·s/m: normal stiffness and damping per point
	Kt, Ct float64 // N/m, N·s/m: tangential (stick) stiffness and damping per point
	Mu     float64 // Coulomb friction coefficient
}

// DefaultGround is a firm, grippy floor: about 1 mm of sink per foot
// corner under a 6 kg robot.
func DefaultGround() Ground { return Ground{K: 3e4, C: 200, Kt: 2e4, Ct: 100, Mu: 0.8} }

// Base is the floating pelvis state.
type Base struct {
	R geom.M3 // orientation (pelvis frame to world)
	P geom.V3 // world position of the pelvis frame origin
	V geom.V3 // world velocity of the pelvis frame origin
	W geom.V3 // angular velocity, world frame
}

// Harness holds the pelvis the way an adult holds a baby learning to walk:
// it carries part of the weight, keeps the pelvis upright and carries it
// along at a set velocity, so the legs can learn to kick and step before
// they can balance. Training fades Strength to 0.
type Harness struct {
	Strength float64 // 0 = off, 1 = full support
	Vel      geom.V3 // world velocity to carry the pelvis at (horizontal)
	Height   float64 // pelvis height to hold
	Target   geom.V3 // where the harness wants the pelvis now (moves with Vel)
	Force    geom.V3 // last force applied (for display)
}

// World is one simulated robot: either on the test stand (pelvis bolted
// to the world) or floating on the ground with foot contacts. Every
// cylinder's pressures are coupled to the articulated dynamics.
type World struct {
	M        *Model
	Floating bool
	Base     Base
	Ground   Ground
	Harness  Harness
	Q        []float64
	Qd       []float64
	Cyl      []hydraulics.State
	Cmd      []float64 // valve commands, one per cylinder, in [-1, 1]
	Len      []float64 // cylinder pin-to-pin lengths for the current pose
	DLdq     [][]float64
	Pose     *Pose
	ContactF []geom.V3 // last ground force on each contact point (world)
	T        float64
	// MassEvery is how many steps reuse one factored mass matrix. Inertia
	// changes slowly next to the 50 µs pressure and contact dynamics, and
	// rebuilding it is most of the cost of a step.
	MassEvery int

	standH  float64
	massAge int
	anchor  []geom.V3 // stick point of each touching contact
	touch   []bool
	nb      int // 6 when floating, else 0
	gen     []float64
	bias    []float64
	acc     []float64
	nu, nud []float64
	mm      [][]float64
}

// NewWorld starts in the neutral pose with chambers at half supply. A
// floating world starts upright with its lowest contact point on the
// ground.
func NewWorld(m *Model, floating bool) *World {
	n := m.NDof()
	nb := 0
	if floating {
		nb = 6
	}
	w := &World{
		M: m, Floating: floating, Ground: DefaultGround(), MassEvery: 5,
		Base: Base{R: geom.Identity()},
		Q:    make([]float64, n), Qd: make([]float64, n),
		Cyl: make([]hydraulics.State, len(m.Cyl)), Cmd: make([]float64, len(m.Cyl)),
		Len: make([]float64, len(m.Cyl)), DLdq: make([][]float64, len(m.Cyl)),
		Pose:     m.NewPose(),
		ContactF: make([]geom.V3, len(m.Contacts)),
		anchor:   make([]geom.V3, len(m.Contacts)), touch: make([]bool, len(m.Contacts)),
		nb:  nb,
		gen: make([]float64, nb+n), bias: make([]float64, nb+n), acc: make([]float64, nb+n),
		nu: make([]float64, nb+n), nud: make([]float64, nb+n),
		mm: make([][]float64, nb+n),
	}
	for i := range w.mm {
		w.mm[i] = make([]float64, nb+n)
	}
	for k, c := range m.Cyl {
		w.DLdq[k] = make([]float64, len(c.Span))
	}
	w.standH = w.StandHeight()
	if floating {
		w.Base.P = geom.V3{0, w.standH, 0}
	}
	w.refresh()
	for k, c := range m.Cyl {
		w.Cyl[k] = hydraulics.NewState(c.P, m.Supply)
		w.Cyl[k].X = w.Len[k] - c.Retracted
	}
	return w
}

// Place puts a floating robot at rest with pelvis orientation r and joint
// angles q (clamped to their limits), lowered until its lowest contact
// point or joint is just above the floor, at horizontal position xz. Use
// it to start lying on the back, front or side.
func (w *World) Place(r geom.M3, q []float64, x, z float64) {
	m := w.M
	for i := range w.Q {
		w.Q[i] = math.Max(m.Lo[i+1], math.Min(m.Hi[i+1], q[i]))
		w.Qd[i] = 0
	}
	w.Base = Base{R: r}
	w.refresh()
	lo := math.Inf(1)
	for _, c := range m.Contacts {
		lo = math.Min(lo, m.Point(w.Pose, c.Link, c.Point)[1])
	}
	for i := range m.Names {
		lo = math.Min(lo, w.Pose.P[i][1])
	}
	w.Base.P = geom.V3{x, 0.005 - lo, z}
	w.refresh()
	for k, c := range m.Cyl {
		w.Cyl[k] = hydraulics.NewState(c.P, m.Supply)
		w.Cyl[k].X = w.Len[k] - c.Retracted
	}
	for i := range w.touch {
		w.touch[i], w.ContactF[i] = false, geom.V3{}
	}
	w.Harness = Harness{}
}

// StandHeight is the pelvis height at which the lowest contact point
// touches the ground in the neutral pose (1 m if there are no contacts).
func (w *World) StandHeight() float64 {
	if len(w.M.Contacts) == 0 {
		return 1
	}
	ps := w.M.NewPose()
	w.M.FK(make([]float64, w.M.NDof()), ps)
	lo := math.Inf(1)
	for _, c := range w.M.Contacts {
		lo = math.Min(lo, w.M.Point(ps, c.Link, c.Point)[1])
	}
	return -lo
}

func (w *World) refresh() {
	w.M.FKBase(w.Base.R, w.Base.P, w.Q, w.Pose)
	for k := range w.M.Cyl {
		w.Len[k] = w.M.CylLength(w.Pose, k, w.DLdq[k])
	}
}

// LengthRate returns cylinder k's extension speed in m/s.
func (w *World) LengthRate(k int) float64 {
	s := 0.0
	for i, q := range w.M.Cyl[k].Span {
		s += w.DLdq[k][i] * w.Qd[q]
	}
	return s
}

// contacts adds ground forces to w.gen and records them in ContactF.
func (w *World) contacts() {
	m, g := w.M, w.Ground
	for i, c := range m.Contacts {
		p := m.Point(w.Pose, c.Link, c.Point)
		if p[1] >= 0 {
			w.touch[i], w.ContactF[i] = false, geom.V3{}
			continue
		}
		v := m.PointVelocity(w.Pose, w.Base.V, w.Base.W, w.Qd, c.Link, p)
		fn := math.Max(0, -g.K*p[1]-g.C*v[1])
		if !w.touch[i] {
			w.touch[i], w.anchor[i] = true, geom.V3{p[0], 0, p[2]}
		}
		ft := geom.V3{
			-g.Kt*(p[0]-w.anchor[i][0]) - g.Ct*v[0], 0,
			-g.Kt*(p[2]-w.anchor[i][2]) - g.Ct*v[2],
		}
		if mag, lim := ft.Norm(), g.Mu*fn; mag > lim {
			// Slipping: limit to the cone and drag the anchor along.
			ft = ft.Scale(lim / mag)
			w.anchor[i] = geom.V3{p[0] + ft[0]/g.Kt, 0, p[2] + ft[2]/g.Kt}
		}
		f := geom.V3{ft[0], fn, ft[2]}
		w.ContactF[i] = f
		m.AddPointForce(w.Pose, c.Link, p, f, w.Floating, w.gen)
	}
}

// Step advances the world by dt (≈50 µs; pressure dynamics are stiff).
func (w *World) Step(dt float64) {
	m, nb := w.M, w.nb
	for i := range w.gen {
		w.gen[i] = 0
	}
	for k := range m.Cyl {
		c := &m.Cyl[k]
		v := w.LengthRate(k)
		x := w.Len[k] - c.Retracted
		hydraulics.StepPressures(c.P, m.Supply, &w.Cyl[k], w.Cmd[k], math.Max(0, math.Min(c.P.Stroke, x)), v, dt)
		f := hydraulics.Force(c.P, w.Cyl[k]) - c.P.Viscous*v
		if x < 0 {
			f += endStopK*(-x) - endStopC*v
		} else if x > c.P.Stroke {
			f -= endStopK*(x-c.P.Stroke) + endStopC*v
		}
		for i, q := range c.Span {
			w.gen[nb+q] += f * w.DLdq[k][i]
		}
	}
	for q := range w.Q {
		l := q + 1
		w.gen[nb+q] -= m.Damping[l] * w.Qd[q]
		if w.Q[q] < m.Lo[l] {
			w.gen[nb+q] += limitK*(m.Lo[l]-w.Q[q]) - limitC*w.Qd[q]
		} else if w.Q[q] > m.Hi[l] {
			w.gen[nb+q] += limitK*(m.Hi[l]-w.Q[q]) - limitC*w.Qd[q]
		}
	}
	refactor := w.massAge <= 0
	if refactor {
		w.massAge = max(1, w.MassEvery)
	}
	w.massAge--
	if w.Floating {
		w.contacts()
		w.harness(dt)
		copy(w.nu[0:3], w.Base.V[:])
		copy(w.nu[3:6], w.Base.W[:])
		copy(w.nu[6:], w.Qd)
		m.RNEAFloating(w.Pose, w.nu, w.nud, Gravity, w.bias)
		if refactor {
			m.MassMatrixFloating(w.Pose, w.mm)
		}
	} else {
		m.RNEA(w.Pose, w.Qd, m.zero, Gravity, w.bias)
		if refactor {
			m.MassMatrix(w.Pose, w.mm)
		}
	}
	if refactor {
		cholesky(w.mm)
	}
	for i := range w.acc {
		w.acc[i] = w.gen[i] - w.bias[i]
	}
	cholSolve(w.mm, w.acc)
	if w.Floating {
		b := &w.Base
		for i := 0; i < 3; i++ {
			b.V[i] += w.acc[i] * dt
			b.W[i] += w.acc[3+i] * dt
		}
		b.P = b.P.Add(b.V.Scale(dt))
		if wn := b.W.Norm(); wn > 0 {
			b.R = orthonormalize(geom.AxisAngle(b.W, wn*dt).Mul(b.R))
		}
	}
	for q := range w.Q {
		w.Qd[q] += w.acc[nb+q] * dt
		w.Q[q] += w.Qd[q] * dt
	}
	w.T += dt
	w.refresh()
}

// harness applies the helping hands to the pelvis (force at its origin and
// an uprighting moment).
func (w *World) harness(dt float64) {
	h := &w.Harness
	if h.Strength <= 0 {
		h.Force = geom.V3{}
		return
	}
	mass := 0.0
	for _, m := range w.M.Mass {
		mass += m
	}
	p, v := w.Base.P, w.Base.V
	// The target moves on, but never runs more than 0.3 m ahead.
	h.Target = h.Target.Add(h.Vel.Scale(dt))
	for _, i := range []int{0, 2} {
		h.Target[i] = p[i] + math.Max(-0.3, math.Min(0.3, h.Target[i]-p[i]))
	}
	k, c := 100*mass, 20*mass // ≈10 rad/s, critically damped
	s := h.Strength
	f := geom.V3{
		s * (k*(h.Target[0]-p[0]) + c*(h.Vel[0]-v[0])),
		// Hands hold up to half the weight and can only lift, never push down.
		math.Max(0, s*(0.5*mass*9.81+k*(h.Height-p[1])-c*v[1])),
		s * (k*(h.Target[2]-p[2]) + c*(h.Vel[2]-v[2])),
	}
	up := w.Up()
	tilt := up.Cross(geom.V3{0, 1, 0}) // axis that turns the pelvis back upright, length sin(tilt)
	wh := geom.V3{w.Base.W[0], 0, w.Base.W[2]}
	tq := tilt.Scale(s * 40 * mass).Sub(wh.Scale(s * 4 * mass))
	h.Force = f
	for i := 0; i < 3; i++ {
		w.gen[i] += f[i]
		w.gen[3+i] += tq[i]
	}
}

// orthonormalize removes rounding drift from a rotation (Gram–Schmidt on
// its columns).
func orthonormalize(r geom.M3) geom.M3 {
	x := geom.V3{r[0][0], r[1][0], r[2][0]}.Unit()
	y := geom.V3{r[0][1], r[1][1], r[2][1]}
	y = y.Sub(x.Scale(x.Dot(y))).Unit()
	z := x.Cross(y)
	return geom.M3{{x[0], y[0], z[0]}, {x[1], y[1], z[1]}, {x[2], y[2], z[2]}}
}

// Healthy reports whether the state is finite.
func (w *World) Healthy() bool {
	bad := func(v float64) bool { return math.IsNaN(v) || math.IsInf(v, 0) }
	for q := range w.Q {
		if bad(w.Q[q]) || bad(w.Qd[q]) {
			return false
		}
	}
	for i := 0; i < 3; i++ {
		if bad(w.Base.P[i]) || bad(w.Base.V[i]) || bad(w.Base.W[i]) {
			return false
		}
	}
	return true
}

// Up is the pelvis up axis in the world; Up()[1] is the cosine of its tilt.
func (w *World) Up() geom.V3 { return geom.V3{w.Base.R[0][1], w.Base.R[1][1], w.Base.R[2][1]} }

// Fallen reports a floating robot that is tilted past 60° or whose pelvis
// has dropped below half its standing height.
func (w *World) Fallen() bool {
	return w.Floating && (w.Up()[1] < 0.5 || w.Base.P[1] < 0.5*w.standH)
}

// Upright reports a robot that is standing: pelvis within 30° of
// vertical and at least 86% of standing height (a 90° squat, about 83%,
// is not up; walking stays around 95%).
func (w *World) Upright() bool {
	return !w.Floating || (w.Up()[1] > 0.85 && w.Base.P[1] > 0.86*w.standH)
}

// StandH is the neutral standing pelvis height.
func (w *World) StandH() float64 { return w.standH }

// CoG returns the whole-body centre of gravity (world) and total mass.
func (w *World) CoG() (geom.V3, float64) { return w.M.CoG(w.Pose) }
