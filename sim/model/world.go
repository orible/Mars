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

	// Self-collision between capsules: a spring-damper along the line
	// between their axes (scaled with the body like the floor) and sliding
	// friction, viscous up to selfMu times the normal force.
	selfK  = 3e4 // N/m
	selfC  = 200 // N·s/m
	selfCt = 100 // N·s/m
	selfMu = 0.5
)

// SelfHit is one capsule pair in contact: the point midway through their
// overlap and the force on the second capsule (the first gets −F).
type SelfHit struct {
	A, B int // capsule indices
	P, F geom.V3
}

// Joint limits are stiff mechanical stops (Model.LimitK), so a knee at its
// extension stop (a few degrees past straight, as in a human knee) locks:
// the weight line falls in front of the joint and pushes it into the
// stop, and standing needs no cylinder force at the knee.

// Ground is the floor: flat at y = 0 unless Bumps shape it. Each contact point is a compliant
// spring-damper normally and a stick–slip spring tangentially: while the
// tangential spring force stays inside the friction cone the point sticks
// to where it touched down, and beyond it the anchor slides so the force
// equals μ·N. These are environment parameters, so randomise them (above
// all Mu) when training for the real floor.
type Ground struct {
	K, C   float64 // N/m, N·s/m: normal stiffness and damping per point
	Kt, Ct float64 // N/m, N·s/m: tangential (stick) stiffness and damping per point
	Mu     float64 // Coulomb friction coefficient
	// Bumps shape the floor: its height is the sum of these waves (zero
	// for a flat floor). MuWaves make friction patchy: Mu is scaled by
	// 1 + the sum of these waves.
	Bumps, MuWaves []Wave
}

// Wave is A·cos(Kx·x + Kz·z + Phase).
type Wave struct{ A, Kx, Kz, Phase float64 }

func sumWaves(ws []Wave, x, z float64) float64 {
	s := 0.0
	for _, w := range ws {
		s += w.A * math.Cos(w.Kx*x+w.Kz*z+w.Phase)
	}
	return s
}

// Height is the floor's height under (x, z).
func (g Ground) Height(x, z float64) float64 { return sumWaves(g.Bumps, x, z) }

// MuAt is the friction coefficient at (x, z).
func (g Ground) MuAt(x, z float64) float64 {
	return math.Max(0.05, g.Mu*(1+sumWaves(g.MuWaves, x, z)))
}

// DefaultGround is a firm, grippy floor: about 1 mm of sink per foot
// corner under a 6 kg robot.
func DefaultGround() Ground { return Ground{K: 3e4, C: 200, Kt: 2e4, Ct: 100, Mu: 0.8} }

// stiffScale grows floor, joint-stop and end-stop stiffness with the body
// so a full-size robot sinks and overshoots about as little as the small
// one: 1 up to 10 kg, then proportional to mass.
func stiffScale(mass float64) float64 { return math.Max(1, mass/10) }

// GroundFor is DefaultGround stiffened for a body of the given mass.
func GroundFor(mass float64) Ground {
	g, s := DefaultGround(), stiffScale(mass)
	g.K, g.C, g.Kt, g.Ct = g.K*s, g.C*s, g.Kt*s, g.Ct*s
	return g
}

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
	// Gentle caps the help for getting up off the floor: at most half the
	// weight, and an uprighting moment of at most mass·g·0.15 m, so the
	// legs still have to do the rising.
	Gentle bool
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
	// Servo drives the valves from Targets every step (see servo.go);
	// otherwise Cmd is used as set. Neutral is each cylinder's extension
	// in the standing pose, where an action of 0 aims.
	Servo   bool
	Targets []float64
	Neutral []float64
	Len      []float64 // cylinder pin-to-pin lengths for the current pose
	DLdq     [][]float64
	Pose     *Pose
	ContactF []geom.V3 // last ground force on each contact point (world)
	// CapA and CapB are each capsule's ends in the world for the current
	// pose; ProbeF is the last ground force on each floor probe, and
	// SelfHits the capsule pairs touching each other on the last step.
	CapA, CapB []geom.V3
	ProbeF     []geom.V3
	SelfHits   []SelfHit
	T          float64
	// MassEvery is how many steps reuse one factored mass matrix. Inertia
	// changes slowly next to the 50 µs pressure and contact dynamics, and
	// rebuilding it is most of the cost of a step.
	MassEvery int

	standH  float64
	stiff   float64 // stiffScale for this body
	// TendonLen and TendonT are each passive tendon's length and its last
	// tension (N).
	TendonLen []float64
	TendonT   []float64
	tdLdq     [][]float64
	massAge int
	anchor  []geom.V3 // stick point of each touching contact
	touch   []bool
	pAnchor []geom.V3 // the same for the floor probes
	pTouch  []bool
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
		M: m, Floating: floating, Ground: GroundFor(m.TotalMass()), MassEvery: 5, stiff: stiffScale(m.TotalMass()),
		Base: Base{R: geom.Identity()},
		Q:    make([]float64, n), Qd: make([]float64, n),
		Cyl: make([]hydraulics.State, len(m.Cyl)), Cmd: make([]float64, len(m.Cyl)),
		Len: make([]float64, len(m.Cyl)), DLdq: make([][]float64, len(m.Cyl)),
		Pose:     m.NewPose(),
		ContactF: make([]geom.V3, len(m.Contacts)),
		anchor:   make([]geom.V3, len(m.Contacts)), touch: make([]bool, len(m.Contacts)),
		CapA: make([]geom.V3, len(m.Capsules)), CapB: make([]geom.V3, len(m.Capsules)),
		ProbeF:  make([]geom.V3, len(m.Probes)),
		pAnchor: make([]geom.V3, len(m.Probes)), pTouch: make([]bool, len(m.Probes)),
		nb:  nb,
		gen: make([]float64, nb+n), bias: make([]float64, nb+n), acc: make([]float64, nb+n),
		nu: make([]float64, nb+n), nud: make([]float64, nb+n),
		mm: make([][]float64, nb+n),
	}
	for i := range w.mm {
		w.mm[i] = make([]float64, nb+n)
	}
	w.TendonLen, w.TendonT = make([]float64, len(m.Tendons)), make([]float64, len(m.Tendons))
	for t := range m.Tendons {
		w.tdLdq = append(w.tdLdq, make([]float64, len(m.Tendons[t].Span)))
	}
	for k, c := range m.Cyl {
		w.DLdq[k] = make([]float64, len(c.Span))
	}
	w.standH = w.StandHeight()
	if floating {
		w.Base.P = geom.V3{0, w.standH, 0}
	}
	w.refresh()
	w.Targets, w.Neutral = make([]float64, len(m.Cyl)), make([]float64, len(m.Cyl))
	for k, c := range m.Cyl {
		w.Cyl[k] = hydraulics.NewState(c.P, m.Supply)
		w.Cyl[k].X = w.Len[k] - c.Retracted
		w.Neutral[k] = math.Max(0, math.Min(c.P.Stroke, w.Cyl[k].X))
	}
	w.HoldPose()
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
	// Rest the lowest point just above the floor under it.
	lo := math.Inf(1)
	for _, c := range m.Contacts {
		p := m.Point(w.Pose, c.Link, c.Point)
		lo = math.Min(lo, p[1]-w.Ground.Height(p[0]+x-w.Pose.P[0][0], p[2]+z-w.Pose.P[0][2]))
	}
	for i := range m.Names {
		p := w.Pose.P[i]
		lo = math.Min(lo, p[1]-w.Ground.Height(p[0]+x-w.Pose.P[0][0], p[2]+z-w.Pose.P[0][2]))
	}
	for _, pr := range m.Probes {
		p := m.Point(w.Pose, m.Capsules[pr.Capsule].Link, pr.C)
		lo = math.Min(lo, p[1]-m.Capsules[pr.Capsule].R-w.Ground.Height(p[0]+x-w.Pose.P[0][0], p[2]+z-w.Pose.P[0][2]))
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
	for i := range w.pTouch {
		w.pTouch[i], w.ProbeF[i] = false, geom.V3{}
	}
	w.SelfHits = w.SelfHits[:0]
	w.Harness = Harness{}
	w.HoldPose() // the servo holds the new pose until told otherwise
}

// StandHeight is the pelvis height at which the lowest contact point or
// collider touches the ground in the neutral pose (1 m if there are
// neither).
func (w *World) StandHeight() float64 {
	m := w.M
	if len(m.Contacts) == 0 && len(m.Probes) == 0 {
		return 1
	}
	ps := m.NewPose()
	m.FK(make([]float64, m.NDof()), ps)
	lo := math.Inf(1)
	for _, c := range m.Contacts {
		lo = math.Min(lo, m.Point(ps, c.Link, c.Point)[1])
	}
	for _, pr := range m.Probes {
		c := m.Capsules[pr.Capsule]
		lo = math.Min(lo, m.Point(ps, c.Link, pr.C)[1]-c.R)
	}
	return -lo
}

// Refresh recomputes the pose and cylinder lengths after Q or Base were
// set directly.
func (w *World) Refresh() { w.refresh() }

func (w *World) refresh() {
	w.M.FKBase(w.Base.R, w.Base.P, w.Q, w.Pose)
	for k := range w.M.Cyl {
		w.Len[k] = w.M.CylLength(w.Pose, k, w.DLdq[k])
	}
	for t := range w.M.Tendons {
		w.TendonLen[t] = w.M.TendonLength(w.Pose, t, w.tdLdq[t])
	}
	for k, c := range w.M.Capsules {
		w.CapA[k], w.CapB[k] = w.M.Point(w.Pose, c.Link, c.A), w.M.Point(w.Pose, c.Link, c.B)
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

// contacts adds ground forces on the contact points and the collider
// probes to w.gen and records them in ContactF and ProbeF.
func (w *World) contacts() {
	m := w.M
	for i, c := range m.Contacts {
		w.ContactF[i] = w.floor(c.Link, m.Point(w.Pose, c.Link, c.Point), &w.touch[i], &w.anchor[i])
	}
	for i, pr := range m.Probes {
		c := m.Capsules[pr.Capsule]
		p := m.Point(w.Pose, c.Link, pr.C)
		p[1] -= c.R // the sphere's lowest point
		w.ProbeF[i] = w.floor(c.Link, p, &w.pTouch[i], &w.pAnchor[i])
	}
}

// floor applies the ground's force at world point p on link l, a compliant
// spring-damper normally and a stick–slip spring tangentially (see
// Ground), and returns it. touch and anchor are the point's stick state.
func (w *World) floor(l int, p geom.V3, touch *bool, anchor *geom.V3) geom.V3 {
	m, g := w.M, w.Ground
	h := g.Height(p[0], p[2])
	if p[1] >= h {
		*touch = false
		return geom.V3{}
	}
	v := m.PointVelocity(w.Pose, w.Base.V, w.Base.W, w.Qd, l, p)
	fn := math.Max(0, -g.K*(p[1]-h)-g.C*v[1])
	if !*touch {
		*touch, *anchor = true, geom.V3{p[0], h, p[2]}
	}
	ft := geom.V3{
		-g.Kt*(p[0]-anchor[0]) - g.Ct*v[0], 0,
		-g.Kt*(p[2]-anchor[2]) - g.Ct*v[2],
	}
	if mag, lim := ft.Norm(), g.MuAt(p[0], p[2])*fn; mag > lim {
		// Slipping: limit to the cone and drag the anchor along.
		ft = ft.Scale(lim / mag)
		*anchor = geom.V3{p[0] + ft[0]/g.Kt, h, p[2] + ft[2]/g.Kt}
	}
	f := geom.V3{ft[0], fn, ft[2]}
	m.AddPointForce(w.Pose, l, p, f, w.Floating, w.gen)
	return f
}

// selfCollide pushes apart every colliding capsule pair that overlaps and
// records them in SelfHits.
func (w *World) selfCollide() {
	m := w.M
	w.SelfHits = w.SelfHits[:0]
	for _, pr := range m.Pairs {
		i, j := pr[0], pr[1]
		ci, cj := &m.Capsules[i], &m.Capsules[j]
		// Cheap reject: the spheres round each capsule's midpoint that hold it.
		mi, mj := w.CapA[i].Add(w.CapB[i]).Scale(0.5), w.CapA[j].Add(w.CapB[j]).Scale(0.5)
		reach := 0.5*(w.CapB[i].Sub(w.CapA[i]).Norm()+w.CapB[j].Sub(w.CapA[j]).Norm()) + ci.R + cj.R
		if d := mj.Sub(mi); d.Dot(d) > reach*reach {
			continue
		}
		p, q := geom.SegmentClosest(w.CapA[i], w.CapB[i], w.CapA[j], w.CapB[j])
		d := q.Sub(p)
		dist := d.Norm()
		over := ci.R + cj.R - dist
		if over <= 0 {
			continue
		}
		n := geom.V3{0, 1, 0} // axes crossing exactly: any direction will do
		if dist > 1e-9 {
			n = d.Scale(1 / dist)
		}
		x := p.Add(n.Scale(ci.R - over/2))
		rel := m.PointVelocity(w.Pose, w.Base.V, w.Base.W, w.Qd, cj.Link, x).Sub(m.PointVelocity(w.Pose, w.Base.V, w.Base.W, w.Qd, ci.Link, x))
		vn := rel.Dot(n)
		fn := math.Max(0, w.stiff*(selfK*over-selfC*vn))
		ft := rel.Sub(n.Scale(vn)).Scale(-w.stiff * selfCt)
		if mag, lim := ft.Norm(), selfMu*fn; mag > lim {
			ft = ft.Scale(lim / mag)
		}
		f := n.Scale(fn).Add(ft)
		m.AddPointForce(w.Pose, cj.Link, x, f, w.Floating, w.gen)
		m.AddPointForce(w.Pose, ci.Link, x, f.Scale(-1), w.Floating, w.gen)
		w.SelfHits = append(w.SelfHits, SelfHit{A: i, B: j, P: x, F: f})
	}
}

// Step advances the world by dt (≈50 µs; pressure dynamics are stiff).
func (w *World) Step(dt float64) {
	m, nb := w.M, w.nb
	for i := range w.gen {
		w.gen[i] = 0
	}
	if w.Servo {
		w.servoCommands()
	}
	for k := range m.Cyl {
		c := &m.Cyl[k]
		v := w.LengthRate(k)
		x := w.Len[k] - c.Retracted
		hydraulics.StepPressures(c.P, m.Supply, &w.Cyl[k], w.Cmd[k], math.Max(0, math.Min(c.P.Stroke, x)), v, dt)
		f := hydraulics.Force(c.P, w.Cyl[k]) - c.P.Viscous*v
		if x < 0 {
			f += w.stiff * (endStopK*(-x) - endStopC*v)
		} else if x > c.P.Stroke {
			f -= w.stiff * (endStopK*(x-c.P.Stroke) + endStopC*v)
		}
		for i, q := range c.Span {
			w.gen[nb+q] += f * w.DLdq[k][i]
		}
	}
	for t := range m.Tendons {
		tn := &m.Tendons[t]
		w.TendonT[t] = 0
		if stretch := w.TendonLen[t] - tn.Rest; stretch > 0 {
			rate := 0.0
			for i, q := range tn.Span {
				rate += w.tdLdq[t][i] * w.Qd[q]
			}
			tension := math.Max(0, tn.K*stretch+tn.C*rate)
			w.TendonT[t] = tension
			for i, q := range tn.Span {
				w.gen[nb+q] -= tension * w.tdLdq[t][i]
			}
		}
	}
	for q := range w.Q {
		l := q + 1
		w.gen[nb+q] -= m.Damping[l]*w.Qd[q] + m.Stiffness[l]*w.Q[q]
		if w.Q[q] < m.Lo[l] {
			w.gen[nb+q] += m.LimitK[l]*(m.Lo[l]-w.Q[q]) - m.LimitC[l]*w.Qd[q]
		} else if w.Q[q] > m.Hi[l] {
			w.gen[nb+q] += m.LimitK[l]*(m.Hi[l]-w.Q[q]) - m.LimitC[l]*w.Qd[q]
		}
	}
	refactor := w.massAge <= 0
	if refactor {
		w.massAge = max(1, w.MassEvery)
	}
	w.massAge--
	if len(m.Pairs) > 0 {
		w.selfCollide()
	}
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
		// Joint damping is integrated implicitly ((M + dt·C)·q̈ = τ − C·q̇),
		// so a light toe or arch hinge can carry real tissue damping
		// without going unstable.
		for q := range w.Q {
			w.mm[nb+q][nb+q] += dt * m.Damping[q+1]
		}
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
		// Hands lift, never push down.
		math.Max(0, s*(0.5*mass*9.81+k*(h.Height-p[1])-c*v[1])),
		s * (k*(h.Target[2]-p[2]) + c*(h.Vel[2]-v[2])),
	}
	up := w.Up()
	tilt := up.Cross(geom.V3{0, 1, 0}) // axis that turns the pelvis back upright, length sin(tilt)
	wh := geom.V3{w.Base.W[0], 0, w.Base.W[2]}
	tq := tilt.Scale(s * 40 * mass).Sub(wh.Scale(s * 4 * mass))
	if h.Gentle { // helping it up off the floor: help, do not crane it up
		f[1] = math.Min(f[1], s*0.5*mass*9.81)
		if lim := s * mass * 9.81 * 0.15; tq.Norm() > lim {
			tq = tq.Scale(lim / tq.Norm())
		}
	}
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
