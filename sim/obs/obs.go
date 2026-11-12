// Package obs turns the simulated world into the network's inputs: one
// token per cylinder, one token per body part, and one global vector.
//
// Cylinder tokens carry the cylinder's state, its strength against the
// body's weight, where its pins are, and which way and how hard it twists
// the joints it spans. Part tokens are the network's sense of its own
// shape: each part's length and extent, its share of the mass, where it
// is and how it is oriented and moving, and how much of the weight it
// carries on the floor. Sensor tokens are what the feet feel: each sole
// pad's place on its foot and in the body, its pressure, how close it is
// to slipping (shear over pressure), the shear's direction and how fast
// it is sliding; and each tendon's tension.
//
// Positions are expressed in the heading frame: centred on the centre of
// gravity (or the pelvis), with gravity's up axis and the pelvis's forward
// direction projected onto the floor. Lengths are divided by the body's
// standing pelvis height, speeds by √(g·height), forces by its weight, so
// a 1.6 m and a 1.9 m body see alike features when they move alike.
//
// Mirror maps an observation to the left–right mirror image of the same
// situation; training uses it to keep the policy symmetric.
package obs

import (
	"math"
	"sort"

	"mars/sim/command"
	"mars/sim/geom"
	"mars/sim/model"
	"mars/sim/policy"
)

// Version is the feature layout. Checkpoints record it, and one trained
// on another layout is refused.
const Version = 3

// Feature sizes.
const (
	TokenDim  = 20
	PartDim   = 19
	SensorDim = 11
	GlobalDim = 19
	// MaxArms is how many of a cylinder's joints (largest lever first)
	// go into its token.
	MaxArms = 2
	// PartMinMass is the share of the body's mass below which a link (a
	// joint bracket) is not a part of its own, unless it touches the floor
	// (a heel, the toes).
	PartMinMass = 0.01
)

const g0 = 9.81

// IMU is the fused pelvis IMU reading.
type IMU struct {
	Pitch float64 // rad, 0 = upright, positive = nose up
	Roll  float64 // rad, positive = right side down
	Gyro  geom.V3 // rad/s, pelvis frame
}

// ReadIMU is what an ideal IMU on the pelvis would report.
func ReadIMU(w *model.World) IMU {
	r := w.Base.R
	fwdY, rightY := r[1][0], r[1][2]
	return IMU{
		Pitch: math.Asin(math.Max(-1, math.Min(1, fwdY))),
		Roll:  math.Asin(math.Max(-1, math.Min(1, -rightY))),
		Gyro:  r.T().MulV(w.Base.W),
	}
}

// heading returns the map from world vectors to the heading frame.
func heading(w *model.World) geom.M3 {
	r := w.Base.R
	yaw := math.Atan2(-r[2][0], r[0][0]) // pelvis forward axis projected onto the floor
	return geom.AxisAngle(geom.V3{0, 1, 0}, yaw).T()
}

// Build returns the observation (without privileged features). cmd is
// the current command from the LLM loop; it is the network's only
// knowledge of where to go.
func Build(w *model.World, imu IMU, cmd command.Command) *policy.Obs {
	m, ps := w.M, w.Pose
	h := heading(w)
	cogW, mass := w.CoG()
	L := w.StandH()
	if L <= 0 {
		L = 1
	}
	vs := math.Sqrt(g0 * L)
	weight := mass * g0
	ps0 := m.Supply.Pressure
	rel := func(p geom.V3) geom.V3 { return h.MulV(p.Sub(cogW)).Scale(1 / L) }

	o := &policy.Obs{Tokens: make([][]float64, len(m.Cyl))}
	type arm struct {
		a float64
		z geom.V3
	}
	arms := make([]arm, 0, 4)
	for k, c := range m.Cyl {
		st := w.Cyl[k]
		arms = arms[:0]
		for i, q := range c.Span {
			arms = append(arms, arm{w.DLdq[k][i], ps.Z[q+1]})
		}
		sort.Slice(arms, func(i, j int) bool { return math.Abs(arms[i].a) > math.Abs(arms[j].a) })
		var tq [MaxArms]geom.V3 // lever × joint axis: the twist one newton gives, heading frame
		for i := 0; i < MaxArms && i < len(arms); i++ {
			tq[i] = h.MulV(arms[i].z).Scale(arms[i].a / (0.1 * L))
		}
		b := rel(m.Point(ps, c.BaseLink, c.Base))
		r := rel(m.Point(ps, c.RodLink, c.Rod))
		force := 0.0
		if c.P.PressureControl {
			force = 1
		}
		o.Tokens[k] = []float64{
			st.X/c.P.Stroke*2 - 1,
			st.V / (0.5 * L),
			st.PA / ps0,
			st.PB / ps0,
			st.Spool,
			ps0 * c.P.BoreArea / weight / 10,
			c.P.RodArea / c.P.BoreArea,
			b[0], b[1], b[2],
			r[0], r[1], r[2],
			tq[0][0], tq[0][1], tq[0][2],
			tq[1][0], tq[1][1], tq[1][2],
			force,
		}
	}

	// Body parts. Static shape (length, extent) comes from the link's own
	// frame: its child joints and contact points.
	n := len(m.Names)
	lo := make([]geom.V3, n)
	hi := make([]geom.V3, n)
	reach := make([]float64, n)
	for i := 0; i < n; i++ {
		lo[i], hi[i] = m.Com[i], m.Com[i]
		lo[i], hi[i] = minV(lo[i], geom.V3{}), maxV(hi[i], geom.V3{})
		reach[i] = m.Com[i].Norm()
	}
	extend := func(i int, p geom.V3) {
		lo[i], hi[i] = minV(lo[i], p), maxV(hi[i], p)
		reach[i] = math.Max(reach[i], p.Norm())
	}
	for i := 1; i < n; i++ {
		extend(m.Parent[i], m.Origin[i])
	}
	load := make([]float64, n)
	touches := make([]bool, n)
	low := make([]float64, n)
	for i := range low {
		low[i] = math.Inf(1)
	}
	floor := math.Inf(1)
	totalLoad := 0.0
	for c, ct := range m.Contacts {
		extend(ct.Link, ct.Point)
		touches[ct.Link] = true
		y := m.Point(ps, ct.Link, ct.Point)[1]
		low[ct.Link] = math.Min(low[ct.Link], y)
		floor = math.Min(floor, y)
		load[ct.Link] += w.ContactF[c][1]
		totalLoad += w.ContactF[c][1]
	}
	var cogV geom.V3
	for i := 0; i < n; i++ {
		com := m.Point(ps, i, m.Com[i])
		v := m.PointVelocity(ps, w.Base.V, w.Base.W, w.Qd, i, com)
		cogV = cogV.Add(v.Scale(m.Mass[i] / mass))
		if m.Mass[i] < PartMinMass*mass && !touches[i] {
			continue
		}
		if math.IsInf(low[i], 1) {
			low[i] = com[1]
		}
		p := rel(com)
		ay := h.MulV(ps.R[i].MulV(geom.V3{0, 1, 0}))
		ax := h.MulV(ps.R[i].MulV(geom.V3{1, 0, 0}))
		vh := h.MulV(v).Scale(1 / vs)
		ext := hi[i].Sub(lo[i]).Scale(1 / L)
		o.Parts = append(o.Parts, []float64{
			p[0], p[1], p[2],
			ay[0], ay[1], ay[2],
			ax[0], ax[1], ax[2],
			vh[0], vh[1], vh[2],
			5 * m.Mass[i] / mass,
			reach[i] / L,
			ext[0], ext[1], ext[2],
			math.Min(1, load[i]/weight),
			(low[i] - floor) / L,
		})
	}

	// Sensors: sole pads, then tendons.
	for _, sn := range m.Sensors {
		ct := m.Contacts[sn.Contact]
		p := m.Point(ps, ct.Link, ct.Point)
		f := w.ContactF[sn.Contact]
		fn := math.Max(0, f[1])
		sh := h.MulV(geom.V3{f[0], 0, f[2]}).Scale(10 / weight)
		ratio, slip := 0.0, 0.0
		if fn > 0 {
			ratio = math.Min(2, math.Hypot(f[0], f[2])/fn)
			v := m.PointVelocity(ps, w.Base.V, w.Base.W, w.Qd, ct.Link, p)
			slip = math.Min(2, math.Hypot(v[0], v[2])/vs)
		}
		r := rel(p)
		o.Sensors = append(o.Sensors, []float64{
			r[0], r[1], r[2],
			sn.Local[0] / L, sn.Local[2] / L,
			10 * fn / weight,
			ratio,
			sh[0], sh[2],
			slip,
			0,
		})
	}
	for t := range m.Tendons {
		pts := m.RoutePoints(ps, t, true)[0]
		mid := pts[0].Add(pts[len(pts)-1]).Scale(0.5)
		r := rel(mid)
		o.Sensors = append(o.Sensors, []float64{
			r[0], r[1], r[2],
			0, 0,
			10 * w.TendonT[t] / weight,
			0, 0, 0, 0,
			1,
		})
	}

	cog := h.MulV(cogW.Sub(w.Base.P)).Scale(1 / L)
	cv := h.MulV(cogV).Scale(1 / vs)
	o.Global = []float64{
		math.Log(mass / 75),
		cog[0], cog[1], cog[2],
		imu.Pitch, imu.Roll,
		imu.Gyro[0] / 5, imu.Gyro[1] / 5, imu.Gyro[2] / 5,
		math.Cos(cmd.HeadingDeg * math.Pi / 180), math.Sin(cmd.HeadingDeg * math.Pi / 180), cmd.Speed / command.MaxSpeed,
		float64(len(m.Cyl)) / 20,
		float64(len(o.Parts)) / 20,
		L - 1,
		math.Min(2, totalLoad/weight),
		cv[0], cv[1], cv[2],
	}
	return o
}

func minV(a, b geom.V3) geom.V3 {
	return geom.V3{math.Min(a[0], b[0]), math.Min(a[1], b[1]), math.Min(a[2], b[2])}
}

func maxV(a, b geom.V3) geom.V3 {
	return geom.V3{math.Max(a[0], b[0]), math.Max(a[1], b[1]), math.Max(a[2], b[2])}
}

// Features that change sign in the mirror image (left ↔ right, the
// heading frame's z axis). Positions and directions flip z; twists (axial
// vectors) flip x and y instead.
var (
	tokenFlip  = []int{9, 12, 13, 14, 16, 17}
	partFlip   = []int{2, 5, 8, 11}
	sensorFlip = []int{2, 4, 8}
	globalFlip = []int{3, 5, 6, 7, 10, 18}
)

// Mirror returns the observation of the left–right mirror image of the
// same situation. Each actuator token stays in place (it becomes its
// mirror twin), so a symmetric policy gives the same command per token.
func Mirror(o *policy.Obs) *policy.Obs {
	flip := func(x []float64, idx []int) []float64 {
		y := append([]float64(nil), x...)
		for _, i := range idx {
			y[i] = -y[i]
		}
		return y
	}
	out := &policy.Obs{Tokens: make([][]float64, len(o.Tokens)), Parts: make([][]float64, len(o.Parts)),
		Sensors: make([][]float64, len(o.Sensors)), Global: flip(o.Global, globalFlip)}
	for i, s := range o.Sensors {
		out.Sensors[i] = flip(s, sensorFlip)
	}
	for i, t := range o.Tokens {
		out.Tokens[i] = flip(t, tokenFlip)
	}
	for i, p := range o.Parts {
		out.Parts[i] = flip(p, partFlip)
	}
	return out
}
