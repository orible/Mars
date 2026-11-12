package optic

import (
	"math"

	"mars/sim/geom"
	"mars/sim/model"
)

// ScriptedWalk poses w kinematically on a scripted walk for demos and
// benchmarks: forward at 0.6 m/s turning slowly left, with step bob,
// sway and trunk roll. look (0..1) bends the spine forward and reaches
// the arms out, so the eyes see the feet, knees and hands.
func ScriptedWalk(w *model.World, t, look float64) {
	step := 2 * math.Pi * 1.8 * t
	yaw := 0.12 * t
	lean := 0.14 + 0.02*math.Sin(2*step) + 0.2*look
	roll := 0.03 * math.Sin(step)
	r := geom.AxisAngle(geom.V3{0, 1, 0}, yaw).
		Mul(geom.AxisAngle(geom.V3{0, 0, 1}, -lean)).
		Mul(geom.AxisAngle(geom.V3{1, 0, 0}, roll))
	fwd := geom.V3{math.Cos(yaw), 0, -math.Sin(yaw)}
	side := geom.V3{math.Sin(yaw), 0, math.Cos(yaw)}
	p := fwd.Scale(0.6 * t).Add(side.Scale(0.03 * math.Sin(step)))
	p[1] = w.StandH() - 0.02 + 0.02*math.Cos(2*step)
	m := w.M
	set := func(name string, q float64) {
		for i, n := range m.Names {
			if n == name && i > 0 {
				w.Q[i-1] = math.Max(m.Lo[i], math.Min(m.Hi[i], q))
			}
		}
	}
	for i := range w.Q {
		w.Q[i] = 0
	}
	set("lumbar", -0.7*look)
	set("chest", -0.3*look)
	set("hip_roll_l", -0.05)
	set("hip_roll_r", 0.05)
	sw := 0.35 * math.Sin(step)
	set("thigh_l", 0.2+0.3*look+sw)
	set("thigh_r", 0.2+0.3*look-sw)
	set("shin_l", -0.3-0.3*math.Max(0, -math.Sin(step)))
	set("shin_r", -0.3-0.3*math.Max(0, math.Sin(step)))
	set("shoulder_l", 1.1*look-0.2*sw)
	set("shoulder_r", 1.1*look+0.2*sw)
	set("forearm_l", 0.5)
	set("forearm_r", 0.5)
	w.Base.R, w.Base.P = r, p
	w.Refresh()
}
