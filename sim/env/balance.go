package env

import (
	"math"

	"mars/sim/geom"
)

// Standing balance: pushes are caught by stepping, not only by the ankles
// and hips. Three pieces work together:
//
//   - holdSpot replaces "do not move" with "stay near your spot": the spot
//     (anchor) follows the pelvis slowly, so a quick recovery step costs
//     little while drifting away still does.
//   - The capture point (where the centre of gravity would come to rest:
//     its position plus its velocity × √(height/g)) is compared with the
//     middle of the loaded feet. Bringing it back over the feet pays, as
//     potential-based shaping, so a step that lands under it pays and
//     cannot be farmed.
//   - A step it did not need (capture point already over the feet) costs a
//     little; see stepping.
//
// impacts penalises hard landings the way an accelerometer feels them: the
// peak acceleration at the pelvis IMU and the head over each policy step.

const (
	anchorTau   = 3.0  // s: how fast the standing spot follows the pelvis
	holdRadius  = 0.15 // m: pelvis this far from its spot halves the hold term (e⁻¹)
	cpWeight    = 2.0  // reward per metre the capture point moves back over the feet
	cpCap       = 0.4  // m: farther than this counts the same (it is falling)
	shuffleCP   = 0.08 // m: capture point within this of the feet needs no step
	shockFree   = 2.0  // g: peaks below this (walking, a firm step) cost nothing
	shockWeight = 0.05 // reward per (g above shockFree)²
	shockCap    = 10.0 // g: one step counts at most this much
	accelTau    = 0.005 // s: accelerometer low-pass (about 30 Hz)
)

// resetBalance starts the standing terms for a new episode.
func (e *Env) resetBalance() {
	e.anchor = e.W.Base.P
	e.prevCoG, _ = e.W.CoG()
	e.support = e.prevCoG
	e.accelOK = false
	e.shock = 0
	e.updateSupport()
	e.cpDist = math.Min(cpCap, horiz(e.prevCoG.Sub(e.support)))
	e.prevCP = e.cpDist
}

// updateSupport moves the support point to the centre of pressure of the
// feet (contacts weighted by their load); with no foot down it keeps the
// last one.
func (e *Env) updateSupport() {
	w := e.W
	var s geom.V3
	total := 0.0
	for _, f := range e.feet {
		for _, c := range f.contacts {
			if fn := w.ContactF[c][1]; fn > 0 {
				s = s.Add(w.M.Point(w.Pose, w.M.Contacts[c].Link, w.M.Contacts[c].Point).Scale(fn))
				total += fn
			}
		}
	}
	if total > 0 {
		e.support = s.Scale(1 / total)
	}
}

// updateCapture computes this step's capture point distance. Call once per
// policy step, after the physics.
func (e *Env) updateCapture() {
	cog, _ := e.W.CoG()
	v := cog.Sub(e.prevCoG).Scale(1 / PolicyDt)
	e.prevCoG = cog
	e.updateSupport()
	h := math.Max(0.1, cog[1]-e.support[1])
	t := math.Sqrt(h / 9.81)
	cp := geom.V3{cog[0] + v[0]*t, 0, cog[2] + v[2]*t}
	e.cpDist = math.Min(cpCap, horiz(cp.Sub(e.support)))
}

// holdSpot returns the standing hold term (1 on the spot) and the capture
// point shaping for this step.
func (e *Env) holdSpot() (hold, recover float64) {
	p := e.W.Base.P
	k := math.Min(1, PolicyDt/anchorTau)
	e.anchor = e.anchor.Add(p.Sub(e.anchor).Scale(k))
	d := horiz(p.Sub(e.anchor))
	hold = math.Exp(-d * d / (holdRadius * holdRadius))
	recover = cpWeight * (e.prevCP - e.cpDist)
	e.prevCP = e.cpDist
	return hold, recover
}

// sampleShock reads the accelerometers: pelvis and head acceleration over
// the last control tick, gravity included as a real one feels it (a − g),
// low-passed at about 30 Hz like a body-worn sensor, so the stiff contact
// springs' millisecond spikes do not count. The reading is how far the
// felt acceleration is from 1 g: 0 at rest, 1 in free fall. Keeps the
// peak in e.shock.
func (e *Env) sampleShock() {
	w := e.W
	head := w.M.Point(w.Pose, e.head, w.M.Com[e.head])
	v := [2]geom.V3{w.Base.V, w.M.PointVelocity(w.Pose, w.Base.V, w.Base.W, w.Qd, e.head, head)}
	if e.accelOK {
		k := ControlDt / (ControlDt + accelTau)
		for i := range v {
			a := v[i].Sub(e.accelV[i]).Scale(1 / ControlDt)
			a[1] += 9.81
			e.accelF[i] = e.accelF[i].Add(a.Sub(e.accelF[i]).Scale(k))
			g := e.accelF[i].Norm()/9.81 - 1
			e.shock = math.Max(e.shock, math.Abs(g))
		}
	} else {
		e.accelF = [2]geom.V3{{0, 9.81, 0}, {0, 9.81, 0}}
	}
	e.accelV, e.accelOK = v, true
}

// impacts is the penalty for the hardest jolt the accelerometers felt this
// step, above what firm walking gives.
func (e *Env) impacts() float64 {
	x := math.Min(shockCap, math.Max(0, e.shock-shockFree))
	return shockWeight * x * x
}

func horiz(v geom.V3) float64 { return math.Hypot(v[0], v[2]) }
