package env

import (
	"math"
	"math/rand"
	"strings"

	"mars/sim/geom"
	"mars/sim/model"
)

// Getting up is learned backwards, the way babies come to it: first from a
// crouch, then from kneeling, then from sitting, and last from lying flat.
// A start level in [0, 1] sets the hardest pose an episode may start in;
// each episode draws a difficulty up to that level:
//
//	0.00–0.25 deep squat, feet on the floor
//	0.25–0.50 kneeling, shins flat on the floor
//	0.50–0.75 sitting, legs out in front
//	0.75–1.00 lying on the back, front or a side
//
// The trainer raises the level as get-ups succeed (see cmd/train).

// FloorStart poses a floating robot for a get-up episode at difficulty up
// to level, at its current horizontal position, facing a random way.
func FloorStart(w *model.World, rng *rand.Rand, level float64) {
	floorPose(w, rng, rng.Float64()*math.Max(0, math.Min(1, level)))
}

// floorPose poses the robot at difficulty d (see the ladder above).
func floorPose(w *model.World, rng *rand.Rand, d float64) {
	if d >= 0.75 {
		LieDown(w, rng)
		return
	}
	m := w.M
	q := make([]float64, m.NDof())
	// Pose by joint role, found from link names; bodies without these
	// names fall back to lying down.
	set := func(prefix string, v float64) bool {
		found := false
		for i := range q {
			if strings.HasPrefix(m.Names[i+1], prefix) {
				q[i] = v
				found = true
			}
		}
		return found
	}
	var ok bool
	yaw := geom.AxisAngle(geom.V3{0, 1, 0}, rng.Float64()*2*math.Pi)
	switch {
	case d < 0.25: // squat, leaning forward so it balances over flat feet
		// The deepest knee bend that still balances: with a 20° ankle
		// dorsiflexion limit (human chart) a flat-footed squat cannot go
		// as deep as the 90° knee limit.
		for knee := 1.45; knee >= 0.6; knee -= 0.05 {
			if r, sq, found := balancedSquat(w, knee-0.2*d); found {
				w.Place(yaw.Mul(r), sq, w.Base.P[0], w.Base.P[2])
				if !w.Upright() {
					return
				}
				break // too shallow to count as down
			}
		}
		LieDown(w, rng)
		return
	case d < 0.5: // kneel: thighs upright, shins flat behind, toes pointed
		ok = set("thigh", 0.1*rng.Float64()) && set("shin", -math.Pi/2) && set("foot", -0.6)
	default: // sit: legs out in front
		ok = set("thigh", math.Pi/2) && set("shin", -0.2*rng.Float64()) && set("foot", 0)
	}
	if !ok {
		LieDown(w, rng)
		return
	}
	for i := range q { // a little asymmetry so it is not always the same pose
		q[i] += 0.05 * rng.NormFloat64()
	}
	w.Place(yaw, q, w.Base.P[0], w.Base.P[2])
}

// balancedSquat finds a squat with the knees bent by knee (rad) that rests
// with flat feet and the centre of gravity over them: it searches the
// pelvis's forward lean and the hip angle, setting each ankle to keep the
// sole level. It returns the pelvis orientation and joint angles.
func balancedSquat(w *model.World, knee float64) (geom.M3, []float64, bool) {
	m := w.M
	n := m.NDof()
	roles := make([]string, n)
	for i := range roles {
		for _, p := range []string{"thigh", "shin", "foot"} {
			if strings.HasPrefix(m.Names[i+1], p) {
				roles[i] = p
			}
		}
	}
	ps := m.NewPose()
	bestErr := math.Inf(1)
	var bestR geom.M3
	var bestQ []float64
	for lean := 0.0; lean <= 1.2; lean += 0.05 { // pelvis pitched forward (nose down)
		r := geom.AxisAngle(geom.V3{0, 0, 1}, -lean)
		for hip := 0.2; hip <= 1.57; hip += 0.05 {
			q := make([]float64, n)
			for i, role := range roles {
				switch role {
				case "thigh":
					q[i] = hip
				case "shin":
					q[i] = -knee
				case "foot": // level sole: undo the lean, hip and knee
					q[i] = lean - hip + knee
				}
				q[i] = math.Max(m.Lo[i+1], math.Min(m.Hi[i+1], q[i]))
			}
			m.FKBase(r, geom.V3{}, q, ps)
			// Feet must end up level (ankle limits may have clipped).
			tilt := 0.0
			for i, role := range roles {
				if role == "foot" {
					fx := ps.R[i+1].MulV(geom.V3{1, 0, 0})
					tilt = math.Max(tilt, math.Abs(fx[1]))
				}
			}
			if tilt > 0.05 {
				continue
			}
			// Centre of gravity versus the middle of the lowest contacts.
			cog, _ := m.CoG(ps)
			lo := math.Inf(1)
			for _, c := range m.Contacts {
				lo = math.Min(lo, m.Point(ps, c.Link, c.Point)[1])
			}
			var sum float64
			cnt := 0
			for _, c := range m.Contacts {
				if p := m.Point(ps, c.Link, c.Point); p[1] < lo+0.01 {
					sum += p[0]
					cnt++
				}
			}
			if e := math.Abs(cog[0] - sum/float64(cnt)); e < bestErr {
				bestErr, bestR, bestQ = e, r, q
			}
		}
	}
	return bestR, bestQ, bestQ != nil && bestErr < 0.02
}
