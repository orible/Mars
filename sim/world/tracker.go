package world

import (
	"math"

	"mars/sim/command"
	"mars/sim/geom"
)

// Dwell is how long the robot is told it has arrived before the next
// objective appears, so stopping is part of the task.
const Dwell = 1.0

// Tracker follows a robot through a world's objectives and produces the
// true perception the LLM loop works from. It is the only place that
// knows where things are.
type Tracker struct {
	W         World
	Index     int     // current objective
	arrivedAt float64 // time of arrival at Index, or -1
	Reached   int
}

func NewTracker(w World) *Tracker { return &Tracker{W: w, arrivedAt: -1} }

// Perceive returns what the robot could know about the current objective
// at time t, given its pelvis position and orientation.
func (tr *Tracker) Perceive(t float64, pelvis geom.V3, r geom.M3, fallen bool) command.Perception {
	p := command.Perception{Fallen: fallen, Objectives: len(tr.W.Objectives) - tr.Index, ElapsedSecs: t}
	if tr.Index >= len(tr.W.Objectives) {
		return p
	}
	o := tr.W.Objectives[tr.Index]
	dx, dz := o[0]-pelvis[0], o[1]-pelvis[2]
	dist := math.Hypot(dx, dz)
	yaw := math.Atan2(-r[2][0], r[0][0])
	fwd := dx*math.Cos(yaw) - dz*math.Sin(yaw)
	right := dx*math.Sin(yaw) + dz*math.Cos(yaw)
	p.Visible, p.DistanceM, p.BearingDeg = true, dist, math.Atan2(right, fwd)*180/math.Pi
	if tr.arrivedAt < 0 && dist < tr.W.ArriveRadius {
		tr.arrivedAt = t
		tr.Reached++
	}
	if tr.arrivedAt >= 0 {
		p.Arrived = true
		if t-tr.arrivedAt >= Dwell {
			tr.Index++
			tr.arrivedAt = -1
		}
	}
	return p
}

// Objective returns the current objective, if any.
func (tr *Tracker) Objective() (XZ, bool) {
	if tr.Index >= len(tr.W.Objectives) {
		return XZ{}, false
	}
	return tr.W.Objectives[tr.Index], true
}
