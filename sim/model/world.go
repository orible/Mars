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
	limitK   = 100 // N·m/rad, joint limit stiffness
	limitC   = 1   // N·m·s/rad
)

// World is one simulated robot on the test stand: pelvis fixed, legs free,
// every cylinder's pressures coupled to the articulated dynamics.
type World struct {
	M    *Model
	Q    []float64
	Qd   []float64
	Cyl  []hydraulics.State
	Cmd  []float64 // valve commands, one per cylinder, in [-1, 1]
	Len  []float64 // cylinder pin-to-pin lengths for the current pose
	DLdq [][]float64
	Pose *Pose
	T    float64

	tau, bias, qdd []float64
	mm             [][]float64
}

// NewWorld starts in the neutral pose with chambers at half supply.
func NewWorld(m *Model) *World {
	n := m.NDof()
	w := &World{
		M: m, Q: make([]float64, n), Qd: make([]float64, n),
		Cyl: make([]hydraulics.State, len(m.Cyl)), Cmd: make([]float64, len(m.Cyl)),
		Len: make([]float64, len(m.Cyl)), DLdq: make([][]float64, len(m.Cyl)),
		Pose: m.NewPose(),
		tau:  make([]float64, n), bias: make([]float64, n), qdd: make([]float64, n),
		mm: make([][]float64, n),
	}
	for i := range w.mm {
		w.mm[i] = make([]float64, n)
	}
	for k, c := range m.Cyl {
		w.DLdq[k] = make([]float64, len(c.Span))
	}
	w.refresh()
	for k, c := range m.Cyl {
		w.Cyl[k] = hydraulics.NewState(c.P, m.Supply)
		w.Cyl[k].X = w.Len[k] - c.Retracted
	}
	return w
}

func (w *World) refresh() {
	w.M.FK(w.Q, w.Pose)
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

// Step advances the world by dt (≈50 µs; pressure dynamics are stiff).
func (w *World) Step(dt float64) {
	m := w.M
	for i := range w.tau {
		w.tau[i] = 0
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
			w.tau[q] += f * w.DLdq[k][i]
		}
	}
	for q := range w.Q {
		l := q + 1
		w.tau[q] -= m.Damping[l] * w.Qd[q]
		if w.Q[q] < m.Lo[l] {
			w.tau[q] += limitK*(m.Lo[l]-w.Q[q]) - limitC*w.Qd[q]
		} else if w.Q[q] > m.Hi[l] {
			w.tau[q] += limitK*(m.Hi[l]-w.Q[q]) - limitC*w.Qd[q]
		}
	}
	m.RNEA(w.Pose, w.Qd, m.zero, Gravity, w.bias)
	m.MassMatrix(w.Pose, w.mm)
	for q := range w.qdd {
		w.qdd[q] = w.tau[q] - w.bias[q]
	}
	solveSPD(w.mm, w.qdd)
	for q := range w.Q {
		w.Qd[q] += w.qdd[q] * dt
		w.Q[q] += w.Qd[q] * dt
	}
	w.T += dt
	w.refresh()
}

// Healthy reports whether the state is finite.
func (w *World) Healthy() bool {
	for q := range w.Q {
		if math.IsNaN(w.Q[q]) || math.IsInf(w.Q[q], 0) || math.IsNaN(w.Qd[q]) {
			return false
		}
	}
	return true
}

// CoG returns the whole-body centre of gravity (world) and total mass.
func (w *World) CoG() (geom.V3, float64) { return w.M.CoG(w.Pose) }
