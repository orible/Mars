// Package hydraulics models a double-acting hydraulic cylinder driven by a
// proportional (servo-driven spool) valve.
//
// Model per cylinder:
//   - spool follows the command through a first-order lag (valve response)
//   - orifice flow Q = K·u·sqrt(ΔP), linearised near ΔP = 0 for numerical stability
//   - chamber pressure dP/dt = β/V · (Q ∓ A·v), with V growing with stroke
//   - piston force F = P_A·A_A − P_B·A_B − c·v − F_ext
//
// Pressure dynamics are stiff, so Step must be called with small dt
// (≈50 µs). Run many environments on goroutines to make up the cost.
package hydraulics

import "math"

// Params describe one cylinder plus its valve. Every value is meant to be
// measured on real hardware and written back into the body spec.
type Params struct {
	BoreArea   float64 `json:"boreArea"`   // m², cap-side piston area (A_A)
	RodArea    float64 `json:"rodArea"`    // m², annulus area on the rod side (A_B)
	Stroke     float64 `json:"stroke"`     // m
	DeadVolume float64 `json:"deadVolume"` // m³ per chamber, including hose
	Viscous    float64 `json:"viscous"`    // N·s/m, friction
	ValveGain  float64 `json:"valveGain"`  // m³/s per √Pa at full spool opening
	ValveTau   float64 `json:"valveTau"`   // s, spool time constant
	// ReliefPressure caps this cylinder's chamber pressures, setting its
	// force ceiling. 0 means 120 % of the supply pressure.
	ReliefPressure float64 `json:"reliefPressure,omitempty"`
	// PressureControl makes the valve a force servo: the command sets the
	// piston force (−1..1 of the rod side's full force) instead of the
	// spool opening, so the cylinder acts like a spring whose stiffness
	// the policy chooses (a tendon foot that conforms to the ground).
	PressureControl bool `json:"pressureControl,omitempty"`
	// Mass is the moving mass for the standalone bench model (Step) only;
	// in the articulated sim the piston mass is lumped into the links.
	Mass float64 `json:"mass,omitempty"`
}

// Supply describes the pump/tank side shared by all cylinders.
type Supply struct {
	Pressure    float64 `json:"pressure"`    // Pa, pump/relief pressure
	Tank        float64 `json:"tank"`        // Pa, return line pressure
	BulkModulus float64 `json:"bulkModulus"` // Pa, effective (oil + hose + entrained air)
}

// DefaultSupply is a hobby RC-excavator system at 35 bar.
func DefaultSupply() Supply {
	return Supply{Pressure: 3.5e6, Tank: 0, BulkModulus: 2e8}
}

// State is the dynamic state of one cylinder.
type State struct {
	X     float64 `json:"x"`     // m, extension from fully retracted
	V     float64 `json:"v"`     // m/s
	PA    float64 `json:"pA"`    // Pa, cap-side chamber
	PB    float64 `json:"pB"`    // Pa, rod-side chamber
	Spool float64 `json:"spool"` // -1..1, actual spool position (force target, lagged, under PressureControl)
}

// NewState places the piston mid-stroke at rest: the rod side at half
// supply and the cap side at the pressure that balances it, so the
// cylinder pushes neither way (equal pressures on unequal areas would
// push out with (A_A − A_B)·P/2, hundreds of newtons at full-size
// pressures).
func NewState(p Params, s Supply) State {
	pb := s.Pressure / 2
	return State{X: p.Stroke / 2, PA: pb * p.RodArea / p.BoreArea, PB: pb}
}

// pressureGain opens a force-controlled valve fully when the force is a
// quarter of its full value off target.
const pressureGain = 4

// linearBelow is the pressure difference below which orifice flow is treated
// as linear, avoiding the infinite slope of sqrt at zero.
const linearBelow = 1e4 // Pa (0.1 bar)

func orifice(hi, lo float64) float64 {
	d := hi - lo
	if math.Abs(d) < linearBelow {
		return d / math.Sqrt(linearBelow)
	}
	return math.Copysign(math.Sqrt(math.Abs(d)), d)
}

// Force returns the net hydraulic force on the piston (before friction and load).
func Force(p Params, st State) float64 {
	return st.PA*p.BoreArea - st.PB*p.RodArea
}

// StepPressures advances the valve and both chamber pressures by dt, given
// the piston position x (0..stroke) and velocity v imposed by the linkage.
// It records x and v in st. Use Force minus Viscous·v for the resulting force.
func StepPressures(p Params, s Supply, st *State, cmd, x, v, dt float64) {
	st.X, st.V = x, v
	cmd = math.Max(-1, math.Min(1, cmd))
	if p.ValveTau > 0 {
		st.Spool += (cmd - st.Spool) * math.Min(1, dt/p.ValveTau)
	} else {
		st.Spool = cmd
	}

	u := st.Spool
	if p.PressureControl {
		// Spool holds the (lagged) force target; the valve's own
		// pressure feedback is fast next to the pilot, so it acts at once.
		full := s.Pressure * p.RodArea
		u = math.Max(-1, math.Min(1, pressureGain*(st.Spool*full-Force(p, *st))/full))
	}
	var qA, qB float64
	if u >= 0 { // supply -> A, B -> tank
		qA = u * p.ValveGain * orifice(s.Pressure, st.PA)
		qB = -u * p.ValveGain * orifice(st.PB, s.Tank)
	} else { // A -> tank, supply -> B
		qA = u * p.ValveGain * orifice(st.PA, s.Tank)
		qB = -u * p.ValveGain * orifice(s.Pressure, st.PB)
	}

	vA := p.DeadVolume + p.BoreArea*st.X
	vB := p.DeadVolume + p.RodArea*(p.Stroke-st.X)
	st.PA += s.BulkModulus / vA * (qA - p.BoreArea*st.V) * dt
	st.PB += s.BulkModulus / vB * (qB + p.RodArea*st.V) * dt

	// Cavitation floor and relief valve.
	maxP := 1.2 * s.Pressure
	if p.ReliefPressure > 0 {
		maxP = p.ReliefPressure
	}
	st.PA = math.Max(0, math.Min(maxP, st.PA))
	st.PB = math.Max(0, math.Min(maxP, st.PB))
}

// Step is the standalone bench model: one cylinder pushing its own moving
// mass p.Mass against an external load fExt (N, opposing extension).
// cmd is the valve command in [-1, 1], positive extends.
func Step(p Params, s Supply, st *State, cmd, fExt, dt float64) {
	StepPressures(p, s, st, cmd, st.X, st.V, dt)
	f := Force(p, *st) - p.Viscous*st.V - fExt
	st.V += f / p.Mass * dt
	st.X += st.V * dt

	if st.X < 0 {
		st.X = 0
		if st.V < 0 {
			st.V = 0
		}
	} else if st.X > p.Stroke {
		st.X = p.Stroke
		if st.V > 0 {
			st.V = 0
		}
	}
}
