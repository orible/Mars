package model

import "math"

// The servo is the movement solver: the layer under the policy that turns
// where each cylinder should be into valve openings, every physics step.
//
// The policy used to set valve openings directly. A valve opening is a
// flow, so every error and every bit of exploration noise added up into
// drifting joints: with the valves shut the body stands like a statue, but
// 2% random valve noise toppled it within four seconds and the 31% noise
// training explores with within one, whatever the policy did. With a
// servo, the policy picks a target extension per cylinder around the
// standing pose; noise then means "aim slightly elsewhere", and holding
// still is the easy, default thing.

var (
	// servoRate is the servo's response rate as a fraction of 1/τ of the
	// valve (its spool lag): fast enough to hold a pose, slow enough that
	// the lag cannot make it oscillate (above about 0.3 the light arm and
	// trunk cylinders do; 0.05–0.2 all hold the standing pose).
	servoRate = 0.15
	// servoLead is the velocity lead, s: the servo aims where the piston
	// will be this much later, which damps it.
	servoLead = 0.01
	// ActionReach is how far an action of ±1 moves a target from the
	// standing pose, as a fraction of the stroke (clamped to the stroke).
	ActionReach = 0.5
)

// SetActionTargets turns policy actions (−1..1, one per cylinder) into
// servo targets: the standing-pose extension plus action × ActionReach ×
// stroke, clamped to the stroke. A pressure-controlled cylinder (a force
// servo of its own) takes its action as its force command. It switches
// the servo on.
func (w *World) SetActionTargets(a []float64) {
	w.Servo = true
	for k, c := range w.M.Cyl {
		u := math.Max(-1, math.Min(1, a[k]))
		if c.P.PressureControl {
			w.Targets[k] = u
			continue
		}
		w.Targets[k] = math.Max(0, math.Min(c.P.Stroke, w.Neutral[k]+u*ActionReach*c.P.Stroke))
	}
}

// HoldPose aims every cylinder at where it is now (force commands at 0).
func (w *World) HoldPose() {
	for k, c := range w.M.Cyl {
		if c.P.PressureControl {
			w.Targets[k] = 0
			continue
		}
		w.Targets[k] = math.Max(0, math.Min(c.P.Stroke, w.Len[k]-c.Retracted))
	}
}

// TargetError is cylinder k's target minus its extension, as a fraction
// of its stroke (for a pressure-controlled cylinder, its force command).
func (w *World) TargetError(k int) float64 {
	c := &w.M.Cyl[k]
	if c.P.PressureControl {
		return w.Targets[k]
	}
	return (w.Targets[k] - (w.Len[k] - c.Retracted)) / c.P.Stroke
}

// servoCommands sets each valve from its cylinder's target. Each cylinder
// gets its own gain from how fast it can move (its free speed at full
// valve opening): the same response rate everywhere, servoRate × 1/τ of
// its valve. One fixed gain for all made the light, fast arm and trunk
// cylinders oscillate between full open each way.
func (w *World) servoCommands() {
	s := w.M.Supply
	for k, c := range w.M.Cyl {
		if c.P.PressureControl {
			w.Cmd[k] = w.Targets[k]
			continue
		}
		x := w.Len[k] - c.Retracted
		v := w.LengthRate(k)
		vmax := c.P.ValveGain * math.Sqrt(math.Max(1, s.Pressure-s.Tank)/2) / c.P.RodArea // m/s, slowest direction
		rate := servoRate / math.Max(1e-3, c.P.ValveTau)                                  // 1/s
		w.Cmd[k] = math.Max(-1, math.Min(1, (w.Targets[k]-x-servoLead*v)*rate/vmax))
	}
}
