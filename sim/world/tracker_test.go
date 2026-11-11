package world

import (
	"math"
	"testing"

	"mars/sim/geom"
)

func TestBearingSignAndArrival(t *testing.T) {
	tr := NewTracker(World{Friction: 1, ArriveRadius: 0.3, Objectives: []XZ{{3, 3}, {3, 0}}})
	up := geom.Identity() // facing +x, right is +z

	p := tr.Perceive(0, geom.V3{}, up, false)
	if math.Abs(p.BearingDeg-45) > 1e-9 || math.Abs(p.DistanceM-math.Sqrt(18)) > 1e-9 {
		t.Fatalf("objective ahead-right: bearing %.1f°, distance %.2f m", p.BearingDeg, p.DistanceM)
	}
	// Turned 90° to the left (yaw +90° about up), the same objective is behind-right.
	p = tr.Perceive(0, geom.V3{}, geom.AxisAngle(geom.V3{0, 1, 0}, math.Pi/2), false)
	if math.Abs(p.BearingDeg-135) > 1e-9 {
		t.Fatalf("after turning left: bearing %.1f°, want 135°", p.BearingDeg)
	}

	at := geom.V3{3, 0.5, 2.9}
	if p = tr.Perceive(1, at, up, false); !p.Arrived || tr.Index != 0 {
		t.Fatalf("within the radius: arrived=%v index=%d", p.Arrived, tr.Index)
	}
	if p = tr.Perceive(1+Dwell, at, up, false); tr.Index != 1 || tr.Reached != 1 {
		t.Fatalf("after the dwell: index=%d reached=%d", tr.Index, tr.Reached)
	}
	if p = tr.Perceive(2+Dwell, at, up, false); p.Arrived || math.Abs(p.BearingDeg+90) > 1e-9 {
		t.Fatalf("next objective should be 90° to the left: %+v", p)
	}
}
