package mocap

import "math"

// Features are what the robot's motion references compare: angles in
// degrees, height as a fraction of standing.
type Features struct {
	TrunkLean float64 // pelvis lean from vertical; + forward (face down), − backward (face up)
	Height    float64 // pelvis height / standing pelvis height
	Hip       [2]float64
	Knee      [2]float64
	Ankle     [2]float64 // raw foot rx relative to standing; sign fixed by the caller
}

// rootUp returns the pelvis up and facing vectors for Acclaim root angles
// (rotation order RX, RY, RZ applied in turn: M = Rz·Ry·Rx), Y up, the
// skeleton facing +Z.
func rootUp(rx, ry, rz float64) (up, facing [3]float64) {
	d := math.Pi / 180
	cx, sx := math.Cos(rx*d), math.Sin(rx*d)
	cy, sy := math.Cos(ry*d), math.Sin(ry*d)
	cz, sz := math.Cos(rz*d), math.Sin(rz*d)
	// Columns of M = Rz·Ry·Rx.
	m := [3][3]float64{
		{cz * cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx},
		{sz * cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx},
		{-sy, cy * sx, cy * cx},
	}
	up = [3]float64{m[0][1], m[1][1], m[2][1]}
	facing = [3]float64{m[0][2], m[1][2], m[2][2]}
	return
}

// Extract computes the features of every frame. standY is the root height
// when standing (in file units); hip flexion is −femur rx and knee flexion
// is +tibia rx in the CMU skeleton; the ankle is foot rx minus its value
// when standing (standFoot).
func Extract(frames []Frame, standY float64, standFoot [2]float64) []Features {
	out := make([]Features, len(frames))
	for i, f := range frames {
		up, facing := rootUp(f.Get("root", "rx"), f.Get("root", "ry"), f.Get("root", "rz"))
		lean := math.Acos(math.Max(-1, math.Min(1, up[1]))) * 180 / math.Pi
		if facing[1] > 0 { // facing upwards: leaning (or lying) back
			lean = -lean
		}
		x := Features{TrunkLean: lean, Height: f.Get("root", "ty") / standY}
		for s, side := range []string{"l", "r"} {
			x.Hip[s] = -f.Get(side+"femur", "rx")
			x.Knee[s] = f.Get(side+"tibia", "rx")
			x.Ankle[s] = f.Get(side+"foot", "rx") - standFoot[s]
		}
		out[i] = x
	}
	return out
}
