// Package obs turns the simulated world into the network's inputs: one
// token per cylinder and one global vector. The centre of gravity is
// computed from the links in the current pose and enters both globally and
// per cylinder (anchor positions relative to the CoG). Each token also
// carries the cylinder's live moment arms, so pelvis and limb geometry is
// visible to the network.
package obs

import (
	"math"
	"sort"

	"mars/sim/geom"
	"mars/sim/model"
)

// Feature sizes. Changing them invalidates checkpoints.
const (
	TokenDim  = 15
	GlobalDim = 9
	// MaxArms is how many of a cylinder's moment arms (largest first) go
	// into its token. A graph network over links, joints and cylinders
	// will remove this cap.
	MaxArms = 2
)

// Normalisation scales, chosen so typical values land near [-1, 1].
const (
	velScale    = 0.1  // m/s
	areaScale   = 2e-4 // m²
	lengthScale = 0.5  // m
	armScale    = 0.05 // m
	massScale   = 10   // kg
	countScale  = 10   // cylinders
)

// IMU is the fused pelvis IMU reading.
type IMU struct {
	Pitch float64 // rad, 0 = upright
	Roll  float64 // rad
}

// Build returns the per-cylinder tokens and the global feature vector.
// goal is the commanded target in the robot's own frame (x forward, z right).
func Build(w *model.World, imu IMU, goal geom.V3) ([][]float64, []float64) {
	m := w.M
	cog, mass := w.CoG()
	ps := m.Supply.Pressure
	tokens := make([][]float64, len(m.Cyl))
	arms := make([]float64, 0, 4)
	for k, c := range m.Cyl {
		st := w.Cyl[k]
		b := m.Point(w.Pose, c.BaseLink, c.Base).Sub(cog)
		r := m.Point(w.Pose, c.RodLink, c.Rod).Sub(cog)
		arms = append(arms[:0], w.DLdq[k]...)
		sort.Slice(arms, func(i, j int) bool { return math.Abs(arms[i]) > math.Abs(arms[j]) })
		var top [MaxArms]float64
		copy(top[:], arms)
		tokens[k] = []float64{
			st.X/c.P.Stroke*2 - 1,
			st.V / velScale,
			st.PA / ps,
			st.PB / ps,
			st.Spool,
			c.P.BoreArea / areaScale,
			c.P.RodArea / c.P.BoreArea,
			b[0] / lengthScale, b[1] / lengthScale, b[2] / lengthScale,
			r[0] / lengthScale, r[1] / lengthScale, r[2] / lengthScale,
			top[0] / armScale, top[1] / armScale,
		}
	}
	global := []float64{
		mass / massScale,
		cog[0] / lengthScale, cog[1] / lengthScale, cog[2] / lengthScale,
		imu.Pitch, imu.Roll,
		goal[0], goal[2],
		float64(len(m.Cyl)) / countScale,
	}
	return tokens, global
}
