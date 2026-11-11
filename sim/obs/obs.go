// Package obs turns the simulated world into the network's inputs: one
// token per cylinder and one global vector. The centre of gravity is
// computed from the links in the current pose and enters both globally and
// per cylinder (anchor positions relative to the CoG). Each token also
// carries the cylinder's live moment arms, so pelvis and limb geometry is
// visible to the network.
//
// Positions are expressed in the heading frame: centred on the pelvis,
// with gravity's up axis and the pelvis's forward direction projected onto
// the floor. Features therefore do not change when the robot walks
// somewhere else or turns around, but do change when it leans.
package obs

import (
	"math"
	"sort"

	"mars/sim/command"
	"mars/sim/geom"
	"mars/sim/model"
)

// Feature sizes. Changing them invalidates checkpoints.
const (
	TokenDim  = 15
	GlobalDim = 13
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
	gyroScale   = 5    // rad/s
)

// IMU is the fused pelvis IMU reading.
type IMU struct {
	Pitch float64 // rad, 0 = upright, positive = nose up
	Roll  float64 // rad, positive = right side down
	Gyro  geom.V3 // rad/s, pelvis frame
}

// ReadIMU is what an ideal IMU on the pelvis would report.
func ReadIMU(w *model.World) IMU {
	r := w.Base.R
	fwdY, rightY := r[1][0], r[1][2]
	return IMU{
		Pitch: math.Asin(math.Max(-1, math.Min(1, fwdY))),
		Roll:  math.Asin(math.Max(-1, math.Min(1, -rightY))),
		Gyro:  r.T().MulV(w.Base.W),
	}
}

// heading returns the map from world vectors to the heading frame.
func heading(w *model.World) geom.M3 {
	r := w.Base.R
	yaw := math.Atan2(-r[2][0], r[0][0]) // pelvis forward axis projected onto the floor
	return geom.AxisAngle(geom.V3{0, 1, 0}, yaw).T()
}

// Build returns the per-cylinder tokens and the global feature vector.
// cmd is the current command from the LLM loop; it is the network's only
// knowledge of where to go.
func Build(w *model.World, imu IMU, cmd command.Command) ([][]float64, []float64) {
	m := w.M
	h := heading(w)
	cogW, mass := w.CoG()
	cog := h.MulV(cogW.Sub(w.Base.P))
	ps := m.Supply.Pressure
	tokens := make([][]float64, len(m.Cyl))
	arms := make([]float64, 0, 4)
	for k, c := range m.Cyl {
		st := w.Cyl[k]
		b := h.MulV(m.Point(w.Pose, c.BaseLink, c.Base).Sub(cogW))
		r := h.MulV(m.Point(w.Pose, c.RodLink, c.Rod).Sub(cogW))
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
		imu.Gyro[0] / gyroScale, imu.Gyro[1] / gyroScale, imu.Gyro[2] / gyroScale,
		math.Cos(cmd.HeadingDeg * math.Pi / 180), math.Sin(cmd.HeadingDeg * math.Pi / 180), cmd.Speed / command.MaxSpeed,
		float64(len(m.Cyl)) / countScale,
	}
	return tokens, global
}
