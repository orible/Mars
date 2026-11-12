package optic

import (
	"math"

	"mars/sim/geom"
)

// Flow is a per-pixel correspondence: the pixel centre (x+0.5, y+0.5) of
// one frame lands at (x+0.5+DX, y+0.5+DY) in another. Valid is false
// where the surface is hidden or off-screen in the other frame.
type Flow struct {
	DX, DY float32
	Valid  bool
}

// Correspond is the exact ground-truth correspondence from frame src to
// frame dst: every visible surface point of src (fixed in its object's
// frame) is moved with its object to dst's time and projected into dst's
// camera. It is optical flow when dst is the next or previous frame of
// the same eye and stereo correspondence when dst is the other eye.
// Occlusion: the point must be the surface dst actually sees there (same
// object, depth within tolerance).
func Correspond(src, dst *Frame, out []Flow) {
	for py := 0; py < src.H; py++ {
		for px := 0; px < src.W; px++ {
			k := py*src.W + px
			out[k] = Flow{}
			id := src.ID[k]
			l := src.Local[k]
			lv := geom.V3{float64(l[0]), float64(l[1]), float64(l[2])}
			var u, v, d float64
			var ok bool
			if id == 0 { // the sky is at infinity: only rotation matters
				q := dst.Cam.R.T().MulV(lv)
				if q[0] <= 1e-6 {
					continue
				}
				u, v, d, ok = dst.Cam.Cx+dst.Cam.F*q[2]/q[0], dst.Cam.Cy-dst.Cam.F*q[1]/q[0], math.Inf(1), true
			} else {
				p := dst.Poses[id-1]
				u, v, d, ok = dst.Cam.Project(p.R.MulV(lv).Add(p.P))
			}
			if !ok || u < 0 || v < 0 || u >= float64(dst.W) || v >= float64(dst.H) {
				continue
			}
			out[k].DX, out[k].DY = float32(u-float64(px)-0.5), float32(v-float64(py)-0.5)
			if visibleAt(dst, u, v, id, d) {
				out[k].Valid = true
			}
		}
	}
}

// visibleAt checks that dst sees object id at depth d near (u, v): one of
// the up to four pixels around the point shows it within tolerance.
func visibleAt(dst *Frame, u, v float64, id uint16, d float64) bool {
	x0, y0 := int(math.Floor(u-0.5)), int(math.Floor(v-0.5))
	tol := 0.02 + 0.02*d
	for yy := y0; yy <= y0+1; yy++ {
		for xx := x0; xx <= x0+1; xx++ {
			if xx < 0 || yy < 0 || xx >= dst.W || yy >= dst.H {
				continue
			}
			k := yy*dst.W + xx
			if dst.ID[k] != id {
				continue
			}
			if id == 0 || math.Abs(float64(dst.Depth[k])-d) < tol {
				return true
			}
		}
	}
	return false
}

// Disparity is the ground-truth disparity map of a rectified pair's left
// frame (left u − right u, pixels; 0 for sky), from depth alone.
func (r Rig) DisparityMap(left *Frame) []float32 {
	out := make([]float32, len(left.Depth))
	for k, d := range left.Depth {
		if !math.IsInf(float64(d), 1) {
			out[k] = float32(r.Disparity(float64(d)))
		}
	}
	return out
}

// GeometricEdges marks pixels on a silhouette or crease: where a
// 4-neighbour shows another object
// or a surface turned by more than 25° (objects are convex, so within one
// object there is no depth step). These are where edges must stay put
// through translation.
func GeometricEdges(f *Frame) []bool { return edges(f, true) }

// Silhouettes marks only pixels next to another object.
func Silhouettes(f *Frame) []bool { return edges(f, false) }

func edges(f *Frame, creases bool) []bool {
	out := make([]bool, len(f.ID))
	for y := 0; y < f.H; y++ {
		for x := 0; x < f.W; x++ {
			k := y*f.W + x
			for _, nb := range [2][2]int{{1, 0}, {0, 1}} {
				xx, yy := x+nb[0], y+nb[1]
				if xx >= f.W || yy >= f.H {
					continue
				}
				j := yy*f.W + xx
				if f.ID[k] != f.ID[j] || creases && edgeBetween(f, k, j) {
					out[k], out[j] = true, true
				}
			}
		}
	}
	return out
}

func edgeBetween(f *Frame, k, j int) bool {
	if f.ID[k] != f.ID[j] {
		return true
	}
	if f.ID[k] == 0 {
		return false
	}
	na, nb := f.Normal[k], f.Normal[j]
	return na[0]*nb[0]+na[1]*nb[1]+na[2]*nb[2] < 0.906
}
