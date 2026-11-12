package optic

import (
	"time"

	"mars/sim/geom"
)

// StereoFrame is one time step of the rig: both eyes' renders and ground
// truth, the realism stage's radiance and the camera's output.
type StereoFrame struct {
	L, R *Frame
	// BackL and BackR are each eye's exact flow to the previous frame
	// (all invalid on the first frame); LR is the left-to-right stereo
	// correspondence.
	BackL, BackR, LR []Flow
	RadL, RadR       []RGB // realism stage (linear radiance)
	OutL, OutR       []RGB // camera output (sRGB)
	First            bool
}

func newStereoFrame(w, h int) *StereoFrame {
	n := w * h
	return &StereoFrame{L: NewFrame(w, h), R: NewFrame(w, h), BackL: make([]Flow, n), BackR: make([]Flow, n),
		LR: make([]Flow, n), RadL: make([]RGB, n), RadR: make([]RGB, n), OutL: make([]RGB, n), OutR: make([]RGB, n)}
}

// Timing is the time spent per stage, summed over frames.
type Timing struct {
	Render, Truth, Realism, Camera time.Duration
	Frames                         int
}

// Pipeline renders a stereo sequence: scene → clean stereo render and
// ground truth → realism → camera. It double-buffers frames so each step
// has the previous one for flow.
type Pipeline struct {
	Scene   *Scene
	Rig     Rig
	Realism *Realism
	Camera  *Stereo // nil skips the camera stage
	Timing  Timing

	r         Renderer
	cur, prev *StereoFrame
}

// NewPipeline sets up the default realism and camera for a rig.
func NewPipeline(s *Scene, rig Rig, seed int64) *Pipeline {
	return &Pipeline{Scene: s, Rig: rig, Realism: NewRealism(s), Camera: NewStereo(DefaultSensor(), rig.W, rig.H, seed),
		cur: newStereoFrame(rig.W, rig.H), prev: newStereoFrame(rig.W, rig.H)}
}

// Step moves the scene to time t, places the eyes for head orientation h
// and cyclopean eye position eye, and produces the next stereo frame. The
// returned frame is reused two steps later; Prev is the one before it.
func (p *Pipeline) Step(h geom.M3, eye geom.V3, t float64) *StereoFrame {
	p.cur, p.prev = p.prev, p.cur
	f, pf := p.cur, p.prev
	first := p.Timing.Frames == 0
	f.First = first

	t0 := time.Now()
	p.Scene.At(t)
	left, right := p.Rig.Eyes(h, eye)
	p.r.Render(p.Scene, left, t, f.L)
	p.r.Render(p.Scene, right, t, f.R)
	t1 := time.Now()
	if first {
		for i := range f.BackL {
			f.BackL[i], f.BackR[i] = Flow{}, Flow{}
		}
	} else {
		Correspond(f.L, pf.L, f.BackL)
		Correspond(f.R, pf.R, f.BackR)
	}
	Correspond(f.L, f.R, f.LR)
	t2 := time.Now()
	p.Realism.Translate(f.L, f.RadL)
	p.Realism.Translate(f.R, f.RadR)
	t3 := time.Now()
	if p.Camera != nil {
		var bl, br []Flow
		if !first {
			bl, br = f.BackL, f.BackR
		}
		p.Camera.Process(f.RadL, f.RadR, bl, br, f.OutL, f.OutR)
	}
	t4 := time.Now()
	p.Timing.Render += t1.Sub(t0)
	p.Timing.Truth += t2.Sub(t1)
	p.Timing.Realism += t3.Sub(t2)
	p.Timing.Camera += t4.Sub(t3)
	p.Timing.Frames++
	return f
}

// Prev is the frame before the last Step (nil before the second step).
func (p *Pipeline) Prev() *StereoFrame {
	if p.Timing.Frames < 2 {
		return nil
	}
	return p.prev
}
