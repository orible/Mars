package optic

import (
	"sync"

	"mars/sim/model"
)

// Profile is a resolution, realism level and camera model.
type Profile struct {
	Rig    Rig
	Fast   bool // Realism.Fast
	Sensor SensorConfig
}

// FastProfile is vision for the RL loop: 64×48 per eye, hard shadows,
// one noise octave, the cut-down camera.
func FastProfile() Profile {
	r := DefaultRig()
	r.W, r.H = 64, 48
	return Profile{Rig: r, Fast: true, Sensor: FastSensor()}
}

// QualityProfile is the dataset profile: 128×96, soft shadows, the full
// camera model.
func QualityProfile() Profile { return Profile{Rig: DefaultRig(), Sensor: DefaultSensor()} }

// Eyes is one environment's vision: its own copy of the scene with the
// robot's body in it, the head mount and a pipeline whose temporal state
// (previous frame for flow, exposure) belongs to this environment.
type Eyes struct {
	Scene *Scene
	Body  *Body
	Head  HeadMount
	Pipe  *Pipeline
	base  int // objects before the body
	seed  int64
}

// NewEyes clones base, adds the body of m and builds the pipeline.
func NewEyes(base *Scene, m *model.Model, p Profile, seed int64) *Eyes {
	s := base.Clone()
	e := &Eyes{Scene: s, base: len(s.Objects), seed: seed}
	pipe := NewPipeline(s, p.Rig, seed)
	pipe.Realism.Fast = p.Fast
	pipe.Camera = NewStereo(p.Sensor, p.Rig.W, p.Rig.H, seed)
	e.Pipe = pipe
	e.SetBody(m)
	return e
}

// SetBody replaces the robot's body (after a morphology change) and
// forgets the temporal state.
func (e *Eyes) SetBody(m *model.Model) {
	e.Scene.Objects = e.Scene.Objects[:e.base]
	e.Head = FindHead(m)
	e.Body = AddBody(e.Scene, m, e.Head, e.seed)
	e.Reset()
}

// Reset starts a new episode: the next frame has no flow history and
// exposure re-meters.
func (e *Eyes) Reset() { e.Pipe.Reset() }

// Step renders the stereo frame for world w at time t (s): body capsules
// follow the links, the eyes follow the head. The frame's Token holds the
// eye token.
func (e *Eyes) Step(w *model.World, t float64) *StereoFrame {
	e.Body.Update(e.Scene, w)
	h, eye := e.Head.HeadPose(w)
	f := e.Pipe.Step(h, eye, t)
	f.Token = e.Head.Token(e.Pipe.Rig, w)
	return f
}

// Batch steps the vision of many environments across a fixed number of
// threads. Each environment is only ever touched by one goroutine at a
// time, so worlds may be read concurrently while physics is paused.
type Batch struct {
	Envs    []*Eyes
	Threads int
	out     []*StereoFrame
}

// NewBatch makes n environments' eyes on scene base for body m.
func NewBatch(n int, base *Scene, m *model.Model, p Profile, threads int, seed int64) *Batch {
	b := &Batch{Threads: max(1, threads), out: make([]*StereoFrame, n)}
	for i := 0; i < n; i++ {
		b.Envs = append(b.Envs, NewEyes(base, m, p, seed+int64(i)*1009))
	}
	return b
}

// Step renders every environment: ws[i] at time ts[i]. The returned
// frames are reused two steps later (copy what must be kept).
func (b *Batch) Step(ws []*model.World, ts []float64) []*StereoFrame {
	var wg sync.WaitGroup
	th := min(b.Threads, len(b.Envs))
	for j := 0; j < th; j++ {
		wg.Add(1)
		go func(j int) {
			defer wg.Done()
			for i := j; i < len(b.Envs); i += th {
				b.out[i] = b.Envs[i].Step(ws[i], ts[i])
			}
		}(j)
	}
	wg.Wait()
	return b.out
}

// Timing sums the stage times over all environments (CPU time, not wall
// time).
func (b *Batch) Timing() Timing {
	var t Timing
	for _, e := range b.Envs {
		x := e.Pipe.Timing
		t.Render += x.Render
		t.Truth += x.Truth
		t.Realism += x.Realism
		t.Camera += x.Camera
		t.Frames += x.Frames
	}
	return t
}
