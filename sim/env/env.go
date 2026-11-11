// Package env is the training environment: one robot on the floor, the
// policy's control rate, randomisation (body, CoG, floor friction, pushes)
// and the reward for the current task.
//
// The policy runs at 100 Hz and each command is held for ten 1 kHz control
// ticks of twenty 50 µs physics steps. The sim server uses the same
// timing, so a trained checkpoint behaves the same there.
package env

import (
	"math"
	"math/rand"

	"mars/sim/body"
	"mars/sim/geom"
	"mars/sim/model"
	"mars/sim/obs"
)

const (
	PhysicsDt   = 50e-6
	Substeps    = 20 // physics steps per control tick
	ControlDt   = PhysicsDt * Substeps
	PolicyEvery = 10 // control ticks per policy decision
	PolicyDt    = ControlDt * PolicyEvery
)

// Config sets the task's randomisation.
type Config struct {
	RandomBodyFrac float64    // fraction of episodes on a random body instead of the base spec
	EpisodeSeconds float64    // time limit
	PushEvery      float64    // mean seconds between pushes (0 = no pushes)
	PushSpeed      [2]float64 // m/s, range of the sideways velocity change a push gives
	Mu             [2]float64 // floor friction range
}

// DefaultStand is the standing-balance task.
func DefaultStand() Config {
	return Config{RandomBodyFrac: 0.5, EpisodeSeconds: 6, PushEvery: 2, PushSpeed: [2]float64{0.1, 0.4}, Mu: [2]float64{0.5, 1.1}}
}

// Env is one robot. It is not safe for concurrent use; run one per
// goroutine.
type Env struct {
	Cfg  Config
	Base body.Body
	W    *model.World
	rng  *rand.Rand
	step int
	h0   float64 // standing pelvis height
}

func New(base body.Body, cfg Config, seed int64) *Env {
	e := &Env{Cfg: cfg, Base: base, rng: rand.New(rand.NewSource(seed))}
	e.Reset()
	return e
}

// Reset starts a new episode on a fresh (possibly random) body.
func (e *Env) Reset() {
	b := e.Base
	if e.rng.Float64() < e.Cfg.RandomBodyFrac {
		b = body.Random(e.rng)
	}
	m, err := model.New(b)
	if err != nil {
		panic(err) // the base spec was validated on load; random bodies are tested to compile
	}
	e.W = model.NewWorld(m, true)
	e.W.Ground.Mu = e.Cfg.Mu[0] + e.rng.Float64()*(e.Cfg.Mu[1]-e.Cfg.Mu[0])
	e.step = 0
	e.h0 = e.W.StandHeight()
}

// Observe returns the policy inputs for the current state.
func (e *Env) Observe() ([][]float64, []float64) {
	return obs.Build(e.W, obs.ReadIMU(e.W), geom.V3{})
}

// Step applies one valve command per cylinder for PolicyDt. It returns the
// reward, whether the robot fell (terminal), and whether the time limit
// ended the episode (truncated, so the return should be bootstrapped).
func (e *Env) Step(cmd []float64) (r float64, fell, truncated bool) {
	w := e.W
	for k := range w.Cmd {
		w.Cmd[k] = math.Max(-1, math.Min(1, cmd[k]))
	}
	for i := 0; i < PolicyEvery*Substeps; i++ {
		w.Step(PhysicsDt)
	}
	e.step++
	if !w.Healthy() || w.Fallen() {
		return 0, true, false
	}
	r = e.reward(cmd)
	if e.Cfg.PushEvery > 0 && e.rng.Float64() < PolicyDt/e.Cfg.PushEvery {
		a := e.rng.Float64() * 2 * math.Pi
		s := e.Cfg.PushSpeed[0] + e.rng.Float64()*(e.Cfg.PushSpeed[1]-e.Cfg.PushSpeed[0])
		w.Base.V = w.Base.V.Add(geom.V3{s * math.Cos(a), 0, s * math.Sin(a)})
	}
	return r, false, float64(e.step)*PolicyDt >= e.Cfg.EpisodeSeconds
}

// reward for standing: stay alive, upright, at standing height, still, and
// without thrashing the valves. Every step upright pays between 0.5 and 1,
// so falling early is never better than staying up.
func (e *Env) reward(cmd []float64) float64 {
	w := e.W
	tilt := 1 - w.Up()[1]
	height := math.Abs(w.Base.P[1]/e.h0 - 1)
	speed := math.Hypot(w.Base.V[0], w.Base.V[2])
	effort := 0.0
	for _, c := range cmd {
		effort += c * c
	}
	effort /= float64(len(cmd))
	cost := 10*tilt + 10*height + speed + 0.1*w.Base.W.Norm() + 0.2*effort
	return 0.5 + 0.5*math.Exp(-cost)
}

// Seconds is the time since the episode started.
func (e *Env) Seconds() float64 { return float64(e.step) * PolicyDt }
