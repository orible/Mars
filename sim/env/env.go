// Package env is the training environment: one robot in one world, the
// policy's control rate, randomisation (body, CoG, friction, pushes), the
// LLM-loop stand-in that issues commands, and the reward.
//
// The movement network is rewarded only for following the current command
// while staying upright. Where the objective is, and whether it has been
// reached, is the command loop's business (see package command). Standing
// is just the stop command.
//
// The policy runs at 100 Hz and each decision is held for ten 1 kHz
// control ticks of twenty 50 µs physics steps. The sim server uses the
// same timing, so a trained checkpoint behaves the same there.
package env

import (
	"math"
	"math/rand"

	"mars/sim/body"
	"mars/sim/command"
	"mars/sim/geom"
	"mars/sim/model"
	"mars/sim/obs"
	"mars/sim/world"
)

const (
	PhysicsDt   = 50e-6
	Substeps    = 20 // physics steps per control tick
	ControlDt   = PhysicsDt * Substeps
	PolicyEvery = 10 // control ticks per policy decision
	PolicyDt    = ControlDt * PolicyEvery
)

// Config sets the task and its randomisation.
type Config struct {
	Walk           bool               // false: always the stop command (standing)
	World          world.RandomConfig // worlds generated for walking
	RandomBodyFrac float64            // fraction of episodes on a random body instead of the base spec
	EpisodeSeconds float64            // time limit
	PushEvery      float64            // mean seconds between pushes (0 = no pushes)
	PushSpeed      [2]float64         // m/s, range of the sideways velocity change a push gives
	// Assist is the harness strength (0–1) for assisted episodes: the
	// pelvis is held up and carried along at the commanded velocity, as a
	// baby is held while it kicks. The trainer fades it as falls drop.
	Assist float64
	// UnassistedFrac of episodes never get the harness, so real ability
	// is always being practised and measured.
	UnassistedFrac float64
}

// DefaultStand is the standing-balance task.
func DefaultStand() Config {
	return Config{RandomBodyFrac: 0.5, EpisodeSeconds: 6, PushEvery: 2, PushSpeed: [2]float64{0.1, 0.4},
		World: world.RandomConfig{Count: 1, Distance: [2]float64{1, 1}, Friction: [2]float64{0.5, 1.1}, Arrive: 0.4}}
}

// DefaultWalk is walking towards objectives on a flat floor, directed by
// the scripted LLM stand-in.
func DefaultWalk() Config {
	return Config{Walk: true, World: world.DefaultRandom(), RandomBodyFrac: 0.5, EpisodeSeconds: 12,
		PushEvery: 4, PushSpeed: [2]float64{0.1, 0.3}, Assist: 1, UnassistedFrac: 0.2}
}

// Env is one robot. It is not safe for concurrent use; run one per
// goroutine.
type Env struct {
	Cfg     Config
	Base    body.Body
	W       *model.World
	Tracker *world.Tracker
	Loop    *command.Loop
	Cmd     command.Command // command in force

	Assisted bool // this episode has the harness

	rng   *rand.Rand
	step  int
	h0    float64 // standing pelvis height
	prevP geom.V3
	feet  []foot
}

// foot is a link that carries the lowest contact points; stepping is
// rewarded through how long feet stay in the air before touching down.
type foot struct {
	contacts []int // indices into model contacts
	down     bool
	air      float64 // s since lift-off
}

func New(base body.Body, cfg Config, seed int64) *Env {
	e := &Env{Cfg: cfg, Base: base, rng: rand.New(rand.NewSource(seed))}
	e.Loop = command.NewLoop(command.Scripted{Cruise: 0.3, SlowDown: 1.5}, e.rng)
	e.Reset()
	return e
}

// Reset starts a new episode on a fresh (possibly random) body and world.
func (e *Env) Reset() {
	b := e.Base
	if e.rng.Float64() < e.Cfg.RandomBodyFrac {
		b = body.Random(e.rng)
	}
	m, err := model.New(b)
	if err != nil {
		panic(err) // the base spec was validated on load; random bodies are tested to compile
	}
	wl := world.Random(e.rng, e.Cfg.World)
	e.W = model.NewWorld(m, true)
	e.W.Ground.Mu = wl.Friction
	e.Tracker = world.NewTracker(wl)
	e.Loop.Reset(0)
	e.Cmd = command.Stop
	e.step = 0
	e.h0 = e.W.StandHeight()
	e.prevP = e.W.Base.P
	e.findFeet()
	e.Assisted = e.Cfg.Assist > 0 && e.rng.Float64() >= e.Cfg.UnassistedFrac
	if e.Assisted {
		e.W.Harness = model.Harness{Strength: e.Cfg.Assist, Height: e.h0, Target: e.W.Base.P}
	}
}

// findFeet groups contact points by link and keeps the links whose
// lowest point is within 2 cm of the ground in the neutral pose.
func (e *Env) findFeet() {
	m, w := e.W.M, e.W
	e.feet = e.feet[:0]
	byLink := map[int]int{}
	for i, c := range m.Contacts {
		if m.Point(w.Pose, c.Link, c.Point)[1] > 0.02 {
			continue
		}
		j, ok := byLink[c.Link]
		if !ok {
			j = len(e.feet)
			byLink[c.Link] = j
			e.feet = append(e.feet, foot{down: true})
		}
		e.feet[j].contacts = append(e.feet[j].contacts, i)
	}
}

// stepping rewards feet that land after a decent swing while the command
// says to move: the kicking hint.
func (e *Env) stepping() float64 {
	r := 0.0
	moving := e.Cmd.Speed > 0.05
	for i := range e.feet {
		f := &e.feet[i]
		down := false
		for _, c := range f.contacts {
			if e.W.ContactF[c][1] > 0 {
				down = true
				break
			}
		}
		if down && !f.down && moving {
			r += math.Min(f.air, 0.5) - 0.2 // short shuffles cost, swings up to 0.5 s pay
		}
		if down {
			f.air = 0
		} else {
			f.air += PolicyDt
		}
		f.down = down
	}
	return r
}

// carry points the harness along the command.
func (e *Env) carry() {
	h := &e.W.Harness
	if !e.Assisted {
		return
	}
	yaw := math.Atan2(-e.W.Base.R[2][0], e.W.Base.R[0][0])
	a := e.Cmd.HeadingDeg * math.Pi / 180
	fwd, right := geom.V3{math.Cos(yaw), 0, -math.Sin(yaw)}, geom.V3{math.Sin(yaw), 0, math.Cos(yaw)}
	h.Vel = fwd.Scale(e.Cmd.Speed * math.Cos(a)).Add(right.Scale(e.Cmd.Speed * math.Sin(a)))
}

// Observe returns the policy inputs for the current state.
func (e *Env) Observe() ([][]float64, []float64) {
	return obs.Build(e.W, obs.ReadIMU(e.W), e.Cmd)
}

// Step runs the command loop, applies one valve command per cylinder for
// PolicyDt, and returns the reward, whether the robot fell (terminal), and
// whether the time limit ended the episode (truncated, so the return
// should be bootstrapped).
func (e *Env) Step(valves []float64) (r float64, fell, truncated bool) {
	w := e.W
	if e.Cfg.Walk {
		e.Cmd = e.Loop.Update(w.T, e.Tracker.Perceive(w.T, w.Base.P, w.Base.R, false))
	}
	e.carry()
	for k := range w.Cmd {
		w.Cmd[k] = math.Max(-1, math.Min(1, valves[k]))
	}
	e.prevP = w.Base.P
	for i := 0; i < PolicyEvery*Substeps; i++ {
		w.Step(PhysicsDt)
	}
	e.step++
	if !w.Healthy() || w.Fallen() {
		return 0, true, false
	}
	r = e.reward(valves)
	if e.Cfg.PushEvery > 0 && e.rng.Float64() < PolicyDt/e.Cfg.PushEvery {
		a := e.rng.Float64() * 2 * math.Pi
		s := e.Cfg.PushSpeed[0] + e.rng.Float64()*(e.Cfg.PushSpeed[1]-e.Cfg.PushSpeed[0])
		w.Base.V = w.Base.V.Add(geom.V3{s * math.Cos(a), 0, s * math.Sin(a)})
	}
	return r, false, float64(e.step)*PolicyDt >= e.Cfg.EpisodeSeconds
}

// reward: stay upright at a sensible height without thrashing the valves,
// move at the commanded velocity, and face the way you are told to go.
// Every step upright pays at least 0.2, so falling early never pays.
func (e *Env) reward(valves []float64) float64 {
	w := e.W
	tilt := 1 - w.Up()[1]
	height := math.Abs(w.Base.P[1]/e.h0 - 1)
	effort := 0.0
	for _, c := range valves {
		effort += c * c
	}
	effort /= float64(len(valves))
	upright := math.Exp(-(10*tilt + 5*height + 0.1*w.Base.W.Norm() + 0.2*effort))

	// Pelvis velocity over the step, in the heading frame.
	yaw := math.Atan2(-w.Base.R[2][0], w.Base.R[0][0])
	dx, dz := (w.Base.P[0]-e.prevP[0])/PolicyDt, (w.Base.P[2]-e.prevP[2])/PolicyDt
	vf := dx*math.Cos(yaw) - dz*math.Sin(yaw)
	vr := dx*math.Sin(yaw) + dz*math.Cos(yaw)
	h := e.Cmd.HeadingDeg * math.Pi / 180
	ef, er := vf-e.Cmd.Speed*math.Cos(h), vr-e.Cmd.Speed*math.Sin(h)
	track := math.Exp(-(ef*ef + er*er) / 0.04)

	facing := 1.0
	if e.Cmd.Speed > 0.05 {
		facing = 0.5 * (1 + math.Cos(h))
	}
	return 0.2 + 0.2*upright + 0.5*track + 0.1*facing + 0.5*e.stepping()
}

// Seconds is the time since the episode started.
func (e *Env) Seconds() float64 { return float64(e.step) * PolicyDt }
