// Package env is the training environment: one robot in one world, the
// policy's control rate, randomisation (body, CoG, friction, pushes,
// starting on the floor), the LLM-loop stand-in that issues commands, and
// the reward. What is trained is chosen by a Scenario (see scenario.go).
//
// The movement network is rewarded for being up and following the current
// command. While it is down it is rewarded for getting up, so falls are
// something to recover from, the way a baby learns to kick and push by
// getting itself up off the floor. Where the objective is, and whether it
// has been reached, is the command loop's business (package command).
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
	"mars/sim/policy"
	"mars/sim/world"
)

const (
	PhysicsDt   = 50e-6
	Substeps    = 20 // physics steps per control tick
	ControlDt   = PhysicsDt * Substeps
	PolicyEvery = 10 // control ticks per policy decision
	PolicyDt    = ControlDt * PolicyEvery
)

// Env is one robot. It is not safe for concurrent use; run one per
// goroutine.
type Env struct {
	Cfg     Scenario
	Base    body.Body
	W       *model.World
	Tracker *world.Tracker
	Loop    *command.Loop
	Cmd     command.Command // command in force

	// Episode facts, read by the trainer when an episode ends.
	Assisted     bool // held by the harness
	StartedLying bool
	Falls        int // times it went from up to fallen
	GetUps       int // times it went from down to upright

	rng     *rand.Rand
	step    int
	h0      float64 // standing pelvis height
	prevP   geom.V3
	feet    []foot
	up      bool    // upright at the last step
	prevPot float64 // get-up shaping potential at the last step
	head    int     // the part highest up in the neutral pose (the sensor head's link)

	// Motion is the human reference being tracked while getting up (nil
	// if the scenario has none or the body has no recognisable legs).
	Motion       *MotionTracker
	refs         []Reference
	prevProgress float64
}

// foot is a link that carries the lowest contact points; stepping is
// rewarded through how long feet stay in the air before touching down.
type foot struct {
	contacts []int // indices into model contacts
	down     bool
	air      float64 // s since lift-off
}

func New(base body.Body, cfg Scenario, seed int64) *Env {
	e := &Env{Cfg: cfg, Base: base, rng: rand.New(rand.NewSource(seed))}
	if cfg.Reference != "" {
		e.refs, _ = LoadReferences(cfg.Reference) // validated with the scenario
	}
	e.Loop = command.NewLoop(command.Scripted{Cruise: 0.3, SlowDown: 1.5}, e.rng)
	e.Reset()
	return e
}

// Reset starts a new episode on a fresh (possibly random) body and world,
// standing or lying on the floor.
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
	e.h0 = e.W.StandHeight()
	e.findFeet() // in the neutral standing pose
	e.head = headLink(m)
	e.Tracker = world.NewTracker(wl)
	e.Loop.Reset(0)
	e.Cmd = command.Stop
	e.step = 0
	e.Falls, e.GetUps = 0, 0
	if e.rng.Float64() < e.Cfg.LyingFrac {
		FloorStart(e.W, e.rng, e.Cfg.StartLevel)
		for i := 0; i < int(0.3/PhysicsDt); i++ { // settle at rest, valves closed
			e.W.Step(PhysicsDt)
		}
		e.W.T = 0
	}
	e.prevP = e.W.Base.P
	e.up = e.W.Upright()
	e.Motion = PickTracker(e.refs, e.W)
	if e.Motion != nil {
		e.prevProgress = e.Motion.Progress
	}
	// A shallow crouch already counts as up; only starts that are really
	// down count as floor starts for the get-up statistics.
	e.StartedLying = !e.up
	e.prevPot = e.potential()
	e.Assisted = (!e.StartedLying || e.Cfg.AssistFloor) && e.Cfg.Assist > 0 && e.rng.Float64() >= e.Cfg.UnassistedFrac
	if e.Assisted {
		e.W.Harness = model.Harness{Strength: e.Cfg.Assist, Height: e.h0, Target: e.W.Base.P, Gentle: e.StartedLying}
	}
}

// LieDown puts the robot on its back, front or a side, facing a random
// way, legs at random angles.
func (e *Env) LieDown() { LieDown(e.W, e.rng) }

// LieDown puts a floating robot on its back, front or a side at its
// current horizontal position, facing a random way, legs at random angles.
func LieDown(w *model.World, rng *rand.Rand) {
	m := w.M
	q := make([]float64, m.NDof())
	for i := range q {
		lo, hi := m.Lo[i+1], m.Hi[i+1]
		q[i] = 0.6 * (lo + rng.Float64()*(hi-lo))
	}
	z, x := geom.V3{0, 0, 1}, geom.V3{1, 0, 0}
	lie := []geom.M3{
		geom.AxisAngle(z, math.Pi/2),  // on its back
		geom.AxisAngle(z, -math.Pi/2), // on its front
		geom.AxisAngle(x, math.Pi/2),  // on its left side
		geom.AxisAngle(x, -math.Pi/2), // on its right side
	}[rng.Intn(4)]
	yaw := geom.AxisAngle(geom.V3{0, 1, 0}, rng.Float64()*2*math.Pi)
	w.Place(yaw.Mul(lie), q, w.Base.P[0], w.Base.P[2])
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

// headLink is the part whose centre of mass is highest in the neutral
// pose: the chest that carries the head, or the pelvis on a legs-only body.
func headLink(m *model.Model) int {
	ps := m.NewPose()
	m.FK(make([]float64, m.NDof()), ps)
	best, hy := 0, math.Inf(-1)
	total := m.TotalMass()
	for i := range m.Names {
		if m.Mass[i] < obs.PartMinMass*total {
			continue
		}
		if y := m.Point(ps, i, m.Com[i])[1]; y > hy {
			best, hy = i, y
		}
	}
	return best
}

// PrivDim is the number of privileged features (see Observe).
const PrivDim = 11

// Observe returns the policy inputs for the current state, plus the
// privileged facts only the critic sees: pelvis height and velocity,
// floor friction, harness, progress along the human reference, whether
// it is up, how much of the episode is left, and balance over the feet.
func (e *Env) Observe() *policy.Obs {
	w := e.W
	o := obs.Build(w, obs.ReadIMU(w), e.Cmd)
	yaw := math.Atan2(-w.Base.R[2][0], w.Base.R[0][0])
	hd := geom.AxisAngle(geom.V3{0, 1, 0}, yaw).T()
	v := hd.MulV(w.Base.V).Scale(1 / math.Sqrt(9.81*e.h0))
	var prog, merr float64
	if e.Motion != nil {
		prog, merr = e.Motion.Progress, e.Motion.Error
	}
	up := 0.0
	if e.up {
		up = 1
	}
	o.Priv = []float64{
		w.Base.P[1] / e.h0,
		v[0], v[1], v[2],
		w.Ground.Mu,
		w.Harness.Strength,
		prog, merr,
		up,
		e.Seconds() / e.Cfg.EpisodeSeconds,
		e.overFeet(),
	}
	return o
}

// Step runs the command loop, applies one valve command per cylinder for
// PolicyDt, and returns the reward, whether the episode ended for good
// (diverged, or fell in a scenario that ends on falls), and whether the
// time limit ended it (truncated, so the return should be bootstrapped).
func (e *Env) Step(valves []float64) (r float64, terminal, truncated bool) {
	w := e.W
	if e.Cfg.Walk && e.up {
		e.Cmd = e.Loop.Update(w.T, e.Tracker.Perceive(w.T, w.Base.P, w.Base.R, false))
	} else if !e.up {
		e.Cmd = command.Stop // the LLM waits while the robot gets up
		e.Loop.Reset(w.T)
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
	if !w.Healthy() {
		return 0, true, false
	}
	up := w.Upright()
	switch {
	case e.up && w.Fallen():
		e.Falls++
		up = false
		if !e.Cfg.AssistFloor {
			w.Harness.Strength = 0 // let go; it has to get itself up
		}
		if e.Cfg.EndOnFall {
			return 0, true, false
		}
	case !e.up && up:
		e.GetUps++
	case e.up && !up && !w.Fallen():
		up = true // stumbling but not down: still counts as up
	}
	e.up = up
	e.track()
	if up {
		r = e.upReward(valves)
	} else {
		r = e.downReward(valves)
	}
	if e.Cfg.PushEvery > 0 && up && e.rng.Float64() < PolicyDt/e.Cfg.PushEvery {
		a := e.rng.Float64() * 2 * math.Pi
		s := e.Cfg.PushSpeed[0] + e.rng.Float64()*(e.Cfg.PushSpeed[1]-e.Cfg.PushSpeed[0])
		w.Base.V = w.Base.V.Add(geom.V3{s * math.Cos(a), 0, s * math.Sin(a)})
	}
	return r, false, float64(e.step)*PolicyDt >= e.Cfg.EpisodeSeconds
}

// potential measures how far up the robot is, 0..1: almost all pelvis
// height (as a fraction of standing height), with only a little for the
// pelvis being upright. Rewarding uprightness strongly taught the robot to
// straighten its trunk early in a rise, which puts the weight behind the
// feet and sits it back down; humans keep leaning forward until the legs
// have done most of the work (the references reward that lean).
func (e *Env) potential() float64 {
	h := math.Max(0, math.Min(1, e.W.Base.P[1]/e.h0))
	return 0.9*h + 0.1*math.Max(0, e.W.Up()[1])
}

// overFeet is 1 when the centre of gravity is over the middle of the feet
// that are on the floor, falling off over a few centimetres, and 0 when no
// foot is on the floor. Rising from a squat or kneeling needs the weight
// over the feet; letting it drift behind them is how the robot sat back
// down.
func (e *Env) overFeet() float64 {
	w := e.W
	var cx, cz float64
	n := 0
	for _, f := range e.feet {
		for _, c := range f.contacts {
			if w.ContactF[c][1] > 0 {
				p := w.M.Point(w.Pose, w.M.Contacts[c].Link, w.M.Contacts[c].Point)
				cx, cz, n = cx+p[0], cz+p[2], n+1
			}
		}
	}
	if n == 0 {
		return 0
	}
	cog, _ := w.CoG()
	dx, dz := cog[0]-cx/float64(n), cog[2]-cz/float64(n)
	return math.Exp(-(dx*dx + dz*dz) / (0.04 * 0.04))
}

// track measures the pose against the human reference, if any.
func (e *Env) track() {
	if e.Motion != nil {
		e.Motion.Update(e.W)
	}
}

// downReward: on the floor. Being down pays little; getting higher pays as
// it happens (potential-based shaping, so it cannot be farmed by bobbing up
// and down), keeping the weight over the feet pays, and standing pays far
// more. With a human reference, moving like the human (close to the
// reference path) and along it (progress towards standing) pays too. The
// steady part stays below the least that being up pays (0.2).
func (e *Env) downReward(valves []float64) float64 {
	pot := e.potential()
	shaping := 2 * (pot - e.prevPot)
	e.prevPot = pot
	r := 0.08*pot + 0.04*e.overFeet() + shaping - 0.02*effort(valves)
	if m := e.Motion; m != nil {
		close := math.Exp(-m.Error * m.Error / 0.09) // ~17° rms error halves it
		r += 0.06*close + 1.0*close*(m.Progress-e.prevProgress)
		e.prevProgress = m.Progress
	}
	return r
}

// upReward: stay upright at a sensible height without thrashing the
// valves, move at the commanded velocity, face the way you are told to go,
// and step (the kicking hint). Every step up pays at least about 0.2, more
// than the best moment on the floor.
func (e *Env) upReward(valves []float64) float64 {
	w := e.W
	e.prevPot = e.potential()
	if e.Motion != nil {
		e.prevProgress = e.Motion.Progress
	}
	tilt := 1 - w.Up()[1]
	height := math.Abs(w.Base.P[1]/e.h0 - 1)
	// The head carries the sensors: keep it level as well as the pelvis.
	headTilt := 1 - w.Pose.R[e.head][1][1]
	upright := math.Exp(-(10*tilt + 5*height + 6*headTilt + 0.1*w.Base.W.Norm() + 0.2*effort(valves)))

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

func effort(valves []float64) float64 {
	s := 0.0
	for _, c := range valves {
		s += c * c
	}
	return s / float64(len(valves))
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
	if h.Strength <= 0 {
		return
	}
	yaw := math.Atan2(-e.W.Base.R[2][0], e.W.Base.R[0][0])
	a := e.Cmd.HeadingDeg * math.Pi / 180
	fwd, right := geom.V3{math.Cos(yaw), 0, -math.Sin(yaw)}, geom.V3{math.Sin(yaw), 0, math.Cos(yaw)}
	h.Vel = fwd.Scale(e.Cmd.Speed * math.Cos(a)).Add(right.Scale(e.Cmd.Speed * math.Sin(a)))
}

// Seconds is the time since the episode started.
func (e *Env) Seconds() float64 { return float64(e.step) * PolicyDt }
