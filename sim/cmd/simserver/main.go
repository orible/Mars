// Command simserver runs the 3D simulator and streams frames to the Blazor
// observer over WebSocket (ws://localhost:8080/ws by default).
//
// Iteration loop: the body spec file is polled for changes and hot-reloaded.
// The policy weights stay loaded across reloads; the network is
// morphology-agnostic, so the same weights keep running on the edited body.
//
// Run with -check to print each cylinder's reachable joint range for the
// spec (pelvis and limb layout check) and exit.
//
// By default the robot stands free on a flat floor (floating pelvis, foot
// contact points from the spec) and is put back upright 1 s after it
// falls. -stand bolts the pelvis to the world instead (test stand), and
// -hold closes every valve so the cylinders lock the joints.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"

	"mars/sim/body"
	"mars/sim/command"
	"mars/sim/env"
	"mars/sim/geom"
	"mars/sim/model"
	"mars/sim/obs"
	"mars/sim/policy"
	"mars/sim/world"
)

// Timing matches training (package env): 50 µs physics, 1 kHz control
// ticks, a policy decision every 10 ms.
const (
	controlDt = env.ControlDt
	substeps  = env.Substeps
	physicsDt = env.PhysicsDt
	frameDt   = 33 * time.Millisecond
)

// Wire format the observer reads. Positions are world coordinates in
// metres: X forward, Y up, Z right.
type LinkFrame struct {
	Name   string  `json:"name"`
	Parent int     `json:"parent"`
	Origin geom.V3 `json:"origin"`
	Com    geom.V3 `json:"com"`
}

type CylFrame struct {
	Name    string  `json:"name"`
	Base    geom.V3 `json:"base"`
	Rod     geom.V3 `json:"rod"`
	Stroke  float64 `json:"stroke"`
	X       float64 `json:"x"`
	V       float64 `json:"v"`
	PA      float64 `json:"pA"`
	PB      float64 `json:"pB"`
	Command float64 `json:"cmd"`
}

type ContactFrame struct {
	P geom.V3 `json:"p"` // world position
	F geom.V3 `json:"f"` // ground force on the robot, N
}

type Frame struct {
	T         float64        `json:"t"`
	BodyName  string         `json:"body"`
	BodyRev   int            `json:"bodyRev"`
	Floating  bool           `json:"floating"`
	Mass      float64        `json:"mass"`
	CoG       geom.V3        `json:"cog"`
	Pelvis    geom.V3        `json:"pelvis"`
	Yaw       float64        `json:"yaw"` // rad, heading about up (0 = +x, positive turns left)
	Pitch     float64        `json:"pitch"`
	Roll      float64        `json:"roll"`
	Falls     int            `json:"falls"`
	Supply    float64        `json:"supply"`
	Links     []LinkFrame    `json:"links"`
	Cylinders []CylFrame     `json:"cylinders"`
	Contacts  []ContactFrame `json:"contacts"`
	// Command loop: what the LLM (or its stand-in) was told and said.
	Objective *world.XZ       `json:"objective"`
	Reached   int             `json:"reached"`
	Prompt    string          `json:"prompt"`
	Command   command.Command `json:"command"`
	Commander string          `json:"commander"`
	Hands     geom.V3         `json:"hands"` // harness force on the pelvis, N (zero when not held)
	Joints    []JointFrame    `json:"joints"`
}

// JointFrame is one joint's angle within its limits. Kind names the
// motion from the joint axis: about X (forward) is roll, about Y (up) is
// yaw, about Z (right) is pitch.
type JointFrame struct {
	Name  string     `json:"name"`
	Kind  string     `json:"kind"`
	Q     float64    `json:"q"`
	Lo    float64    `json:"lo"`
	Hi    float64    `json:"hi"`
	Reach [2]float64 `json:"reach"` // the range its cylinders' strokes allow
}

func axisKind(a geom.V3) string {
	ax, ay, az := math.Abs(a[0]), math.Abs(a[1]), math.Abs(a[2])
	switch {
	case ax >= ay && ax >= az:
		return "roll"
	case ay >= az:
		return "yaw"
	}
	return "pitch"
}

// llmLink is the state shared with the HTTP endpoints an outside LLM
// uses: it reads the perception and posts commands.
type llmLink struct {
	mu         sync.Mutex
	perception command.Perception
	prompt     string
	posted     *command.Command // latest command from outside, not yet applied
}

type sim struct {
	tracker  *world.Tracker
	worldCfg *world.World // nil: random worlds
	loop     *command.Loop
	external bool    // commands come from POST /command instead of the scripted stand-in
	assist   float64 // harness strength (training's helping hands), 0 = none
	reach    []model.ReachRow
	llm      *llmLink
	cmd      command.Command

	world    *model.World
	rev      int
	pol      *policy.SetPolicy
	floating bool
	hold     bool
	fallenAt float64 // sim time the robot fell, or -1
	falls    int
	tick     int
	policies <-chan *policy.SetPolicy
	push     float64 // mean seconds between random pushes, 0 = none
	rng      *rand.Rand
	scen     env.Scenario
	up       bool // upright (or stumbling, not yet fallen)
}

func (s *sim) reset(b body.Body) error {
	m, err := model.New(b)
	if err != nil {
		return err
	}
	if s.floating && len(m.Contacts) == 0 {
		return fmt.Errorf("body %q has no contact points, so it cannot stand on the floor; add \"contacts\" to its feet or run with -stand", b.Name)
	}
	s.world = model.NewWorld(m, s.floating)
	s.restart()
	s.rev++
	limited := 0
	s.reach = m.Reach(200)
	for _, r := range s.reach {
		if !r.Full() {
			limited++
		}
	}
	log.Printf("body %q (rev %d): %d links, %d cylinders, %d cylinder/joint pairs limited by stroke (run -check for detail); same policy weights",
		b.Name, s.rev, len(b.Links), len(b.Cylinders), limited)
	return nil
}

func (s *sim) control() {
	w := s.world
	select {
	case p := <-s.policies:
		s.pol = p
	default:
	}
	if !s.hold && s.tick%env.PolicyEvery == 0 {
		s.command()
		tokens, global := obs.Build(w, obs.ReadIMU(w), s.cmd)
		copy(w.Cmd, s.pol.Act(tokens, global))
	}
	s.tick++
	if s.push > 0 && w.Floating && s.up && s.rng.Float64() < controlDt/s.push {
		// Same pushes as training: a sudden sideways velocity change.
		lo, hi := s.scen.PushSpeed[0], s.scen.PushSpeed[1]
		a, v := s.rng.Float64()*2*math.Pi, lo+s.rng.Float64()*(hi-lo)
		w.Base.V = w.Base.V.Add(geom.V3{v * math.Cos(a), 0, v * math.Sin(a)})
	}
	for i := 0; i < substeps; i++ {
		w.Step(physicsDt)
	}
	switch {
	case !w.Healthy():
		log.Printf("simulation diverged at t=%.3f s; resetting pose", w.T)
		s.restart()
		return
	case s.up && w.Fallen():
		s.up, s.fallenAt = false, w.T
		s.falls++
		w.Harness.Strength = 0 // let go; it has to get itself up
	case !s.up && w.Upright():
		s.up, s.fallenAt = true, -1
	}
	// Scenarios that end on a fall reset 1 s later; the others leave the
	// robot to get itself up, and only reset if it is still down after 15 s.
	if s.fallenAt >= 0 && (s.scen.EndOnFall && w.T-s.fallenAt > 1 || w.T-s.fallenAt > 15) {
		s.restart()
	}
}

// restart puts the same body back at the origin, standing or (as often as
// the scenario says) lying on the floor, keeping sim time, and restarts
// the world's objectives.
func (s *sim) restart() {
	t := s.world.T
	s.world = model.NewWorld(s.world.M, s.floating)
	s.world.T = t
	s.fallenAt = -1
	s.up = true
	s.newWorld(geom.V3{})
	if s.floating && s.rng.Float64() < s.scen.LyingFrac {
		env.LieDown(s.world, s.rng)
		s.up, s.fallenAt = false, t
	}
}

// newWorld starts the objectives again: the -world file, or a fresh random
// chain laid out from the robot's current position.
func (s *sim) newWorld(from geom.V3) {
	var wl world.World
	if s.worldCfg != nil {
		wl = *s.worldCfg
	} else {
		wl = world.Random(s.rng, world.DefaultRandom())
		for i := range wl.Objectives {
			wl.Objectives[i][0] += from[0]
			wl.Objectives[i][1] += from[2]
		}
	}
	s.world.Ground.Mu = wl.Friction
	if s.assist > 0 && s.world.Floating {
		s.world.Harness = model.Harness{Strength: s.assist, Height: s.world.StandHeight(), Target: s.world.Base.P}
	}
	s.tracker = world.NewTracker(wl)
	s.loop.Reset(s.world.T)
	s.cmd = command.Stop
}

// command runs one turn of the perception → LLM → command loop.
func (s *sim) command() {
	w := s.world
	p := s.tracker.Perceive(w.T, w.Base.P, w.Base.R, !s.up)
	if p.Objectives == 0 && s.worldCfg == nil {
		s.newWorld(w.Base.P) // random worlds go on forever
		p = s.tracker.Perceive(w.T, w.Base.P, w.Base.R, !s.up)
	}
	s.llm.mu.Lock()
	s.llm.perception, s.llm.prompt = p, p.Prompt()
	posted := s.llm.posted
	s.llm.posted = nil
	s.llm.mu.Unlock()
	if s.external {
		if posted != nil {
			s.cmd = *posted
		}
	} else if s.scen.Walk && s.up {
		s.cmd = s.loop.Update(w.T, p)
	} else {
		s.cmd = command.Stop // stand still, or wait while it gets up
		s.loop.Reset(w.T)
	}
	if w.Harness.Strength > 0 {
		yaw := math.Atan2(-w.Base.R[2][0], w.Base.R[0][0])
		a := s.cmd.HeadingDeg * math.Pi / 180
		fwd, right := geom.V3{math.Cos(yaw), 0, -math.Sin(yaw)}, geom.V3{math.Sin(yaw), 0, math.Cos(yaw)}
		w.Harness.Vel = fwd.Scale(s.cmd.Speed * math.Cos(a)).Add(right.Scale(s.cmd.Speed * math.Sin(a)))
	}
}

func (s *sim) frame() Frame {
	w := s.world
	m := w.M
	cog, mass := w.CoG()
	imu := obs.ReadIMU(w)
	f := Frame{T: w.T, BodyName: m.Body.Name, BodyRev: s.rev, Floating: w.Floating, Mass: mass, CoG: cog,
		Pelvis: w.Base.P, Yaw: math.Atan2(-w.Base.R[2][0], w.Base.R[0][0]), Pitch: imu.Pitch, Roll: imu.Roll, Falls: s.falls, Supply: m.Supply.Pressure,
		Contacts: []ContactFrame{}, Reached: s.tracker.Reached, Command: s.cmd, Commander: "scripted", Hands: w.Harness.Force}
	if s.external {
		f.Commander = "external (POST /command)"
	}
	if o, ok := s.tracker.Objective(); ok {
		f.Objective = &o
	}
	s.llm.mu.Lock()
	f.Prompt = s.llm.prompt
	s.llm.mu.Unlock()
	for i, c := range m.Contacts {
		f.Contacts = append(f.Contacts, ContactFrame{P: m.Point(w.Pose, c.Link, c.Point), F: w.ContactF[i]})
	}
	for q := range w.Q {
		l := q + 1
		j := JointFrame{Name: m.Names[l], Kind: axisKind(m.Axis[l]), Q: w.Q[q], Lo: m.Lo[l], Hi: m.Hi[l], Reach: [2]float64{m.Lo[l], m.Hi[l]}}
		for _, r := range s.reach {
			if r.Joint == j.Name {
				j.Reach[0], j.Reach[1] = math.Max(j.Reach[0], r.Lo), math.Min(j.Reach[1], r.Hi)
			}
		}
		f.Joints = append(f.Joints, j)
	}
	for i, name := range m.Names {
		f.Links = append(f.Links, LinkFrame{Name: name, Parent: m.Parent[i], Origin: w.Pose.P[i], Com: m.Point(w.Pose, i, m.Com[i])})
	}
	for k, c := range m.Cyl {
		st := w.Cyl[k]
		f.Cylinders = append(f.Cylinders, CylFrame{
			Name: c.Name, Base: m.Point(w.Pose, c.BaseLink, c.Base), Rod: m.Point(w.Pose, c.RodLink, c.Rod),
			Stroke: c.P.Stroke, X: st.X, V: st.V, PA: st.PA, PB: st.PB, Command: w.Cmd[k],
		})
	}
	return f
}

// watch polls the spec file and sends each valid new version on out.
func watch(ctx context.Context, path string, out chan<- body.Body) {
	var last time.Time
	tick := time.NewTicker(500 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
		info, err := os.Stat(path)
		if err != nil || !info.ModTime().After(last) {
			continue
		}
		last = info.ModTime()
		b, err := body.Load(path)
		if err != nil {
			log.Printf("body spec not reloaded: %v", err)
			continue
		}
		out <- b
	}
}

// watchPolicy polls a checkpoint file (the trainer rewrites it every
// iteration) and sends each new version on out.
func watchPolicy(ctx context.Context, path string, out chan<- *policy.SetPolicy) {
	var last time.Time
	tick := time.NewTicker(time.Second)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
		info, err := os.Stat(path)
		if err != nil || !info.ModTime().After(last) {
			continue
		}
		last = info.ModTime()
		p, err := loadPolicy(path)
		if err != nil {
			log.Printf("policy not reloaded: %v", err)
			continue
		}
		select {
		case out <- p:
		default:
		}
	}
}

func loadPolicy(path string) (*policy.SetPolicy, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	p, err := policy.Load(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if p.TokenDim != obs.TokenDim || p.GlobalDim != obs.GlobalDim {
		return nil, fmt.Errorf("%s: checkpoint features %d/%d, simulator provides %d/%d", path, p.TokenDim, p.GlobalDim, obs.TokenDim, obs.GlobalDim)
	}
	return p, nil
}

func deg(r float64) float64 { return r * 180 / math.Pi }

func check(path string) {
	b, err := body.Load(path)
	if err != nil {
		log.Fatal(err)
	}
	m, err := model.New(b)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%-14s %-12s %20s %20s\n", "cylinder", "joint", "reachable (deg)", "joint limits (deg)")
	for _, r := range m.Reach(400) {
		note := ""
		if !r.Full() {
			note = "  limited by stroke"
		}
		fmt.Printf("%-14s %-12s %9.1f .. %7.1f %9.1f .. %7.1f%s\n", r.Cylinder, r.Joint, deg(r.Lo), deg(r.Hi), deg(r.LimitLo), deg(r.LimitHi), note)
	}
	fmt.Println()
	if v := m.CornerViolations(); len(v) > 0 {
		fmt.Println("Combined joint extremes past a cylinder's end stop:")
		for _, s := range v {
			fmt.Println("  " + s)
		}
	} else {
		fmt.Println("Multi-joint cylinders stay within stroke at every combination of joint limits.")
	}
}

func main() {
	addr := flag.String("addr", "localhost:8080", "listen address")
	bodyPath := flag.String("body", "bodies/biped.json", "body spec to load and watch")
	randomEvery := flag.Duration("random", 0, "if > 0, replace the body with a random one at this interval (morphology randomisation demo)")
	ckpt := flag.String("policy", "", "policy checkpoint (JSON), reloaded when the file changes; random weights if empty")
	seed := flag.Int64("seed", 1, "random seed")
	checkOnly := flag.Bool("check", false, "print the cylinder reach report for -body and exit")
	stand := flag.Bool("stand", false, "test stand: bolt the pelvis to the world instead of standing on the floor")
	hold := flag.Bool("hold", false, "close every valve (cylinders lock the joints) instead of running the policy")
	push := flag.Float64("push", -1, "push the robot sideways on average every this many seconds (0 = never; default: the scenario's)")
	scenarioArg := flag.String("scenario", "life", "scenario to watch ("+strings.Join(env.ScenarioNames(), ", ")+", or a .json file): whether it walks, starts on the floor, and is reset or left to get up after a fall")
	speed := flag.Float64("speed", 1, "simulated seconds per wall-clock second; 0 = as fast as one core allows (the trainer always runs flat out on every core)")
	assist := flag.Float64("assist", 0, "harness strength 0–1: hold the pelvis up and carry it along the command, as in early training")
	worldPath := flag.String("world", "", "world file (objectives, friction); random worlds if empty")
	commander := flag.String("commander", "scripted", "who issues walking commands: scripted (the training stand-in for the LLM) or external (an LLM reading GET /perception and posting to /command)")
	flag.Parse()
	if *commander != "scripted" && *commander != "external" {
		log.Fatalf("-commander must be scripted or external, not %q", *commander)
	}
	var worldCfg *world.World
	if *worldPath != "" {
		wl, err := world.Load(*worldPath)
		if err != nil {
			log.Fatal(err)
		}
		worldCfg = &wl
	}
	link := &llmLink{}
	scen, err := env.LoadScenario(*scenarioArg)
	if err != nil {
		log.Fatal(err)
	}
	if *push < 0 {
		*push = scen.PushEvery
	}
	log.Printf("scenario %q: %s", scen.Name, scen.Description)

	if *checkOnly {
		check(*bodyPath)
		return
	}

	rng := rand.New(rand.NewSource(*seed))
	pol := policy.New(rng, obs.TokenDim, obs.GlobalDim, 32, 64)
	ctx := context.Background()
	policies := make(chan *policy.SetPolicy, 1)
	if *ckpt != "" {
		p, err := loadPolicy(*ckpt)
		if err != nil {
			log.Fatal(err)
		}
		pol = p
		go watchPolicy(ctx, *ckpt, policies)
	}

	bodies := make(chan body.Body, 1)
	if *randomEvery > 0 {
		go func() {
			bodies <- body.Random(rng)
			for range time.Tick(*randomEvery) {
				bodies <- body.Random(rng)
			}
		}()
	} else {
		go watch(ctx, *bodyPath, bodies)
	}

	var latest atomic.Pointer[[]byte]
	go func() {
		s := &sim{pol: pol, floating: !*stand, hold: *hold, policies: policies, push: *push, rng: rng, scen: scen,
			worldCfg: worldCfg, external: *commander == "external", llm: link, assist: *assist,
			loop: command.NewLoop(command.Scripted{Cruise: 0.3, SlowDown: 1.5}, rng)}
		for s.world == nil {
			if err := s.reset(<-bodies); err != nil {
				log.Printf("body rejected: %v", err)
			}
		}
		tick := time.NewTicker(frameDt)
		defer tick.Stop()
		owed := 0.0 // control ticks owed at the requested speed
		for {
			frameStart := time.Now()
			select {
			case b := <-bodies:
				if err := s.reset(b); err != nil {
					log.Printf("body rejected, keeping the previous one: %v", err)
				}
			default:
			}
			if *speed > 0 {
				owed += *speed * frameDt.Seconds() / controlDt
				for ; owed >= 1; owed-- {
					s.control()
					if time.Since(frameStart) > frameDt {
						owed = 0 // CPU-bound: run as fast as it can, don't build a backlog
						break
					}
				}
			} else {
				for time.Since(frameStart) < frameDt {
					s.control()
				}
			}
			if data, err := json.Marshal(s.frame()); err != nil {
				log.Printf("marshal frame: %v", err)
			} else {
				latest.Store(&data)
			}
			if *speed > 0 {
				<-tick.C
			}
		}
	}()

	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, &websocket.AcceptOptions{OriginPatterns: []string{"localhost:*", "127.0.0.1:*"}})
		if err != nil {
			return
		}
		defer c.CloseNow()
		cctx := c.CloseRead(r.Context())
		tick := time.NewTicker(frameDt)
		defer tick.Stop()
		var sent *[]byte
		for {
			select {
			case <-cctx.Done():
				return
			case <-tick.C:
			}
			if d := latest.Load(); d != nil && d != sent {
				if err := c.Write(cctx, websocket.MessageText, *d); err != nil {
					return
				}
				sent = d
			}
		}
	})

	// The LLM loop over HTTP: read what the robot perceives, reply with a
	// command. Only applied with -commander external.
	http.HandleFunc("/perception", func(w http.ResponseWriter, r *http.Request) {
		link.mu.Lock()
		body := struct {
			Perception command.Perception `json:"perception"`
			Prompt     string             `json:"prompt"`
		}{link.perception, link.prompt}
		link.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(body)
	})
	http.HandleFunc("/command", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST a JSON command", http.StatusMethodNotAllowed)
			return
		}
		if *commander != "external" {
			http.Error(w, "the scripted commander is in charge; restart with -commander external", http.StatusConflict)
			return
		}
		data, err := io.ReadAll(io.LimitReader(r.Body, 4096))
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		c, err := command.Parse(string(data))
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		link.mu.Lock()
		link.posted = &c
		link.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(c)
	})

	log.Printf("commander: %s; LLM loop at http://%s/perception and POST http://%s/command", *commander, *addr, *addr)
	log.Printf("streaming on ws://%s/ws", *addr)
	log.Fatal(http.ListenAndServe(*addr, nil))
}
