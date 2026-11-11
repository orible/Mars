// Package command is the interface between the LLM loop and the movement
// network.
//
// The movement network has no idea of objectives, maps or paths. It only
// follows the current command: walk in this direction (relative to where
// it is facing) at this speed, or stop. Deciding the command is the job of
// the fast LLM layer, which runs an input/output loop at its own slow pace:
//
//	perception (text) ──► LLM ──► command (JSON) ──► movement network
//	      ▲                                                 │
//	      └───────────── the robot moves, the view changes ─┘
//
// In simulation the perception is computed from the world. A Scripted
// commander stands in for the LLM during training, with an LLM-like
// update period, latency and misjudged angles and distances, so the
// network learns to follow a slow, imperfect boss. The sim server exposes
// the same loop over HTTP so a real LLM can take over.
package command

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
)

// Command is what the movement network is told.
type Command struct {
	// HeadingDeg is the direction to walk, relative to where the robot
	// faces: 0 straight ahead, positive to the right, negative to the left.
	HeadingDeg float64 `json:"heading_deg"`
	// Speed in m/s; 0 means stand still.
	Speed float64 `json:"speed"`
}

// MaxSpeed bounds commanded speed (m/s).
const MaxSpeed = 0.5

// Stop is the zero command.
var Stop = Command{}

// Clamp limits speed to [0, MaxSpeed] and wraps the heading to ±180°.
func (c Command) Clamp() Command {
	c.Speed = math.Max(0, math.Min(MaxSpeed, c.Speed))
	c.HeadingDeg = math.Remainder(c.HeadingDeg, 360)
	return c
}

// Parse reads an LLM's JSON reply.
func Parse(s string) (Command, error) {
	var c Command
	if err := json.Unmarshal([]byte(s), &c); err != nil {
		return Stop, fmt.Errorf("command: %w", err)
	}
	if math.IsNaN(c.HeadingDeg) || math.IsNaN(c.Speed) {
		return Stop, fmt.Errorf("command: not a number")
	}
	return c.Clamp(), nil
}

// Perception is what the LLM is told about the objective. Values are
// estimates, as a person would judge them by eye.
type Perception struct {
	Visible     bool    `json:"visible"`
	DistanceM   float64 `json:"distance_m"`
	BearingDeg  float64 `json:"bearing_deg"` // relative to facing, positive right
	Arrived     bool    `json:"arrived"`
	Fallen      bool    `json:"fallen"`
	Objectives  int     `json:"objectives_left"`
	ElapsedSecs float64 `json:"elapsed_s"`
}

// Prompt renders the perception as the LLM's input text.
func (p Perception) Prompt() string {
	switch {
	case p.Fallen:
		return "You have fallen over. Reply with a command to stand still while you are reset."
	case p.Objectives == 0:
		return "There are no objectives left. Reply with a command to stand still."
	case p.Arrived:
		return "You have reached the objective. Reply with a command to stand still."
	case !p.Visible:
		return "You cannot see the objective. Reply with a command."
	}
	side := "right"
	if p.BearingDeg < 0 {
		side = "left"
	}
	return fmt.Sprintf("The objective is about %.1f m away, %.0f° to your %s. Reply with a JSON command {\"heading_deg\": <positive right>, \"speed\": <0 to %.1f m/s>} to walk towards it; slow down as it grows near.",
		p.DistanceM, math.Abs(p.BearingDeg), side, MaxSpeed)
}

// Commander turns perceptions into commands.
type Commander interface {
	Decide(p Perception) Command
}

// Scripted is the training stand-in for the LLM: aim at the objective as
// perceived, at a speed that falls off as it grows near.
type Scripted struct {
	Cruise   float64 // m/s far from the objective
	SlowDown float64 // m: start slowing within this distance
}

func (s Scripted) Decide(p Perception) Command {
	if p.Fallen || p.Arrived || !p.Visible || p.Objectives == 0 {
		return Stop
	}
	speed := s.Cruise * math.Min(1, p.DistanceM/s.SlowDown)
	return Command{HeadingDeg: p.BearingDeg, Speed: speed}.Clamp()
}

// Loop runs a Commander at an LLM's pace: it looks every Period seconds
// (with jitter), its estimates carry noise, and its command takes Latency
// seconds to arrive.
type Loop struct {
	Commander  Commander
	Period     [2]float64 // s, range of time between looks
	Latency    [2]float64 // s, range of think time
	AngleNoise float64    // deg, s.d. of bearing misjudgement
	DistNoise  float64    // fraction, s.d. of distance misjudgement

	rng      *rand.Rand
	current  Command
	pending  Command
	arriveAt float64 // when pending takes effect (-1 none)
	nextLook float64
}

// NewLoop returns a loop with LLM-like timing around commander c.
func NewLoop(c Commander, rng *rand.Rand) *Loop {
	return &Loop{Commander: c, Period: [2]float64{0.5, 1.5}, Latency: [2]float64{0.2, 0.8},
		AngleNoise: 8, DistNoise: 0.15, rng: rng, arriveAt: -1}
}

// Reset clears the loop for a new episode at time t.
func (l *Loop) Reset(t float64) {
	l.current, l.pending, l.arriveAt, l.nextLook = Stop, Stop, -1, t
}

// Update advances the loop to time t. truth is the exact perception; the
// commander sees a misjudged copy. It returns the command in force.
func (l *Loop) Update(t float64, truth Perception) Command {
	if l.arriveAt >= 0 && t >= l.arriveAt {
		l.current, l.arriveAt = l.pending, -1
	}
	if t >= l.nextLook {
		p := truth
		p.BearingDeg += l.rng.NormFloat64() * l.AngleNoise
		p.DistanceM *= math.Max(0.2, 1+l.rng.NormFloat64()*l.DistNoise)
		l.pending = l.Commander.Decide(p)
		l.arriveAt = t + l.uniform(l.Latency)
		l.nextLook = t + l.uniform(l.Period)
	}
	return l.current
}

func (l *Loop) uniform(r [2]float64) float64 { return r[0] + l.rng.Float64()*(r[1]-r[0]) }
