package env

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"mars/sim/world"
)

// Scenario is what a training run practises: how episodes start, what the
// command loop asks for, what ends an episode, and the randomisation. Use
// a built-in by name, or write one as a JSON file (see sim/scenarios).
// Every scenario trains the same network, so skills learned in one carry
// into the next: getup → stand → walk → life.
type Scenario struct {
	Name        string `json:"name"`
	Description string `json:"description"`

	// Walk: the scripted LLM stand-in issues walking commands towards
	// random objectives. Otherwise the command is always "stand still".
	Walk  bool               `json:"walk"`
	World world.RandomConfig `json:"world"`

	LyingFrac float64 `json:"lyingFrac"` // fraction of episodes that start on the floor
	EndOnFall bool    `json:"endOnFall"` // a fall ends the episode; otherwise it must get back up
	// StartLevel is how far down floor starts go: 0 crouched … 1 lying
	// flat (see FloorStart). With Curriculum the trainer moves it up as
	// get-ups succeed and back down if they stop.
	StartLevel float64 `json:"startLevel"`
	Curriculum bool    `json:"curriculum"`
	// Reference lists human motions to imitate while getting up
	// (comma-separated built-in names from env/references, or .json
	// files); each episode tracks the one its start pose fits best. ""
	// for none.
	Reference string `json:"reference"`

	RandomBodyFrac float64    `json:"randomBodyFrac"` // fraction of episodes on a random body instead of the base spec
	EpisodeSeconds float64    `json:"episodeSeconds"`
	PushEvery      float64    `json:"pushEvery"` // mean seconds between pushes while up (0 = none)
	PushSpeed      [2]float64 `json:"pushSpeed"` // m/s range of the sideways velocity change a push gives

	// Assist is the harness strength (0–1) for episodes that start
	// standing: the pelvis is held up and carried along the command, as a
	// baby is held while it kicks. The trainer fades it as falls drop.
	Assist float64 `json:"assist"`
	// UnassistedFrac of starts never get the harness, so real ability is
	// always being practised and measured.
	UnassistedFrac float64 `json:"unassistedFrac"`
	// AssistFloor also holds floor starts: the harness takes part of the
	// weight and helps the pelvis upright while the legs learn to push,
	// and stays on through falls. The trainer fades it as held get-ups
	// succeed.
	AssistFloor bool `json:"assistFloor"`
}

// allRises are the human getting-up references: every built-in one (one
// from the sit-to-stand literature, the rest from CMU motion capture).
const allRises = "all"

func flatWorld() world.RandomConfig {
	return world.RandomConfig{Count: 1, Distance: [2]float64{1, 1}, Friction: [2]float64{0.5, 1.1}, Arrive: 0.4}
}

// Scenarios are the built-ins.
var Scenarios = map[string]Scenario{
	"getup": {
		Description: "start on the floor and get up, then stay up; the kicking and pushing a baby learns on the floor. Starts crouched and works down through kneeling and sitting to lying flat as get-ups succeed",
		World:       flatWorld(), LyingFrac: 1, EndOnFall: false, StartLevel: 0.1, Curriculum: true, Reference: allRises,
		Assist: 1, UnassistedFrac: 0.3, AssistFloor: true,
		RandomBodyFrac: 0.5, EpisodeSeconds: 20,
	},
	"stand": {
		Description: "stand still on the stop command through random pushes",
		World:       flatWorld(), EndOnFall: true,
		RandomBodyFrac: 0.5, EpisodeSeconds: 6, PushEvery: 2, PushSpeed: [2]float64{0.1, 0.4},
	},
	"walk": {
		Description: "follow the LLM stand-in's walking commands towards random objectives, held by a fading harness at first",
		Walk:        true, World: world.DefaultRandom(), EndOnFall: true,
		RandomBodyFrac: 0.5, EpisodeSeconds: 12, PushEvery: 4, PushSpeed: [2]float64{0.1, 0.3},
		Assist: 1, UnassistedFrac: 0.2,
	},
	"life": {
		Description: "everything at once: start on the floor or standing, get up, walk where told, get pushed, fall and get back up",
		Walk:        true, World: world.DefaultRandom(), LyingFrac: 0.4, EndOnFall: false, StartLevel: 1, Reference: allRises,
		RandomBodyFrac: 0.5, EpisodeSeconds: 15, PushEvery: 4, PushSpeed: [2]float64{0.1, 0.3},
		Assist: 0.5, UnassistedFrac: 0.3,
	},
}

// TrainingStatus is what a training run is practising right now. The
// trainer writes it next to the checkpoint every iteration; the sim server
// reads it to follow the training course live.
type TrainingStatus struct {
	Scenario     string  `json:"scenario"`
	Description  string  `json:"description"`
	File         string  `json:"file,omitempty"` // scenario file, if not a built-in
	Iter         int     `json:"iter"`
	Elapsed      string  `json:"elapsed"`
	StartLevel   float64 `json:"startLevel"`
	Assist       float64 `json:"assist"`
	Episodes     int     `json:"episodes"`
	Falls        int     `json:"falls"`
	FloorStarts  int     `json:"floorStarts"`
	GotUp        int     `json:"gotUp"`
	HeldFloor    int     `json:"heldFloor"`
	HeldGotUp    int     `json:"heldGotUp"`
	SoloEpisodes int     `json:"soloEpisodes"`
	SoloFalls    int     `json:"soloFalls"`
	MeanReturn   float64 `json:"meanReturn"`
	// LR is the adaptive policy learning rate, restored on -resume.
	LR float64 `json:"lr,omitempty"`
}

// StatusPath is where the trainer writing checkpoint ckpt reports its
// status: checkpoints/policy.json → checkpoints/policy.status.json.
func StatusPath(ckpt string) string { return strings.TrimSuffix(ckpt, ".json") + ".status.json" }

// ScenarioInfo describes one scenario for pickers.
type ScenarioInfo struct {
	Name        string `json:"name"` // built-in name or file path
	Description string `json:"description"`
}

// ListScenarios returns the built-ins and the scenario files in dir.
func ListScenarios(dir string) []ScenarioInfo {
	var out []ScenarioInfo
	for _, n := range ScenarioNames() {
		out = append(out, ScenarioInfo{n, Scenarios[n].Description})
	}
	files, _ := filepath.Glob(filepath.Join(dir, "*.json"))
	for _, f := range files {
		if s, err := LoadScenario(f); err == nil {
			out = append(out, ScenarioInfo{filepath.ToSlash(f), s.Description})
		}
	}
	return out
}

func referenceOK(s string) bool { _, err := LoadReferences(s); return err == nil }

// ScenarioNames lists the built-ins.
func ScenarioNames() []string {
	var n []string
	for k := range Scenarios {
		n = append(n, k)
	}
	sort.Strings(n)
	return n
}

// LoadScenario returns a built-in by name, or reads a JSON scenario file
// (any value ending in .json). A file may name a built-in in "base" and
// override only some fields.
func LoadScenario(s string) (Scenario, error) {
	if !strings.HasSuffix(s, ".json") {
		sc, ok := Scenarios[s]
		if !ok {
			return Scenario{}, fmt.Errorf("unknown scenario %q (built-ins: %s, or a .json file)", s, strings.Join(ScenarioNames(), ", "))
		}
		sc.Name = s
		return sc, nil
	}
	data, err := os.ReadFile(s)
	if err != nil {
		return Scenario{}, err
	}
	var base struct {
		Base string `json:"base"`
	}
	if err := json.Unmarshal(data, &base); err != nil {
		return Scenario{}, fmt.Errorf("%s: %w", s, err)
	}
	sc := Scenarios["stand"]
	if base.Base != "" {
		var ok bool
		if sc, ok = Scenarios[base.Base]; !ok {
			return Scenario{}, fmt.Errorf("%s: unknown base scenario %q", s, base.Base)
		}
	}
	if err := json.Unmarshal(data, &sc); err != nil {
		return Scenario{}, fmt.Errorf("%s: %w", s, err)
	}
	if sc.Name == "" {
		sc.Name = strings.TrimSuffix(s, ".json")
	}
	return sc, sc.Validate()
}

// Validate rejects scenarios the environment cannot run.
func (s Scenario) Validate() error {
	in01 := func(v float64) bool { return v >= 0 && v <= 1 }
	switch {
	case s.EpisodeSeconds <= 0:
		return fmt.Errorf("scenario %q: episodeSeconds must be positive", s.Name)
	case !in01(s.LyingFrac) || !in01(s.RandomBodyFrac) || !in01(s.Assist) || !in01(s.UnassistedFrac) || !in01(s.StartLevel):
		return fmt.Errorf("scenario %q: lyingFrac, randomBodyFrac, assist, unassistedFrac and startLevel must be within 0–1", s.Name)
	case s.World.Count < 1 || s.World.Arrive <= 0 || s.World.Friction[0] <= 0 || s.World.Friction[1] < s.World.Friction[0]:
		return fmt.Errorf("scenario %q: world needs count ≥ 1, arrive > 0 and a friction range", s.Name)
	case s.PushEvery < 0 || s.PushSpeed[1] < s.PushSpeed[0]:
		return fmt.Errorf("scenario %q: bad push settings", s.Name)
	case s.Reference != "" && !referenceOK(s.Reference):
		return fmt.Errorf("scenario %q: cannot load reference %q", s.Name, s.Reference)
	case math.IsNaN(s.EpisodeSeconds):
		return fmt.Errorf("scenario %q: episodeSeconds is not a number", s.Name)
	}
	return nil
}
