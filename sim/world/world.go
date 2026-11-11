// Package world describes the environment the robot walks in: the ground
// and a sequence of objectives. Like a body spec, a world is a data file;
// the trainer also generates random ones.
//
// The robot never sees world coordinates or a map. It is only told which
// way the current objective lies and how near it is (see obs.Goal), and
// walks that way until it arrives; then the next objective takes over.
//
// Stage 1 is a flat floor. Obstacles, terrain and friction regions come
// next and will be added here.
package world

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
)

// XZ is a point on the floor: x forward from the start, z to the right.
type XZ [2]float64

// World is one environment.
type World struct {
	Name     string  `json:"name"`
	Friction float64 `json:"friction"` // floor Coulomb coefficient
	// Objectives are visited in order, starting from the origin facing +x.
	Objectives []XZ `json:"objectives"`
	// ArriveRadius is how close counts as arrived (m).
	ArriveRadius float64 `json:"arriveRadius"`
}

func (w World) Validate() error {
	switch {
	case w.Friction <= 0:
		return errors.New("friction must be positive")
	case len(w.Objectives) == 0:
		return errors.New("world needs at least one objective")
	case w.ArriveRadius <= 0:
		return errors.New("arriveRadius must be positive")
	}
	for i, o := range w.Objectives {
		if math.IsNaN(o[0]) || math.IsNaN(o[1]) {
			return fmt.Errorf("objective %d is not a number", i)
		}
	}
	return nil
}

// Load reads and validates a world file.
func Load(path string) (World, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return World{}, err
	}
	var w World
	if err := json.Unmarshal(data, &w); err != nil {
		return World{}, fmt.Errorf("%s: %w", path, err)
	}
	if err := w.Validate(); err != nil {
		return World{}, fmt.Errorf("%s: %w", path, err)
	}
	return w, nil
}

// RandomConfig shapes generated worlds.
type RandomConfig struct {
	Count    int        `json:"count"`    // objectives per world
	Distance [2]float64 `json:"distance"` // m between consecutive objectives
	MaxTurn  float64    `json:"maxTurn"`  // rad, largest change of direction between legs (π = anywhere)
	Friction [2]float64 `json:"friction"`
	Arrive   float64    `json:"arrive"`
}

// DefaultRandom starts easy: short legs, mostly ahead.
func DefaultRandom() RandomConfig {
	return RandomConfig{Count: 6, Distance: [2]float64{1.5, 5}, MaxTurn: math.Pi, Friction: [2]float64{0.5, 1.1}, Arrive: 0.4}
}

// Random generates a flat world with a chain of objectives.
func Random(rng *rand.Rand, c RandomConfig) World {
	u := func(lo, hi float64) float64 { return lo + rng.Float64()*(hi-lo) }
	w := World{Name: "random", Friction: u(c.Friction[0], c.Friction[1]), ArriveRadius: c.Arrive}
	var p XZ
	dir := u(-c.MaxTurn, c.MaxTurn)
	for i := 0; i < c.Count; i++ {
		if i > 0 {
			dir += u(-c.MaxTurn, c.MaxTurn)
		}
		d := u(c.Distance[0], c.Distance[1])
		p = XZ{p[0] + d*math.Cos(dir), p[1] + d*math.Sin(dir)}
		w.Objectives = append(w.Objectives, p)
	}
	return w
}
