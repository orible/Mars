package env

import (
	"mars/sim/model"
	"mars/sim/terrain"
	"mars/sim/world"
)

// ApplyTerrain builds spec (seed is used if the spec has none) and puts it
// under w, keeping scattered objects clear of the objectives and the
// start, then stands the robot back on the new floor at the origin in its
// current pose. If the spec has a Goal, it returns that as the only
// objective; otherwise objectives come back unchanged.
func ApplyTerrain(w *model.World, spec terrain.Spec, seed int64, objectives []world.XZ) ([]world.XZ, error) {
	rs, err := spec.Resolve()
	if err != nil {
		return nil, err
	}
	if rs.Goal != [2]float64{} {
		objectives = []world.XZ{world.XZ(rs.Goal)}
	}
	keep := make([][2]float64, len(objectives))
	for i, o := range objectives {
		keep[i] = o
	}
	t, err := terrain.Build(spec, seed, keep)
	if err != nil {
		return nil, err
	}
	w.Ground.Terrain = t
	if w.Floating {
		w.Place(w.Base.R, w.Q, 0, 0)
	}
	return objectives, nil
}

// testWorld is the base for the terrain test worlds: walk where told on
// the base body, no pushes, no harness, no random floor bumps, ended by a
// fall. Walking these worlds is a later test; nothing trains on them yet.
func testWorld(desc string, t terrain.Spec) Scenario {
	return Scenario{
		Description: desc,
		Walk:        true,
		World:       world.RandomConfig{Count: 1, Distance: [2]float64{4, 4}, Friction: [2]float64{0.8, 0.8}, Arrive: 0.4},
		EndOnFall:   true, EpisodeSeconds: 30,
		Terrain: &t,
	}
}

func ladder(kind string, level float64) terrain.Spec { return terrain.Spec{Ladder: kind, Level: level} }

// The terrain test worlds, built in. Each is also a ladder at a fixed
// level; a scenario file can set "terrain": {"ladder": …, "level": …,
// "scale": …} to sweep it (see sim/scenarios).
func init() {
	worlds := map[string]Scenario{
		"stairs-up":    testWorld("climb 6 steps of 17 cm rise and 28 cm run to a landing (terrain test world)", ladder("stairs-up", 0.8)),
		"stairs-down":  testWorld("start on a platform and go down 6 steps of 17 cm (terrain test world)", ladder("stairs-down", 0.8)),
		"stairs-over":  testWorld("up 4 steps, across a landing and down 4 (terrain test world)", ladder("stairs-over", 0.7)),
		"slope-10deg":  testWorld("walk up a 10° slope (terrain test world)", terrain.Spec{Slope: &terrain.Slope{AngleDeg: 10}, Goal: [2]float64{4, 0}}),
		"ramp":         testWorld("a 10° ramp up to a landing 40 cm high and down again (terrain test world)", terrain.Spec{Ramps: []terrain.Ramp{{At: [2]float64{1, 0}, AngleDeg: 10, Rise: 0.4, Landing: 1.5}}, Goal: [2]float64{7.5, 0}}),
		"field-gentle": testWorld("gently uneven octave-noise ground, ±2 cm (terrain test world)", terrain.Spec{NoisePreset: "gentle", Goal: [2]float64{4, 0}}),
		"field-rough":  testWorld("a rough field of octave noise, ±12 cm (terrain test world)", terrain.Spec{NoisePreset: "rough", Goal: [2]float64{4, 0}}),
		"cluttered-room": testWorld("a floor scattered with 16 boxes and cylinders to walk between and look at (terrain test world)", terrain.Spec{
			Scatter: &terrain.Scatter{Count: 16, Min: [2]float64{-3, -3}, Max: [2]float64{6, 3}, Size: [2]float64{0.15, 0.6}, Height: [2]float64{0.05, 0.9}, Clear: 0.7, Gap: 0.2},
			Goal:    [2]float64{5, 0}}),
		"course": testWorld("a mixed course on uneven ground: a ramp, stairs up and over, then a cluttered stretch (terrain test world)", terrain.Spec{
			NoisePreset: "gentle",
			Ramps:       []terrain.Ramp{{At: [2]float64{1.5, 0}, AngleDeg: 8, Rise: 0.25, Landing: 1}},
			Stairs:      []terrain.Stairs{{At: [2]float64{7, 0}, Rise: 0.15, Run: 0.3, Steps: 4, Width: 1.6, Landing: 1, Over: true}},
			Scatter:     &terrain.Scatter{Count: 10, Min: [2]float64{11, -2.5}, Max: [2]float64{16, 2.5}, Size: [2]float64{0.15, 0.5}, Height: [2]float64{0.05, 0.6}, Clear: 0.7, Gap: 0.3},
			Objects:     []terrain.Object{{Kind: "box", At: [2]float64{3, 2}, Size: [3]float64{0.6, 0.8, 0.6}}, {Kind: "cylinder", At: [2]float64{3, -2}, Size: [3]float64{0.4, 1.2, 0.4}}},
			Goal:        [2]float64{17, 0},
		}),
	}
	for name, sc := range worlds {
		Scenarios[name] = sc
	}
}
