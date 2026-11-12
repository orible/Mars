package env

import (
	"math"
	"path/filepath"
	"reflect"
	"testing"

	"mars/sim/terrain"
)

// terrainWorlds lists the built-in terrain worlds and the scenario files
// with a terrain.
func terrainWorlds() []string {
	var out []string
	for _, n := range ScenarioNames() {
		if Scenarios[n].Terrain != nil {
			out = append(out, n)
		}
	}
	files, _ := filepath.Glob("../scenarios/*.json")
	for _, f := range files {
		if s, err := LoadScenario(f); err == nil && s.Terrain != nil {
			out = append(out, f)
		}
	}
	return out
}

// Every terrain world starts with the robot standing on the floor (not
// in it or above it), its feet found, the objective at the end of the
// course, nothing standing on the start or the objective, and survives a
// second of the policy's zero action.
func TestTerrainWorldsStart(t *testing.T) {
	names := terrainWorlds()
	if len(names) < 14 {
		t.Fatalf("only %d terrain worlds: %v", len(names), names)
	}
	for _, n := range names {
		t.Run(filepath.Base(n), func(t *testing.T) {
			sc, _ := LoadScenario(n)
			e := New(biped(t), sc, 3)
			w := e.W
			tr := w.Ground.Terrain
			if tr == nil {
				t.Fatal("no terrain under the robot")
			}
			lo := math.Inf(1)
			for _, c := range w.M.Contacts {
				d, _ := tr.Distance(w.M.Point(w.Pose, c.Link, c.Point), 1, 0, 0, 0)
				lo = math.Min(lo, d)
			}
			if lo < -0.001 || lo > 0.02 {
				t.Errorf("lowest contact %.1f mm from the floor", 1000*lo)
			}
			if !w.Upright() || len(e.feet) < 2 {
				t.Errorf("start: upright %v, %d feet", w.Upright(), len(e.feet))
			}
			goal := tr.Spec.Goal
			if obj := e.Tracker.W.Objectives; len(obj) != 1 || obj[0][0] != goal[0] || obj[0][1] != goal[1] {
				t.Errorf("objectives %v, want the goal %v", obj, goal)
			}
			for _, p := range tr.Primitives() {
				if p.Role != "object" {
					continue
				}
				for _, k := range [][2]float64{{0, 0}, goal} {
					if p.FootprintDistance(k[0], k[1]) < 0.5 {
						t.Errorf("object at (%.2f, %.2f) stands on %v", p.Center[0], p.Center[2], k)
					}
				}
			}
			act := make([]float64, len(w.M.Cyl))
			for i := 0; i < 100; i++ {
				if _, term, _ := e.Step(act); term && !w.Healthy() {
					t.Fatal("diverged")
				}
			}
		})
	}
}

// The same seed builds the same world and runs the same episode; another
// seed scatters a different one.
func TestTerrainWorldDeterminism(t *testing.T) {
	sc, _ := LoadScenario("cluttered-room")
	run := func(seed int64) ([]terrain.Primitive, [3]float64) {
		e := New(biped(t), sc, seed)
		act := make([]float64, len(e.W.M.Cyl))
		for i := 0; i < 20; i++ {
			e.Step(act)
		}
		return e.W.Ground.Terrain.Primitives(), e.W.Base.P
	}
	pa, xa := run(11)
	pb, xb := run(11)
	pc, _ := run(12)
	if !reflect.DeepEqual(pa, pb) || xa != xb {
		t.Error("the same seed gave a different world or episode")
	}
	if reflect.DeepEqual(pa, pc) {
		t.Error("another seed gave the same objects")
	}
}

// A scenario without terrain keeps the plain floor, judged from y = 0.
func TestNoTerrainLeavesFloorAlone(t *testing.T) {
	sc, _ := LoadScenario("balance")
	e := New(biped(t), sc, 5)
	if e.W.Ground.Terrain != nil || e.W.GroundUnder() != 0 {
		t.Error("a plain scenario got a terrain")
	}
}

// Loading a file that changes a built-in's terrain leaves the built-in as
// it was.
func TestTerrainFileDoesNotChangeBuiltIn(t *testing.T) {
	before := cloneTerrain(*Scenarios["stairs-up"].Terrain)
	sc, err := LoadScenario("../scenarios/stairs-sweep.json")
	if err != nil {
		t.Fatal(err)
	}
	if sc.Terrain.Ladder != "stairs-up" || sc.Terrain.Level != 0.4 {
		t.Errorf("file terrain %+v, want the stairs-up ladder at 0.4", *sc.Terrain)
	}
	if after := *Scenarios["stairs-up"].Terrain; !reflect.DeepEqual(before, after) {
		t.Errorf("built-in changed from %+v to %+v", before, after)
	}
}
