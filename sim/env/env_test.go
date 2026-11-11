package env

import (
	"math"
	"path/filepath"
	"testing"

	"mars/sim/body"
)

func biped(t *testing.T) body.Body {
	t.Helper()
	b, err := body.Load("../bodies/biped.json")
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestScenariosLoad(t *testing.T) {
	for _, n := range ScenarioNames() {
		s, err := LoadScenario(n)
		if err != nil {
			t.Fatal(err)
		}
		if err := s.Validate(); err != nil {
			t.Errorf("built-in %s: %v", n, err)
		}
	}
	files, _ := filepath.Glob("../scenarios/*.json")
	if len(files) == 0 {
		t.Fatal("no scenario files found")
	}
	for _, f := range files {
		s, err := LoadScenario(f)
		if err != nil {
			t.Errorf("%s: %v", f, err)
		}
		if f == "../scenarios/getup-then-walk.json" && (!s.Walk || s.LyingFrac != 1 || s.EndOnFall) {
			t.Errorf("%s: base fields not inherited or overrides not applied: %+v", f, s)
		}
	}
	if _, err := LoadScenario("swim"); err == nil {
		t.Error("unknown scenario accepted")
	}
}

// Starting on the floor counts as down, lying still pays less than any
// step spent up, and a fall does not end a getup episode.
func TestGetUpEpisode(t *testing.T) {
	sc, _ := LoadScenario("getup")
	sc.RandomBodyFrac = 0
	e := New(biped(t), sc, 3)
	if !e.StartedLying || e.up {
		t.Fatalf("getup episode should start lying down: lying=%v up=%v", e.StartedLying, e.up)
	}
	still := make([]float64, len(e.W.Cmd))
	best := 0.0
	for i := 0; i < 100; i++ { // 1 s lying with valves closed
		r, terminal, _ := e.Step(still)
		if terminal {
			t.Fatal("episode ended while lying on the floor")
		}
		if i > 50 {
			best = math.Max(best, r)
		}
	}
	if best >= 0.2 {
		t.Errorf("lying still earns %.3f per step; being up must pay more (≥ 0.2)", best)
	}
	t.Logf("lying still: best reward %.3f per step, pelvis %.2f m", best, e.W.Base.P[1])

	st, _ := LoadScenario("stand")
	st.RandomBodyFrac, st.PushEvery = 0, 0
	u := New(biped(t), st, 4)
	r, terminal, _ := u.Step(still)
	if terminal || !u.up || r < 0.2 {
		t.Fatalf("standing still: reward %.3f, up %v, terminal %v", r, u.up, terminal)
	}
}
