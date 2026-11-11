package env

import (
	"math"
	"math/rand"
	"path/filepath"
	"strings"
	"testing"

	"mars/sim/body"
	"mars/sim/geom"
	"mars/sim/model"
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

// standUpController is a hand-written sagittal balance controller, used
// only to prove the body can stand up from a squat: knees straighten
// slowly, hips bring the pelvis upright, ankles keep the centre of gravity
// over the feet, hip roll holds zero. It returns one valve command per
// cylinder.
func standUpController(w *model.World, feet []int) []float64 {
	m := w.M
	yaw := math.Atan2(-w.Base.R[2][0], w.Base.R[0][0])
	fwd := geom.V3{math.Cos(yaw), 0, -math.Sin(yaw)}
	cog, _ := w.CoG()
	var fx float64
	for _, c := range feet {
		fx += m.Point(w.Pose, m.Contacts[c].Link, m.Contacts[c].Point).Dot(fwd)
	}
	ahead := cog.Dot(fwd) - fx/float64(len(feet)) // CoG ahead of the middle of the feet
	pitch := math.Asin(math.Max(-1, math.Min(1, w.Base.R[1][0])))
	vd := make([]float64, m.NDof()) // desired joint rates
	for q := range vd {
		switch name := m.Names[q+1]; {
		case strings.HasPrefix(name, "shin"):
			vd[q] = -0.8 * w.Q[q]
		case strings.HasPrefix(name, "thigh"):
			vd[q] = 3 * pitch
		case strings.HasPrefix(name, "foot"):
			vd[q] = -10 * ahead
		default:
			vd[q] = -2 * w.Q[q]
		}
	}
	cmd := make([]float64, len(m.Cyl))
	for k, c := range m.Cyl {
		ldot := 0.0
		for n, j := range c.Span {
			ldot += w.DLdq[k][n] * vd[j]
		}
		cmd[k] = math.Max(-1, math.Min(1, ldot/0.03))
	}
	return cmd
}

// Strength check for getting up: from the easiest floor start (a squat
// balanced over flat feet) a simple controller must straighten the knees
// against the body's weight and lift the pelvis to standing height. Staying
// balanced on the way up is left to training: with the hip at its 90° limit
// and the ankle at its 20° dorsiflexion limit in the squat, there is little
// range left to keep the weight over the feet, and this hand controller
// sits back down after rising (see docs/architecture.md).
func TestSquatToStandStrength(t *testing.T) {
	sc, _ := LoadScenario("getup")
	sc.RandomBodyFrac = 0
	e := New(biped(t), sc, 1)
	w := e.W
	floorPose(w, rand.New(rand.NewSource(2)), 0)
	t.Logf("squat pose q %v, pelvis up %v", w.Q, w.Up())
	for i := 0; i < int(0.3/PhysicsDt); i++ {
		w.Step(PhysicsDt)
	}
	var feet []int
	for _, f := range e.feet {
		feet = append(feet, f.contacts...)
	}
	startH, startUp := w.Base.P[1], w.Upright()
	best := 0.0
	for i := 0; i < int(4/PhysicsDt); i++ {
		if i%Substeps == 0 { // 1 kHz
			copy(w.Cmd, standUpController(w, feet))
		}
		w.Step(PhysicsDt)
		best = math.Max(best, w.Base.P[1]/w.StandH())
	}
	knees := []float64{}
	for q := range w.Q {
		if strings.HasPrefix(w.M.Names[q+1], "shin") {
			knees = append(knees, math.Round(w.Q[q]*180/math.Pi))
		}
	}
	t.Logf("squat start: pelvis %.2f m, upright %v; after 4 s: pelvis %.2f m (%.0f%% of standing, best %.0f%%), up %.2f, knees %v deg",
		startH, startUp, w.Base.P[1], 100*w.Base.P[1]/w.StandH(), 100*best, w.Up()[1], knees)
	if startUp {
		t.Errorf("the squat start should count as down")
	}
	if best < 0.86 {
		t.Errorf("the legs could only lift the pelvis to %.0f%% of standing height", 100*best)
	}
	for _, k := range knees {
		if k < -10 {
			t.Errorf("knee only straightened to %.0f deg under load", k)
		}
	}
}

// The squat-to-stand reference should read standing as its end, and the
// balanced squat start (trunk leaning ~45°, hips at their 90° limit) as
// the human at about seat-off, with the extension phase still to come.
func TestSquatToStandReference(t *testing.T) {
	ref, err := LoadReference("squat-to-stand")
	if err != nil {
		t.Fatal(err)
	}
	sc, _ := LoadScenario("getup")
	sc.RandomBodyFrac = 0
	e := New(biped(t), sc, 1)
	tr := NewMotionTracker(ref, e.W.M)
	if tr == nil || len(tr.legs) != 2 {
		t.Fatalf("expected two legs to track, got %v", tr)
	}
	stand := model.NewWorld(e.W.M, true)
	tr.Update(stand)
	t.Logf("standing: progress %.2f, error %.1f deg", tr.Progress, tr.Error*180/math.Pi)
	if tr.Progress < 0.95 || tr.Error > 0.05 {
		t.Errorf("standing should be the end of the reference")
	}
	floorPose(e.W, rand.New(rand.NewSource(2)), 0)
	tr.Update(e.W)
	t.Logf("squat start: progress %.2f, error %.1f deg", tr.Progress, tr.Error*180/math.Pi)
	if tr.Progress < 0.3 || tr.Progress > 0.8 || tr.Error > 20*math.Pi/180 {
		t.Errorf("the squat start should match the human around seat-off, got progress %.2f, error %.1f deg", tr.Progress, tr.Error*180/math.Pi)
	}
}

// Every built-in reference loads and retargets to the biped, standing is
// the end of each, and each floor start picks a reference that fits it.
func TestRiseReferences(t *testing.T) {
	refs, err := LoadReferences(allRises)
	if err != nil {
		t.Fatal(err)
	}
	if len(refs) != len(ReferenceNames()) {
		t.Errorf("allRises lists %d references, %d are built in", len(refs), len(ReferenceNames()))
	}
	sc, _ := LoadScenario("getup")
	sc.RandomBodyFrac = 0
	e := New(biped(t), sc, 1)
	stand := model.NewWorld(e.W.M, true)
	for _, r := range refs {
		tr := NewMotionTracker(r, e.W.M)
		if tr == nil {
			t.Fatalf("%s: cannot retarget to the biped", r.Name)
		}
		tr.Update(stand)
		if tr.Progress < 0.9 || tr.Error > 10*math.Pi/180 {
			t.Errorf("%s: standing reads as %.0f%% along, %.0f deg off; want the end", r.Name, 100*tr.Progress, tr.Error*180/math.Pi)
		}
	}
	for _, c := range []struct {
		name string
		d    float64
	}{{"squat", 0}, {"kneel", 0.3}, {"sit", 0.6}, {"lying", 0.9}} {
		w := model.NewWorld(e.W.M, true)
		floorPose(w, rand.New(rand.NewSource(5)), c.d)
		for i := 0; i < int(0.3/PhysicsDt); i++ {
			w.Step(PhysicsDt)
		}
		tr := PickTracker(refs, w)
		t.Logf("%s start tracks %q: %.0f%% along, %.0f deg off", c.name, tr.Ref.Name, 100*tr.Progress, tr.Error*180/math.Pi)
		if tr.Progress > 0.9 {
			t.Errorf("%s start should not read as nearly standing", c.name)
		}
	}
}

// Each rung of the floor-start ladder should settle at rest without
// blowing up; squatting, kneeling, sitting and lying all count as down.
func TestFloorStartLadder(t *testing.T) {
	sc, _ := LoadScenario("getup")
	sc.RandomBodyFrac = 0
	for _, c := range []struct {
		name string
		d    float64 // difficulty
	}{{"squat", 0.0}, {"deepest squat", 0.24}, {"kneel", 0.3}, {"sit", 0.6}, {"lying", 0.9}} {
		e := New(biped(t), sc, 1)
		floorPose(e.W, rand.New(rand.NewSource(2)), c.d)
		for i := 0; i < 50*PolicyEvery*Substeps; i++ { // 0.5 s, valves closed
			e.W.Step(PhysicsDt)
		}
		if !e.W.Healthy() {
			t.Fatalf("%s: diverged", c.name)
		}
		t.Logf("%s: pelvis %.2f m (stand %.2f), up %.2f, upright %v", c.name, e.W.Base.P[1], e.W.StandH(), e.W.Up()[1], e.W.Upright())
		if e.W.Upright() {
			t.Errorf("%s should start down", c.name)
		}
	}
}

// Starting on the floor counts as down, lying still pays less than any
// step spent up, and a fall does not end a getup episode.
func TestGetUpEpisode(t *testing.T) {
	sc, _ := LoadScenario("getup")
	sc.RandomBodyFrac = 0
	e := New(biped(t), sc, 3)
	if !e.StartedLying || e.up {
		t.Fatalf("getup episode should start down: lying=%v up=%v; pelvis %.3f of %.3f m, up axis %v, q %v",
			e.StartedLying, e.up, e.W.Base.P[1], e.h0, e.W.Up(), e.W.Q)
	}
	still := make([]float64, len(e.W.Cmd))
	best := 0.0
	for i := 0; i < 100; i++ { // 1 s on the floor with valves closed
		r, terminal, _ := e.Step(still)
		if terminal {
			t.Fatal("episode ended while on the floor")
		}
		if i > 50 {
			best = math.Max(best, r)
		}
	}
	if best >= 0.2 {
		t.Errorf("staying down earns %.3f per step; being up must pay more (at least 0.2)", best)
	}
	t.Logf("staying down: best reward %.3f per step, pelvis %.2f m", best, e.W.Base.P[1])

	st, _ := LoadScenario("stand")
	st.RandomBodyFrac, st.PushEvery = 0, 0
	u := New(biped(t), st, 4)
	r, terminal, _ := u.Step(still)
	if terminal || !u.up || r < 0.2 {
		t.Fatalf("standing still: reward %.3f, up %v, terminal %v", r, u.up, terminal)
	}
}
