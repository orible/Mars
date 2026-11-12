package body

import (
	"math"
	"math/rand"
	"strings"
	"testing"

	"mars/sim/geom"
)

func twin(name string) string {
	switch {
	case strings.HasSuffix(name, "_l"):
		return strings.TrimSuffix(name, "_l") + "_r"
	case strings.HasSuffix(name, "_r"):
		return strings.TrimSuffix(name, "_r") + "_l"
	}
	return name
}

func mirror(v geom.V3) geom.V3 { return geom.V3{v[0], v[1], -v[2]} }

func near(a, b geom.V3) bool { return a.Sub(b).Norm() < 1e-12 }

// Training's symmetry loss assumes the body is its own mirror image, so
// every capsule must have a twin on the twin link (or on the same link,
// for the trunk).
func TestHumanCollidersMirror(t *testing.T) {
	for _, spine := range []int{0, 1, 2} {
		s := DefaultHuman()
		s.Spine = spine
		b := Human(s)
		n := 0
		for _, l := range b.Links {
			tw := b.Links[b.LinkIndex(twin(l.Name))]
			for _, c := range l.Colliders {
				n++
				found := false
				for _, d := range tw.Colliders {
					found = found || d.Radius == c.Radius &&
						(near(mirror(c.A), d.A) && near(mirror(c.B), d.B) || near(mirror(c.A), d.B) && near(mirror(c.B), d.A))
				}
				if !found {
					t.Errorf("spine %d: %s capsule %v has no mirror image on %s", spine, l.Name, c, tw.Name)
				}
			}
		}
		if want := 4 + 2*4 + 2*2; n != want {
			t.Errorf("spine %d: %d capsules, want trunk 4 + 2×(thigh, shin, heel, midfoot) + 2×(upper arm, forearm) = %d", spine, n, want)
		}
	}
}

// Random bodies must come out valid, colliders included.
func TestRandomHumansValidate(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for i := 0; i < 300; i++ {
		if err := Random(rng).Validate(); err != nil {
			t.Fatalf("random body %d: %v", i, err)
		}
	}
}

func TestSelfCollisionPairsSkipCollisionParents(t *testing.T) {
	b := Human(DefaultHuman())
	pairs := map[[2]string]bool{}
	for _, p := range b.SelfCollisionPairs() {
		pairs[[2]string{b.Links[p[0]].Name, b.Links[p[1]].Name}] = true
	}
	for _, p := range [][2]string{{"pelvis", "thigh_l"}, {"thigh_l", "shin_l"}, {"shin_l", "heel_l"}, {"heel_l", "midfoot_l"}, {"chest", "upper_arm_r"}, {"upper_arm_r", "forearm_r"}, {"pelvis", "abdomen"}} {
		if pairs[p] {
			t.Errorf("%s and %s meet at a joint and should not collide", p[0], p[1])
		}
	}
	for _, p := range [][2]string{{"thigh_l", "thigh_r"}, {"shin_l", "shin_r"}, {"heel_l", "heel_r"}, {"pelvis", "forearm_l"}, {"thigh_r", "forearm_r"}, {"shin_l", "midfoot_l"}, {"pelvis", "chest"}} {
		if !pairs[p] {
			t.Errorf("%s and %s should collide", p[0], p[1])
		}
	}
}

func TestSegmentClosest(t *testing.T) {
	for _, c := range []struct {
		p1, q1, p2, q2 geom.V3
		d              float64
	}{
		{geom.V3{0, 0, 0}, geom.V3{1, 0, 0}, geom.V3{0.5, 1, -1}, geom.V3{0.5, 1, 1}, 1},      // crossing
		{geom.V3{0, 0, 0}, geom.V3{1, 0, 0}, geom.V3{2, 0, 0}, geom.V3{3, 0, 0}, 1},           // collinear, apart
		{geom.V3{0, 0, 0}, geom.V3{1, 0, 0}, geom.V3{0.2, 0.5, 0}, geom.V3{0.8, 0.5, 0}, 0.5}, // parallel
		{geom.V3{0, 0, 0}, geom.V3{0, 0, 0}, geom.V3{-1, 2, 0}, geom.V3{1, 2, 0}, 2},          // point and segment
		{geom.V3{0, 0, 0}, geom.V3{1, 0, 0}, geom.V3{1.5, 0.5, 0}, geom.V3{1.5, 0.5, 0}, math.Sqrt(0.5)},
	} {
		a, b := geom.SegmentClosest(c.p1, c.q1, c.p2, c.q2)
		if d := b.Sub(a).Norm(); math.Abs(d-c.d) > 1e-12 {
			t.Errorf("%v–%v to %v–%v: distance %v, want %v", c.p1, c.q1, c.p2, c.q2, d, c.d)
		}
	}
}
