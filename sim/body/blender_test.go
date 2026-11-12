package body

import (
	"math"
	"testing"

	"mars/sim/geom"
)

// These tests check the coordinate maths of the Blender add-on
// (tools/blender/mars_body.py) in Go: the same mapping, the same float32
// storage Blender uses for bone heads and object positions, and the same
// rule on export (a value within keepTol of the original keeps the
// original number exactly; anything else is rounded to the micrometre).
// The add-on's own round trip runs inside Blender with
// tools/blender/test_roundtrip.py.

// toBlender maps sim (x forward, y up, z right) to Blender (a character
// faces −Y with Z up and its right side at −X): (x, y, z) → (−z, −x, y).
func toBlender(v geom.V3) geom.V3 { return geom.V3{-v[2], -v[0], v[1]} }

// fromBlender is its inverse: (X, Y, Z) → (−Y, Z, −X).
func fromBlender(v geom.V3) geom.V3 { return geom.V3{-v[1], v[2], -v[0]} }

func f32(v geom.V3) geom.V3 {
	return geom.V3{float64(float32(v[0])), float64(float32(v[1])), float64(float32(v[2]))}
}

const keepTol = 1e-5

func keep(orig, got geom.V3) geom.V3 {
	if orig.Sub(got).Norm() <= keepTol {
		return orig
	}
	for i := range got {
		got[i] = math.Round(got[i]*1e6) / 1e6
	}
	return got
}

func TestBlenderMappingIsARotation(t *testing.T) {
	cases := []struct{ sim, blender geom.V3 }{
		{geom.V3{1, 0, 0}, geom.V3{0, -1, 0}}, // forward → −Y
		{geom.V3{0, 1, 0}, geom.V3{0, 0, 1}},  // up → Z
		{geom.V3{0, 0, 1}, geom.V3{-1, 0, 0}}, // right → −X
	}
	for _, c := range cases {
		if got := toBlender(c.sim); got != c.blender {
			t.Errorf("sim %v → %v, want %v", c.sim, got, c.blender)
		}
		if got := fromBlender(c.blender); got != c.sim {
			t.Errorf("blender %v → %v, want %v", c.blender, got, c.sim)
		}
	}
	// Right-handed both sides: x × y = z must map to X' × Y' = Z'.
	if got := toBlender(geom.V3{1, 0, 0}).Cross(toBlender(geom.V3{0, 1, 0})); got != toBlender(geom.V3{0, 0, 1}) {
		t.Errorf("mapping is a reflection: %v", got)
	}
}

// blenderRoundTrip imports b the way the add-on does (bone heads and
// points stored as float32 in Blender space, the armature lifted off the
// floor), lets edit move points in Blender space, and exports.
func blenderRoundTrip(b Body, edit func(what string, p geom.V3) geom.V3) Body {
	lift := geom.V3{0.3, -0.2, 1.1}
	origin := make([]geom.V3, len(b.Links))
	head := make([]geom.V3, len(b.Links))
	for i, l := range b.Links {
		origin[i] = b.ZeroPosition(Anchor{l.Name, geom.V3{}})
		head[i] = f32(toBlender(origin[i]).Add(lift))
	}
	// Export: everything relative to the pelvis bone's head.
	newOrigin := make([]geom.V3, len(b.Links))
	for i := range b.Links {
		newOrigin[i] = fromBlender(head[i].Sub(head[0]))
	}
	point := func(what string, l string, p geom.V3) geom.V3 {
		i := b.LinkIndex(l)
		w := f32(toBlender(origin[i].Add(p)).Add(lift))
		w = edit(what, w)
		return keep(p, fromBlender(w.Sub(head[0])).Sub(newOrigin[i]))
	}
	out := b
	out.Links = append([]Link(nil), b.Links...)
	for i := range out.Links {
		l := &out.Links[i]
		if l.Joint != nil {
			j := *l.Joint
			j.Origin = keep(j.Origin, newOrigin[i].Sub(newOrigin[b.LinkIndex(l.Parent)]))
			l.Joint = &j
		}
		l.Contacts = append([]geom.V3(nil), l.Contacts...)
		for k, c := range l.Contacts {
			l.Contacts[k] = point("contact", l.Name, c)
		}
		l.Colliders = append([]Capsule(nil), l.Colliders...)
		for k, c := range l.Colliders {
			l.Colliders[k] = Capsule{A: point("collider a", l.Name, c.A), B: point("collider b", l.Name, c.B), Radius: c.Radius}
		}
	}
	out.Cylinders = append([]Cylinder(nil), b.Cylinders...)
	for k := range out.Cylinders {
		c := &out.Cylinders[k]
		c.Base.Point = point(c.Name+" base", c.Base.Link, c.Base.Point)
		c.Rod.Point = point(c.Name+" rod", c.Rod.Link, c.Rod.Point)
	}
	return out
}

func TestBlenderRoundTripKeepsEveryNumber(t *testing.T) {
	b, err := Load("../bodies/biped.json")
	if err != nil {
		t.Fatal(err)
	}
	same := blenderRoundTrip(b, func(_ string, p geom.V3) geom.V3 { return p })
	for i, l := range b.Links {
		o := same.Links[i]
		if l.Joint != nil && o.Joint.Origin != l.Joint.Origin {
			t.Errorf("%s joint origin %v → %v", l.Name, l.Joint.Origin, o.Joint.Origin)
		}
		for k := range l.Contacts {
			if o.Contacts[k] != l.Contacts[k] {
				t.Errorf("%s contact %d %v → %v", l.Name, k, l.Contacts[k], o.Contacts[k])
			}
		}
		for k := range l.Colliders {
			if o.Colliders[k] != l.Colliders[k] {
				t.Errorf("%s collider %d %v → %v", l.Name, k, l.Colliders[k], o.Colliders[k])
			}
		}
	}
	for k, c := range b.Cylinders {
		if same.Cylinders[k].Base != c.Base || same.Cylinders[k].Rod != c.Rod {
			t.Errorf("%s anchors changed", c.Name)
		}
	}

	// Moving the left knee cylinder's rod pin 1 cm to Blender +X (the
	// robot's left, sim −z) changes that number and nothing else.
	moved := blenderRoundTrip(b, func(what string, p geom.V3) geom.V3 {
		if what == "knee_l rod" {
			return p.Add(geom.V3{0.01, 0, 0})
		}
		return p
	})
	for k, c := range b.Cylinders {
		got := moved.Cylinders[k]
		switch {
		case c.Name == "knee_l":
			if d := got.Rod.Point.Sub(c.Rod.Point); d.Sub(geom.V3{0, 0, -0.01}).Norm() > 2e-6 {
				t.Errorf("knee_l rod moved by %v, want [0 0 -0.01]", d)
			}
			if got.Base != c.Base {
				t.Errorf("knee_l base moved")
			}
		case got.Base != c.Base || got.Rod != c.Rod:
			t.Errorf("%s moved although only knee_l's rod was edited", c.Name)
		}
	}
	if err := moved.Validate(); err != nil {
		t.Errorf("edited body no longer valid: %v", err)
	}
}
