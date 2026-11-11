// Package body holds the data-driven 3D body description shared by the
// simulator, the observation builder and the observer.
//
// A body is a tree of rigid links joined by hinge joints, rooted at the
// pelvis. A multi-axis joint (such as a hip with roll and pitch) is a chain
// of hinges through a light intermediate link. Cylinders are defined by two
// anchors, each a point on a link, so mounting points, which joints a
// cylinder spans, and its strength are all data. The pelvis configuration
// (hip spacing, cylinder anchor plates) is therefore edited in the spec, not
// in code.
//
// When measurements of the real robot disagree with the spec, edit the
// file: the sim server reloads it and the same policy weights keep running.
package body

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"os"

	"mars/sim/geom"
	"mars/sim/hydraulics"
)

// Joint connects a link to its parent. q = 0 is the neutral pose.
type Joint struct {
	Origin  geom.V3    `json:"origin"`  // joint centre in the parent link's frame
	Axis    geom.V3    `json:"axis"`    // hinge axis in the parent link's frame
	Limits  [2]float64 `json:"limits"`  // rad, must contain 0
	Damping float64    `json:"damping"` // N·m·s/rad
}

// Link is one rigid body. Its frame sits at its joint centre and coincides
// with the parent's frame orientation when q = 0.
type Link struct {
	Name    string  `json:"name"`
	Parent  string  `json:"parent,omitempty"`
	Joint   *Joint  `json:"joint,omitempty"`
	Mass    float64 `json:"mass"`    // kg
	Com     geom.V3 `json:"com"`     // centre of mass in the link frame
	Inertia geom.V3 `json:"inertia"` // kg·m², principal moments about the CoM along the link axes
	// Contacts are points in the link frame that can touch the ground,
	// such as the heel and toe corners of a foot sole. Only these points
	// collide; add some to knees or the pelvis to catch falls.
	Contacts []geom.V3 `json:"contacts,omitempty"`
}

// Anchor is a pin position on a link, in that link's frame.
type Anchor struct {
	Link  string  `json:"link"`
	Point geom.V3 `json:"point"`
}

// Cylinder is a hydraulic actuator pinned between two links. It acts along
// the line between its anchors, and its leverage about each joint it spans
// changes with the pose.
type Cylinder struct {
	Name            string  `json:"name"`
	Base            Anchor  `json:"base"`            // barrel end
	Rod             Anchor  `json:"rod"`             // rod end
	RetractedLength float64 `json:"retractedLength"` // m, pin to pin when fully retracted
	hydraulics.Params
}

// Body is a full robot description. Links[0] is the pelvis.
type Body struct {
	Name      string            `json:"name"`
	Supply    hydraulics.Supply `json:"supply"`
	Links     []Link            `json:"links"`
	Cylinders []Cylinder        `json:"cylinders"`
}

// LinkIndex returns the index of the named link, or -1.
func (b Body) LinkIndex(name string) int {
	for i, l := range b.Links {
		if l.Name == name {
			return i
		}
	}
	return -1
}

// ZeroPosition returns an anchor's position in the pelvis frame in the
// neutral pose (every joint at 0, so no rotations).
func (b Body) ZeroPosition(a Anchor) geom.V3 {
	p := a.Point
	for i := b.LinkIndex(a.Link); i > 0; i = b.LinkIndex(b.Links[i].Parent) {
		p = p.Add(b.Links[i].Joint.Origin)
	}
	return p
}

// Validate rejects specs the simulator cannot run, and pelvis or limb
// geometries in which a cylinder cannot be assembled in the neutral pose.
func (b Body) Validate() error {
	if len(b.Links) == 0 {
		return errors.New("body needs at least one link")
	}
	if b.Supply.Pressure <= 0 || b.Supply.BulkModulus <= 0 {
		return errors.New("supply pressure and bulk modulus must be positive")
	}
	seen := map[string]bool{}
	for i, l := range b.Links {
		switch {
		case l.Name == "":
			return fmt.Errorf("link %d has no name", i)
		case seen[l.Name]:
			return fmt.Errorf("link name %q is used twice", l.Name)
		case i == 0 && (l.Parent != "" || l.Joint != nil):
			return fmt.Errorf("link %q: the first link is the pelvis (root) and has no parent or joint", l.Name)
		case i > 0 && (l.Parent == "" || l.Joint == nil):
			return fmt.Errorf("link %q needs a parent and a joint", l.Name)
		case i > 0 && !seen[l.Parent]:
			return fmt.Errorf("link %q: parent %q must be listed before it", l.Name, l.Parent)
		case l.Mass <= 0 || l.Inertia[0] <= 0 || l.Inertia[1] <= 0 || l.Inertia[2] <= 0:
			return fmt.Errorf("link %q: mass and all three inertia moments must be positive", l.Name)
		}
		if j := l.Joint; j != nil {
			if j.Axis.Norm() == 0 {
				return fmt.Errorf("link %q: joint axis is zero", l.Name)
			}
			if !(j.Limits[0] <= 0 && 0 <= j.Limits[1] && j.Limits[0] < j.Limits[1]) {
				return fmt.Errorf("link %q: joint limits %v must contain 0 (the neutral pose)", l.Name, j.Limits)
			}
		}
		seen[l.Name] = true
	}
	if len(b.Cylinders) == 0 {
		return errors.New("body needs at least one cylinder")
	}
	for _, c := range b.Cylinders {
		switch {
		case !seen[c.Base.Link] || !seen[c.Rod.Link]:
			return fmt.Errorf("cylinder %q: anchor links %q / %q must exist", c.Name, c.Base.Link, c.Rod.Link)
		case c.Base.Link == c.Rod.Link:
			return fmt.Errorf("cylinder %q: both anchors are on %q, so it cannot move anything", c.Name, c.Base.Link)
		case c.BoreArea <= 0 || c.RodArea <= 0 || c.RodArea >= c.BoreArea:
			return fmt.Errorf("cylinder %q: need 0 < rodArea < boreArea", c.Name)
		case c.Stroke <= 0 || c.DeadVolume <= 0 || c.ValveGain <= 0 || c.RetractedLength <= 0:
			return fmt.Errorf("cylinder %q: stroke, deadVolume, valveGain and retractedLength must be positive", c.Name)
		}
		l0 := b.ZeroPosition(c.Rod).Sub(b.ZeroPosition(c.Base)).Norm()
		if l0 < c.RetractedLength || l0 > c.RetractedLength+c.Stroke {
			return fmt.Errorf("cylinder %q: pin-to-pin length in the neutral pose is %.4f m but the cylinder covers %.4f–%.4f m; move an anchor or set retractedLength to about %.4f",
				c.Name, l0, c.RetractedLength, c.RetractedLength+c.Stroke, l0-c.Stroke/2)
		}
	}
	return nil
}

// Load reads and validates a body spec.
func Load(path string) (Body, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Body{}, err
	}
	var b Body
	if err := json.Unmarshal(data, &b); err != nil {
		return Body{}, fmt.Errorf("%s: %w", path, err)
	}
	if err := b.Validate(); err != nil {
		return Body{}, fmt.Errorf("%s: %w", path, err)
	}
	return b, nil
}

func rodInertia(m, l float64) geom.V3 {
	i := m * l * l / 12
	return geom.V3{i, m * 4e-4, i}
}

// Random draws a biped for morphology randomisation: pelvis width and hip
// drop, segment lengths, masses, payload CoG, supply, cylinder sizes and
// anchor points all vary; some bodies get a second knee cylinder and some
// lose their hip-roll cylinders.
func Random(rng *rand.Rand) Body {
	u := func(lo, hi float64) float64 { return lo + rng.Float64()*(hi-lo) }
	jit := func(v geom.V3) geom.V3 {
		const s = 0.01
		return v.Add(geom.V3{u(-s, s), u(-s, s), u(-s, s)})
	}
	hw, drop := u(0.06, 0.12), u(0.02, 0.08)
	th, sh := u(0.18, 0.32), u(0.18, 0.32)
	b := Body{
		Supply: hydraulics.Supply{Pressure: u(2.5e6, 4e6), BulkModulus: u(1e8, 4e8)},
		Links: []Link{{
			Name: "pelvis", Mass: u(1.5, 6),
			Com:      geom.V3{u(-0.05, 0.05), u(0, 0.12), u(-0.03, 0.03)},
			Inertia:  geom.V3{0.01, 0.01, 0.01},
			Contacts: []geom.V3{{0.08, 0, -hw - 0.02}, {0.08, 0, hw + 0.02}, {-0.08, 0, -hw - 0.02}, {-0.08, 0, hw + 0.02}},
		}},
	}
	link := func(name, parent string, origin, axis geom.V3, lo, hi, mass float64, com, inertia geom.V3) {
		b.Links = append(b.Links, Link{Name: name, Parent: parent, Mass: mass, Com: com, Inertia: inertia,
			Joint: &Joint{Origin: origin, Axis: axis, Limits: [2]float64{lo, hi}, Damping: 0.05}})
	}
	cyl := func(name string, base, rod Anchor) {
		bore, stroke := u(1.2e-4, 3.2e-4), u(0.04, 0.09)
		l0 := b.ZeroPosition(rod).Sub(b.ZeroPosition(base)).Norm()
		b.Cylinders = append(b.Cylinders, Cylinder{
			Name: name, Base: base, Rod: rod, RetractedLength: l0 - stroke*u(0.3, 0.7),
			Params: hydraulics.Params{
				BoreArea: bore, RodArea: bore * u(0.5, 0.75), Stroke: stroke,
				DeadVolume: u(3e-6, 1e-5), Viscous: u(100, 400),
				ValveGain: u(6e-9, 1.2e-8), ValveTau: u(0.005, 0.02),
			},
		})
	}
	hipRoll := rng.Float64() > 0.25
	doubleKnee := rng.Float64() < 0.3
	for _, side := range []struct {
		s string
		z float64
	}{{"l", -1}, {"r", 1}} {
		s, z := side.s, side.z
		thM, shM, ftM := u(0.5, 1.2), u(0.4, 0.9), u(0.2, 0.5)
		link("hip_roll_"+s, "pelvis", geom.V3{0, -drop, z * hw}, geom.V3{1, 0, 0}, -0.35, 0.35, 0.05, geom.V3{}, geom.V3{1e-5, 1e-5, 1e-5})
		link("thigh_"+s, "hip_roll_"+s, geom.V3{}, geom.V3{0, 0, 1}, -0.35, 1.0, thM, geom.V3{0, -th / 2, 0}, rodInertia(thM, th))
		link("shin_"+s, "thigh_"+s, geom.V3{0, -th, 0}, geom.V3{0, 0, 1}, -1.5, 0, shM, geom.V3{0, -sh / 2, 0}, rodInertia(shM, sh))
		b.Links[len(b.Links)-1].Contacts = []geom.V3{{0.03, 0, 0}} // kneecap
		link("foot_"+s, "shin_"+s, geom.V3{0, -sh, 0}, geom.V3{0, 0, 1}, -0.5, 0.4, ftM, geom.V3{0.03, -0.03, 0}, geom.V3{2e-4, 4e-4, 3e-4})
		sole, heel, toe, half := -u(0.04, 0.06), -u(0.04, 0.06), u(0.09, 0.14), u(0.025, 0.045)
		b.Links[len(b.Links)-1].Contacts = []geom.V3{{heel, sole, -half}, {heel, sole, half}, {toe, sole, -half}, {toe, sole, half}}

		// Template: the anchor layout found for biped.json (walking range
		// with the most leverage), scaled to segment length and jittered.
		cyl("hip_pitch_"+s, Anchor{"pelvis", jit(geom.V3{0.06, 0.005 - drop, z * hw})}, Anchor{"thigh_" + s, jit(geom.V3{0, -0.8 * th, 0})})
		if hipRoll {
			cyl("hip_roll_"+s, Anchor{"pelvis", jit(geom.V3{0, 0.07 - drop, z * (hw - 0.06)})}, Anchor{"thigh_" + s, jit(geom.V3{0, -0.08, z * 0.035})})
		}
		cyl("knee_"+s, Anchor{"thigh_" + s, jit(geom.V3{0.035, -0.2 * th, 0})}, Anchor{"shin_" + s, jit(geom.V3{-0.05, -0.02, 0})})
		if doubleKnee {
			cyl("knee_back_"+s, Anchor{"thigh_" + s, jit(geom.V3{-0.035, -0.2 * th, 0})}, Anchor{"shin_" + s, jit(geom.V3{0.05, -0.02, 0})})
		}
		cyl("ankle_"+s, Anchor{"shin_" + s, jit(geom.V3{-0.035, -0.2 * sh, 0})}, Anchor{"foot_" + s, jit(geom.V3{-0.05, 0, 0})})
	}
	b.Name = fmt.Sprintf("random-%d", len(b.Cylinders))
	return b
}
