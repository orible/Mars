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
	"math"
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
	// Stiffness is a passive spring towards 0 (ligaments, a toe's
	// return spring), N·m/rad.
	Stiffness float64 `json:"stiffness,omitempty"`
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
	// Sensors lists the contacts (indices into Contacts) that carry a
	// pressure sensor, such as the pads on a sole.
	Sensors []int `json:"sensors,omitempty"`
	// Colliders are the link's solid volume as capsules. They rest on the
	// floor like contacts do, and keep limbs from passing through each
	// other: every two links' capsules collide, except a link's and its
	// collision parent's (see SelfCollisionPairs), which meet at the joint
	// between them.
	Colliders []Capsule `json:"colliders,omitempty"`
}

// Capsule is the segment A–B swept by a sphere of Radius, in the link
// frame (metres). A == B makes a sphere.
type Capsule struct {
	A      geom.V3 `json:"a"`
	B      geom.V3 `json:"b"`
	Radius float64 `json:"radius"`
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
	// Cables, if set, make this a tendon drive: the cylinder body sits
	// between Base and Rod (often on one link, such as up in the shin)
	// and pulls cables routed over pulleys. Its effective length is the
	// signed sum of the cable lengths: a loop that pulls one cable while
	// paying out its opposite is two cables with signs +½ and −½.
	Cables []Cable `json:"cables,omitempty"`
}

// Cable is one tendon run: a polyline of anchors from the drive to the
// insertion, with pulleys and guides as the points between.
type Cable struct {
	Sign float64  `json:"sign"`
	Path []Anchor `json:"path"`
}

// Tendon is a passive elastic cable, such as the plantar fascia from the
// heel under the toe joint to the toes: taut beyond RestLength, slack
// below it.
type Tendon struct {
	Name       string   `json:"name"`
	Path       []Anchor `json:"path"`
	RestLength float64  `json:"restLength"` // m
	Stiffness  float64  `json:"stiffness"`  // N/m
	Damping    float64  `json:"damping"`    // N·s/m while taut
}

// Routes returns the cylinder's length as signed polylines: its two pins,
// or its cables.
func (c Cylinder) Routes() []Cable {
	if len(c.Cables) > 0 {
		return c.Cables
	}
	return []Cable{{Sign: 1, Path: []Anchor{c.Base, c.Rod}}}
}

// NeutralLength is the cylinder's effective length in the neutral pose.
func (b Body) NeutralLength(routes []Cable) float64 {
	l := 0.0
	for _, r := range routes {
		for i := 1; i < len(r.Path); i++ {
			l += r.Sign * b.ZeroPosition(r.Path[i]).Sub(b.ZeroPosition(r.Path[i-1])).Norm()
		}
	}
	return l
}

// Body is a full robot description. Links[0] is the pelvis.
type Body struct {
	Name      string            `json:"name"`
	Supply    hydraulics.Supply `json:"supply"`
	Links     []Link            `json:"links"`
	Cylinders []Cylinder        `json:"cylinders"`
	Tendons   []Tendon          `json:"tendons,omitempty"`
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

// CollisionParent returns the index of link i's nearest ancestor that has
// colliders, or -1. Light brackets between two limbs (a hip's yaw and
// roll links) usually have none, so a thigh's collision parent is the
// pelvis.
func (b Body) CollisionParent(i int) int {
	for p := b.LinkIndex(b.Links[i].Parent); p >= 0; p = b.LinkIndex(b.Links[p].Parent) {
		if len(b.Links[p].Colliders) > 0 {
			return p
		}
		if p == 0 {
			break
		}
	}
	return -1
}

// SelfCollisionPairs lists the link pairs (lower index first) whose
// capsules collide with each other: every two links with colliders except
// a link and its collision parent, whose capsules overlap round the
// joint between them.
func (b Body) SelfCollisionPairs() [][2]int {
	var out [][2]int
	for i := range b.Links {
		if len(b.Links[i].Colliders) == 0 {
			continue
		}
		for j := i + 1; j < len(b.Links); j++ {
			if len(b.Links[j].Colliders) == 0 || b.CollisionParent(j) == i || b.CollisionParent(i) == j {
				continue
			}
			out = append(out, [2]int{i, j})
		}
	}
	return out
}

// ZeroCapsule returns capsule c of link l in the pelvis frame in the
// neutral pose.
func (b Body) ZeroCapsule(l string, c Capsule) (geom.V3, geom.V3) {
	return b.ZeroPosition(Anchor{l, c.A}), b.ZeroPosition(Anchor{l, c.B})
}

// validateColliders checks capsule sizes and that no two colliding
// capsules overlap in the neutral pose (they would push the body apart
// the moment it starts).
func (b Body) validateColliders() error {
	for _, l := range b.Links {
		for k, c := range l.Colliders {
			ok := c.Radius > 0 && c.Radius < 1
			for i := 0; i < 3; i++ {
				ok = ok && !math.IsNaN(c.A[i]+c.B[i]) && !math.IsInf(c.A[i]+c.B[i], 0)
			}
			if !ok {
				return fmt.Errorf("link %q: collider %d needs finite ends and a radius between 0 and 1 m", l.Name, k)
			}
		}
	}
	for _, pr := range b.SelfCollisionPairs() {
		li, lj := b.Links[pr[0]], b.Links[pr[1]]
		for ki, ci := range li.Colliders {
			a1, b1 := b.ZeroCapsule(li.Name, ci)
			for kj, cj := range lj.Colliders {
				a2, b2 := b.ZeroCapsule(lj.Name, cj)
				p, q := geom.SegmentClosest(a1, b1, a2, b2)
				if over := ci.Radius + cj.Radius - q.Sub(p).Norm(); over > 0 {
					return fmt.Errorf("link %q collider %d and link %q collider %d overlap by %.1f mm in the neutral pose; shrink or move one (only a link and its collision parent may overlap)",
						li.Name, ki, lj.Name, kj, over*1000)
				}
			}
		}
	}
	return nil
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
		for _, k := range l.Sensors {
			if k < 0 || k >= len(l.Contacts) {
				return fmt.Errorf("link %q: sensor %d is not one of its %d contacts", l.Name, k, len(l.Contacts))
			}
		}
		if j := l.Joint; j != nil {
			if j.Stiffness < 0 {
				return fmt.Errorf("link %q: joint stiffness must not be negative", l.Name)
			}
			if j.Axis.Norm() == 0 {
				return fmt.Errorf("link %q: joint axis is zero", l.Name)
			}
			if !(j.Limits[0] <= 0 && 0 <= j.Limits[1] && j.Limits[0] < j.Limits[1]) {
				return fmt.Errorf("link %q: joint limits %v must contain 0 (the neutral pose)", l.Name, j.Limits)
			}
		}
		seen[l.Name] = true
	}
	if err := b.validateColliders(); err != nil {
		return err
	}
	if len(b.Cylinders) == 0 {
		return errors.New("body needs at least one cylinder")
	}
	path := func(what string, p []Anchor) error {
		if len(p) < 2 {
			return fmt.Errorf("%s: a path needs at least two anchors", what)
		}
		for _, a := range p {
			if !seen[a.Link] {
				return fmt.Errorf("%s: anchor link %q must exist", what, a.Link)
			}
		}
		return nil
	}
	for _, c := range b.Cylinders {
		cable := len(c.Cables) > 0
		switch {
		case !seen[c.Base.Link] || !seen[c.Rod.Link]:
			return fmt.Errorf("cylinder %q: anchor links %q / %q must exist", c.Name, c.Base.Link, c.Rod.Link)
		case !cable && c.Base.Link == c.Rod.Link:
			return fmt.Errorf("cylinder %q: both anchors are on %q, so it cannot move anything", c.Name, c.Base.Link)
		case c.BoreArea <= 0 || c.RodArea <= 0 || c.RodArea >= c.BoreArea:
			return fmt.Errorf("cylinder %q: need 0 < rodArea < boreArea", c.Name)
		case c.Stroke <= 0 || c.DeadVolume <= 0 || c.ValveGain <= 0 || (!cable && c.RetractedLength <= 0):
			return fmt.Errorf("cylinder %q: stroke, deadVolume, valveGain and retractedLength must be positive", c.Name)
		}
		for _, r := range c.Cables {
			if err := path(fmt.Sprintf("cylinder %q cable", c.Name), r.Path); err != nil {
				return err
			}
			if r.Sign == 0 {
				return fmt.Errorf("cylinder %q: a cable needs a non-zero sign", c.Name)
			}
		}
		l0 := b.NeutralLength(c.Routes())
		if l0 < c.RetractedLength || l0 > c.RetractedLength+c.Stroke {
			return fmt.Errorf("cylinder %q: its length in the neutral pose is %.4f m but the cylinder covers %.4f–%.4f m; move an anchor or set retractedLength to about %.4f",
				c.Name, l0, c.RetractedLength, c.RetractedLength+c.Stroke, l0-c.Stroke/2)
		}
	}
	for _, t := range b.Tendons {
		if err := path(fmt.Sprintf("tendon %q", t.Name), t.Path); err != nil {
			return err
		}
		if t.Stiffness <= 0 || t.Damping < 0 || t.RestLength <= 0 {
			return fmt.Errorf("tendon %q: need stiffness > 0, damping ≥ 0 and restLength > 0", t.Name)
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
