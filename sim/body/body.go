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
