package env

import (
	"embed"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"mars/sim/model"
)

// Motion references are human movements the robot is rewarded for
// following, such as rising from a squat or getting up off the floor. A
// reference is a list of keyframes in human joint angles (degrees), from
// the literature or extracted from motion capture (cmd/mocapref); it is
// retargeted to each body by clipping to that body's joint limits.
//
// Tracking is state-based: each step the tracker finds the point on the
// reference path nearest the robot's current pose and reports how close it
// is and how far along. No phase clock is fed to the network, so a
// reference can be added without changing the network's inputs. Legs are
// matched to the reference either way round, since either leg may lead.

//go:embed references/*.json
var builtinRefs embed.FS

// Reference is a human motion to imitate.
type Reference struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Sources     []string `json:"sources"`
	// IgnoreAnkle leaves the ankle out of the comparison (for recordings
	// whose foot data cannot be trusted).
	IgnoreAnkle bool       `json:"ignoreAnkle,omitempty"`
	Keyframes   []Keyframe `json:"keyframes"`
}

// Keyframe is one moment of the reference, angles in degrees: forward
// trunk (pelvis) lean, then hip flexion, knee flexion and ankle
// dorsiflexion for both legs alike, or per leg in Legs. Height, if given,
// is the pelvis height as a fraction of standing.
type Keyframe struct {
	T         float64       `json:"t"`
	TrunkLean float64       `json:"trunkLean"`
	Height    float64       `json:"height,omitempty"`
	Hip       float64       `json:"hip,omitempty"`
	Knee      float64       `json:"knee,omitempty"`
	Ankle     float64       `json:"ankle,omitempty"`
	Legs      *[2]LegAngles `json:"legs,omitempty"`
}

// LegAngles are one leg's angles in a keyframe (degrees).
type LegAngles struct {
	Hip   float64 `json:"hip"`
	Knee  float64 `json:"knee"`
	Ankle float64 `json:"ankle"`
}

func (k Keyframe) legs() [2]LegAngles {
	if k.Legs != nil {
		return *k.Legs
	}
	l := LegAngles{k.Hip, k.Knee, k.Ankle}
	return [2]LegAngles{l, l}
}

// ReferenceNames lists the built-in references.
func ReferenceNames() []string {
	ents, _ := builtinRefs.ReadDir("references")
	var n []string
	for _, e := range ents {
		n = append(n, strings.TrimSuffix(e.Name(), ".json"))
	}
	sort.Strings(n)
	return n
}

// LoadReferences reads a comma-separated list of references; "all" means
// every built-in one.
func LoadReferences(list string) ([]Reference, error) {
	if strings.TrimSpace(list) == "all" {
		list = strings.Join(ReferenceNames(), ",")
	}
	var out []Reference
	for _, s := range strings.Split(list, ",") {
		if s = strings.TrimSpace(s); s == "" {
			continue
		}
		r, err := LoadReference(s)
		if err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, nil
}

// PickTracker builds a tracker for each reference on this body and
// returns the one whose path the current pose is closest to, so each
// episode imitates the human movement that fits how it starts.
func PickTracker(refs []Reference, w *model.World) *MotionTracker {
	var best *MotionTracker
	for _, r := range refs {
		tr := NewMotionTracker(r, w.M)
		if tr == nil {
			continue
		}
		tr.Update(w)
		if best == nil || tr.Error < best.Error {
			best = tr
		}
	}
	return best
}

// LoadReference reads a built-in reference by name (see env/references)
// or a reference JSON file.
func LoadReference(s string) (Reference, error) {
	var data []byte
	var err error
	if strings.HasSuffix(s, ".json") {
		data, err = os.ReadFile(s)
	} else {
		data, err = builtinRefs.ReadFile("references/" + s + ".json")
	}
	if err != nil {
		return Reference{}, fmt.Errorf("reference %q: %w", s, err)
	}
	var r Reference
	if err := json.Unmarshal(data, &r); err != nil {
		return Reference{}, fmt.Errorf("reference %q: %w", s, err)
	}
	if len(r.Keyframes) < 2 {
		return Reference{}, fmt.Errorf("reference %q needs at least two keyframes", s)
	}
	for i := 1; i < len(r.Keyframes); i++ {
		if r.Keyframes[i].T <= r.Keyframes[i-1].T {
			return Reference{}, fmt.Errorf("reference %q: keyframe times must increase", s)
		}
	}
	return r, nil
}

// pose is one point of a reference path, in radians (height a fraction).
type pose struct {
	lean, height float64
	legs         [2][3]float64 // hip, knee, ankle per leg
}

// at interpolates the reference at normalised time t (0..1).
func (r Reference) at(t float64) pose {
	k := r.Keyframes
	t = math.Max(k[0].T, math.Min(k[len(k)-1].T, t))
	i := 1
	for i < len(k)-1 && k[i].T < t {
		i++
	}
	a, b := k[i-1], k[i]
	f := (t - a.T) / (b.T - a.T)
	lerp := func(x, y float64) float64 { return x + f*(y-x) }
	rad := math.Pi / 180
	p := pose{lean: lerp(a.TrunkLean, b.TrunkLean) * rad, height: lerp(a.Height, b.Height)}
	la, lb := a.legs(), b.legs()
	for s := 0; s < 2; s++ {
		p.legs[s] = [3]float64{lerp(la[s].Hip, lb[s].Hip) * rad, lerp(la[s].Knee, lb[s].Knee) * rad, lerp(la[s].Ankle, lb[s].Ankle) * rad}
	}
	return p
}

func (r Reference) hasHeight() bool {
	for _, k := range r.Keyframes {
		if k.Height <= 0 {
			return false
		}
	}
	return true
}

// TrunkLean is the pelvis's tilt from vertical (rad), measured the way
// the motion capture is: positive when the front faces down (leaning
// forward, lying face down), negative when it faces up (leaning back,
// lying on the back). Lying on a side reads about ±90°.
func TrunkLean(w *model.World) float64 {
	up := w.Up()
	lean := math.Acos(math.Max(-1, math.Min(1, up[1])))
	if w.Base.R[1][0] > 0 { // forward axis points up
		lean = -lean
	}
	return lean
}

// MotionTracker follows one robot along a reference.
type MotionTracker struct {
	Ref       Reference
	path      []pose // retargeted samples
	legs      []leg  // the robot's legs, left first
	useHeight bool

	Progress float64 // 0..1 along the reference, at the nearest point
	Error    float64 // rms joint-angle error to that point, rad
}

// leg holds the q indices of one leg's hip pitch, knee and ankle.
type leg struct{ hip, knee, ankle int }

// trackSamples is how finely the reference path is sampled.
const trackSamples = 101

// NewMotionTracker retargets ref to the body of m: joint roles come from
// link names (thigh = hip pitch, shin = knee, foot = ankle; _l and _r
// sides), and the path is clipped to the joint limits. It returns nil if
// the body has no two legs it can recognise.
func NewMotionTracker(ref Reference, m *model.Model) *MotionTracker {
	tr := &MotionTracker{Ref: ref, useHeight: ref.hasHeight()}
	bySide := map[string]*leg{}
	for q := 0; q < m.NDof(); q++ {
		name := m.Names[q+1]
		i := strings.LastIndex(name, "_")
		if i < 0 {
			continue
		}
		role, side := name[:i], name[i+1:]
		l := bySide[side]
		if l == nil {
			l = &leg{-1, -1, -1}
			bySide[side] = l
		}
		switch role {
		case "thigh":
			l.hip = q
		case "shin":
			l.knee = q
		case "foot":
			l.ankle = q
		}
	}
	for _, side := range []string{"l", "r"} {
		if l := bySide[side]; l != nil && l.hip >= 0 && l.knee >= 0 && l.ankle >= 0 {
			tr.legs = append(tr.legs, *l)
		}
	}
	if len(tr.legs) != 2 {
		return nil
	}
	clip := func(q int, v float64) float64 { return math.Max(m.Lo[q+1], math.Min(m.Hi[q+1], v)) }
	for i := 0; i < trackSamples; i++ {
		p := ref.at(float64(i) / (trackSamples - 1))
		// Robot conventions: hip flexion +q, knee flexion −q, dorsiflexion +q.
		// Store the clipped targets back in human convention.
		for s, l := range tr.legs {
			p.legs[s] = [3]float64{clip(l.hip, p.legs[s][0]), -clip(l.knee, -p.legs[s][1]), clip(l.ankle, p.legs[s][2])}
		}
		tr.path = append(tr.path, p)
	}
	return tr
}

// Update measures the robot's pose against the reference: the nearest
// point on the path (Progress) and the rms angle error to it (Error).
func (tr *MotionTracker) Update(w *model.World) {
	lean := TrunkLean(w)
	h := (w.Base.P[1] - w.GroundUnder()) / w.StandH()
	var robot [2][3]float64
	for s, l := range tr.legs {
		robot[s] = [3]float64{w.Q[l.hip], -w.Q[l.knee], w.Q[l.ankle]}
	}
	ankleW := 0.5
	if tr.Ref.IgnoreAnkle {
		ankleW = 0
	}
	legErr := func(a, b [3]float64) float64 {
		dh, dk, da := a[0]-b[0], a[1]-b[1], a[2]-b[2]
		return dh*dh + dk*dk + ankleW*da*da
	}
	best, bestI := math.Inf(1), 0
	for i, p := range tr.path {
		d := (lean - p.lean) * (lean - p.lean)
		if tr.useHeight {
			dh := 2 * (h - p.height) // 10% of standing height counts like 0.2 rad
			d += dh * dh
		}
		same := legErr(robot[0], p.legs[0]) + legErr(robot[1], p.legs[1])
		swap := legErr(robot[0], p.legs[1]) + legErr(robot[1], p.legs[0])
		d += math.Min(same, swap) / 2
		if d < best {
			best, bestI = d, i
		}
	}
	tr.Progress = float64(bestI) / (trackSamples - 1)
	n := 3 + ankleW
	if tr.useHeight {
		n++
	}
	tr.Error = math.Sqrt(best / n)
}
