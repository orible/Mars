package env

import (
	"embed"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"

	"mars/sim/model"
)

// Motion references are human movements the robot is rewarded for
// following, such as rising from a squat. A reference is a list of
// keyframes in human joint angles (degrees, from the literature); it is
// retargeted to each body by clipping to that body's joint limits.
//
// Tracking is state-based: each step the tracker finds the point on the
// reference path nearest the robot's current pose and reports how close it
// is and how far along. No phase clock is fed to the network, so a
// reference can be added without changing the network's inputs.

//go:embed references/*.json
var builtinRefs embed.FS

// Reference is a human motion to imitate.
type Reference struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Sources     []string   `json:"sources"`
	Keyframes   []Keyframe `json:"keyframes"`
}

// Keyframe is one moment of the reference, in degrees: forward trunk
// (pelvis) lean, hip flexion, knee flexion, ankle dorsiflexion.
type Keyframe struct {
	T         float64 `json:"t"`
	TrunkLean float64 `json:"trunkLean"`
	Hip       float64 `json:"hip"`
	Knee      float64 `json:"knee"`
	Ankle     float64 `json:"ankle"`
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

// at interpolates the reference at normalised time t (0..1), in radians.
func (r Reference) at(t float64) [4]float64 {
	k := r.Keyframes
	t = math.Max(k[0].T, math.Min(k[len(k)-1].T, t))
	i := 1
	for i < len(k)-1 && k[i].T < t {
		i++
	}
	a, b := k[i-1], k[i]
	f := (t - a.T) / (b.T - a.T)
	lerp := func(x, y float64) float64 { return (x + f*(y-x)) * math.Pi / 180 }
	return [4]float64{lerp(a.TrunkLean, b.TrunkLean), lerp(a.Hip, b.Hip), lerp(a.Knee, b.Knee), lerp(a.Ankle, b.Ankle)}
}

// MotionTracker follows one robot along a reference.
type MotionTracker struct {
	Ref  Reference
	path [][4]float64 // retargeted samples: lean, hip, knee, ankle (rad)
	legs []leg

	Progress float64 // 0..1 along the reference, at the nearest point
	Error    float64 // rms joint-angle error to that point, rad
}

// leg holds the q indices of one leg's hip pitch, knee and ankle.
type leg struct{ hip, knee, ankle int }

// trackSamples is how finely the reference path is sampled.
const trackSamples = 101

// NewMotionTracker retargets ref to the body of m: joint roles come from link
// names (thigh = hip pitch, shin = knee, foot = ankle), and the path is
// clipped to the joint limits. It returns nil if the body has no legs it
// can recognise.
func NewMotionTracker(ref Reference, m *model.Model) *MotionTracker {
	tr := &MotionTracker{Ref: ref}
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
	for _, l := range bySide {
		if l.hip >= 0 && l.knee >= 0 && l.ankle >= 0 {
			tr.legs = append(tr.legs, *l)
		}
	}
	if len(tr.legs) == 0 {
		return nil
	}
	l0 := tr.legs[0]
	clip := func(q int, v float64) float64 { return math.Max(m.Lo[q+1], math.Min(m.Hi[q+1], v)) }
	for i := 0; i < trackSamples; i++ {
		p := ref.at(float64(i) / (trackSamples - 1))
		// Robot joint conventions: hip flexion +q, knee flexion −q, dorsiflexion +q.
		tr.path = append(tr.path, [4]float64{p[0], clip(l0.hip, p[1]), -clip(l0.knee, -p[2]), clip(l0.ankle, p[3])})
	}
	return tr
}

// Update measures the robot's pose against the reference: the nearest
// point on the path (Progress) and the rms angle error to it (Error).
func (tr *MotionTracker) Update(w *model.World, pitch float64) {
	lean := -pitch // nose down is leaning forward
	best, bestI := math.Inf(1), 0
	for i, p := range tr.path {
		d := (lean - p[0]) * (lean - p[0])
		for _, l := range tr.legs {
			dh, dk, da := w.Q[l.hip]-p[1], -w.Q[l.knee]-p[2], w.Q[l.ankle]-p[3]
			d += (dh*dh + dk*dk + 0.5*da*da) / float64(len(tr.legs))
		}
		if d < best {
			best, bestI = d, i
		}
	}
	tr.Progress = float64(bestI) / (trackSamples - 1)
	tr.Error = math.Sqrt(best / 3.5)
}
