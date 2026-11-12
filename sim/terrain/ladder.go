package terrain

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// Ladders are the terrain kinds a curriculum can sweep: Ladder(kind, 0)
// is the easiest world of a kind and Ladder(kind, 1) the hardest. Every
// length is for a full-size (1.8 m) robot; set Spec.Scale for a smaller
// one.
var Ladders = map[string]struct {
	Doc  string
	Make func(level float64) Spec
}{
	"stairs-up": {"6 steps up to a 1 m landing and on along it; rise 4→20 cm, run 34→26 cm", func(l float64) Spec {
		return Spec{Stairs: []Stairs{{At: [2]float64{1, 0}, Rise: lerp(0.04, 0.20, l), Run: lerp(0.34, 0.26, l), Steps: 6, Width: 1.6, Landing: 3}}, Goal: [2]float64{1 + 6*lerp(0.34, 0.26, l) + 1, 0}}
	}},
	"stairs-down": {"start on a platform, 6 steps down; rise 4→20 cm, run 34→26 cm", func(l float64) Spec {
		return Spec{Stairs: []Stairs{{At: [2]float64{0.6, 0}, Rise: lerp(0.04, 0.20, l), Run: lerp(0.34, 0.26, l), Steps: 6, Width: 1.6, Down: true, Platform: 2}}, Goal: [2]float64{0.6 + 6*lerp(0.34, 0.26, l) + 1.2, 0}}
	}},
	"stairs-over": {"up 4 steps, across a 1.2 m landing and down 4 steps; rise 4→18 cm", func(l float64) Spec {
		run := lerp(0.34, 0.28, l)
		return Spec{Stairs: []Stairs{{At: [2]float64{1, 0}, Rise: lerp(0.04, 0.18, l), Run: run, Steps: 4, Width: 1.6, Landing: 1.2, Over: true}}, Goal: [2]float64{1 + 7*run + 1.2 + 1, 0}}
	}},
	"slope": {"the whole floor tilted uphill ahead; 0→15°", func(l float64) Spec {
		return Spec{Slope: &Slope{AngleDeg: 15 * l}, Goal: [2]float64{4, 0}}
	}},
	"slope-across": {"the whole floor tilted to the robot's left side; 0→12°", func(l float64) Spec {
		return Spec{Slope: &Slope{AngleDeg: 12 * l, HeadingDeg: 90}, Goal: [2]float64{4, 0}}
	}},
	"ramp": {"a ramp up to a 1.5 m landing and down again, 0.4 m high; 3→15°", func(l float64) Spec {
		return Spec{Ramps: []Ramp{{At: [2]float64{1, 0}, AngleDeg: lerp(3, 15, l), Rise: 0.4, Landing: 1.5}}, Goal: [2]float64{1 + 2*0.4/math.Tan(rad(lerp(3, 15, l))) + 1.5 + 1, 0}}
	}},
	"field": {"octave-noise ground; peak height 0→12 cm, rougher as it rises", func(l float64) Spec {
		return Spec{Noise: &Noise{Amplitude: 0.12 * l, Wavelength: lerp(3, 2, l), Octaves: 5, Persistence: lerp(0.4, 0.55, l)}, Goal: [2]float64{4, 0}}
	}},
	"clutter": {"boxes and cylinders scattered round the start and the way ahead; 4→30 objects", func(l float64) Spec {
		return Spec{Scatter: &Scatter{Count: int(math.Round(lerp(4, 30, l))), Min: [2]float64{-3, -3}, Max: [2]float64{6, 3},
			Size: [2]float64{0.15, 0.6}, Height: [2]float64{0.05, lerp(0.3, 1.0, l)}, Clear: 0.7, Gap: 0.15}, Goal: [2]float64{5, 0}}
	}},
}

// LadderNames lists the ladder kinds.
func LadderNames() []string {
	var n []string
	for k := range Ladders {
		n = append(n, k)
	}
	sort.Strings(n)
	return n
}

// Ladder returns the world of kind at difficulty level (clamped to 0–1).
func Ladder(kind string, level float64) (Spec, error) {
	l, ok := Ladders[kind]
	if !ok {
		return Spec{}, fmt.Errorf("terrain: unknown ladder %q (%s)", kind, strings.Join(LadderNames(), ", "))
	}
	return l.Make(math.Max(0, math.Min(1, level))), nil
}

func lerp(a, b, t float64) float64 { return a + (b-a)*t }

// scaled returns s with every length multiplied by k (angles kept).
func (s Spec) scaled(k float64) Spec {
	if k == 1 {
		return s
	}
	xz := func(p [2]float64) [2]float64 { return [2]float64{p[0] * k, p[1] * k} }
	out := s
	out.Scale = 1
	out.Goal = xz(s.Goal)
	if s.Noise != nil {
		n := *s.Noise
		n.Amplitude, n.Wavelength = n.Amplitude*k, n.Wavelength*k
		out.Noise = &n
	} else if s.NoisePreset != "" {
		n := NoisePresets[s.NoisePreset]
		n.Amplitude, n.Wavelength = n.Amplitude*k, n.Wavelength*k
		out.Noise, out.NoisePreset = &n, ""
	}
	out.Ramps = nil
	for _, r := range s.Ramps {
		r.At, r.Rise, r.Landing = xz(r.At), r.Rise*k, r.Landing*k
		out.Ramps = append(out.Ramps, r)
	}
	out.Stairs = nil
	for _, st := range s.Stairs {
		st.At, st.Rise, st.Run, st.Width, st.Landing, st.Platform = xz(st.At), st.Rise*k, st.Run*k, st.Width*k, st.Landing*k, st.Platform*k
		out.Stairs = append(out.Stairs, st)
	}
	out.Objects = nil
	for _, o := range s.Objects {
		o.At, o.Size = xz(o.At), [3]float64{o.Size[0] * k, o.Size[1] * k, o.Size[2] * k}
		out.Objects = append(out.Objects, o)
	}
	if s.Scatter != nil {
		sc := *s.Scatter
		sc.Min, sc.Max, sc.Size, sc.Height = xz(sc.Min), xz(sc.Max), xz(sc.Size), xz(sc.Height)
		sc.Clear, sc.Gap = sc.Clear*k, sc.Gap*k
		out.Scatter = &sc
	}
	return out
}
