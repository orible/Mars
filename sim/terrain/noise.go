package terrain

import (
	"math"
	"math/rand"
)

// Noise is fractal (octave) Perlin noise: Octaves layers of gradient
// noise, each Lacunarity times finer than the last and Persistence times
// as tall. It is smooth (quintic fade, so the slope and curvature are
// continuous) and its gradient is exact, so contacts get a true normal.
//
// The height never exceeds ±Amplitude: the layers are normalised so their
// weights sum to one, and each layer is scaled to about ±1.
type Noise struct {
	Amplitude   float64 `json:"amplitude"`            // m, largest height above or below 0
	Wavelength  float64 `json:"wavelength"`           // m, of the coarsest octave
	Octaves     int     `json:"octaves"`              // layers (1 = one smooth swell)
	Persistence float64 `json:"persistence"`          // amplitude ratio between octaves (roughness, 0.3 gentle … 0.7 rough)
	Lacunarity  float64 `json:"lacunarity,omitempty"` // frequency ratio between octaves (default 2)
}

// NoisePresets run from a floor that is only slightly uneven to a rough
// field. Scale Amplitude to sweep difficulty.
var NoisePresets = map[string]Noise{
	"gentle":   {Amplitude: 0.02, Wavelength: 3, Octaves: 3, Persistence: 0.4},
	"uneven":   {Amplitude: 0.04, Wavelength: 2.5, Octaves: 4, Persistence: 0.45},
	"bumpy":    {Amplitude: 0.07, Wavelength: 2, Octaves: 5, Persistence: 0.5},
	"rough":    {Amplitude: 0.12, Wavelength: 2, Octaves: 5, Persistence: 0.55},
	"hillside": {Amplitude: 0.35, Wavelength: 8, Octaves: 4, Persistence: 0.4},
}

// noiseField is a compiled Noise.
type noiseField struct {
	oct  []octave
	norm float64 // Amplitude / Σ weights
}

type octave struct {
	perm   [512]uint8
	freq   float64 // 1/m
	weight float64
	dx, dz float64 // offset so octaves do not share lattice points
}

func newNoiseField(n Noise, seed int64) *noiseField {
	lac := n.Lacunarity
	if lac <= 0 {
		lac = 2
	}
	oc := max(1, n.Octaves)
	f := &noiseField{}
	rng := rand.New(rand.NewSource(seed))
	w, freq, sum := 1.0, 1/n.Wavelength, 0.0
	for i := 0; i < oc; i++ {
		o := octave{freq: freq, weight: w, dx: rng.Float64() * 256, dz: rng.Float64() * 256}
		p := rng.Perm(256)
		for j := 0; j < 512; j++ {
			o.perm[j] = uint8(p[j&255])
		}
		f.oct = append(f.oct, o)
		sum += w
		w *= n.Persistence
		freq *= lac
	}
	f.norm = n.Amplitude / sum
	return f
}

// eval returns the height and its slope (∂h/∂x, ∂h/∂z).
func (f *noiseField) eval(x, z float64) (h, gx, gz float64) {
	for i := range f.oct {
		o := &f.oct[i]
		n, nx, nz := o.perlin(x*o.freq+o.dx, z*o.freq+o.dz)
		h += o.weight * n
		gx += o.weight * o.freq * nx
		gz += o.weight * o.freq * nz
	}
	return h * f.norm, gx * f.norm, gz * f.norm
}

// Eight unit gradient directions.
var grads = func() (g [8][2]float64) {
	for i := range g {
		a := float64(i) * math.Pi / 4
		g[i] = [2]float64{math.Cos(a), math.Sin(a)}
	}
	return
}()

// perlin is 2D gradient noise scaled to about ±1, with its derivatives.
func (o *octave) perlin(x, y float64) (n, dx, dy float64) {
	fx, fy := math.Floor(x), math.Floor(y)
	xf, yf := x-fx, y-fy
	xi, yi := int(fx)&255, int(fy)&255
	g := func(i, j int) [2]float64 { return grads[o.perm[int(o.perm[i])+j]&7] }
	ga, gb, gc, gd := g(xi, yi), g(xi+1, yi), g(xi, yi+1), g(xi+1, yi+1)
	a := ga[0]*xf + ga[1]*yf
	b := gb[0]*(xf-1) + gb[1]*yf
	c := gc[0]*xf + gc[1]*(yf-1)
	d := gd[0]*(xf-1) + gd[1]*(yf-1)
	u, du := fade(xf)
	v, dv := fade(yf)
	k := a - b - c + d
	n = a + u*(b-a) + v*(c-a) + u*v*k
	dx = ga[0] + u*(gb[0]-ga[0]) + v*(gc[0]-ga[0]) + u*v*(ga[0]-gb[0]-gc[0]+gd[0]) + du*(b-a+v*k)
	dy = ga[1] + u*(gb[1]-ga[1]) + v*(gc[1]-ga[1]) + u*v*(ga[1]-gb[1]-gc[1]+gd[1]) + dv*(c-a+u*k)
	const s = math.Sqrt2 // 2D Perlin with unit gradients stays within ±√2/2
	return n * s, dx * s, dy * s
}

func fade(t float64) (f, df float64) {
	return t * t * t * (t*(t*6-15) + 10), 30 * t * t * (t*(t-2) + 1)
}
