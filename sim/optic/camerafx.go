package optic

import (
	"math"
	"math/rand"
)

// SensorConfig is the procedural camera: optics, shutter, sensor and ISP.
// Everything is cheap per-pixel or 8×8-block work.
type SensorConfig struct {
	// Shutter: the fraction of the frame interval the shutter is open
	// (motion blur length) and the rolling-shutter readout time, as a
	// fraction of the frame interval from the top row to the bottom row.
	Exposure, Readout float32
	BlurTaps          int // samples along the motion path
	// Optics: Gaussian PSF sigma (px), vignetting at the corners (0..1),
	// lateral chromatic aberration (px at the corners).
	PSF, Vignette, ChromAb float32
	// Auto-exposure: target mean (linear), adaptation rate per frame,
	// and the size of a slow random drift of the target (exposure
	// hunting).
	AETarget, AERate, AEDrift float32
	// Sensor: full-well electrons at gain 1, read noise (e⁻), dark offset
	// non-uniformity (e⁻), photo-response non-uniformity (fraction), ADC
	// bits.
	FullWell, ReadNoise, DSNU, PRNU float32
	Bits                            int
	// ISP: white balance error (fraction per channel), unsharp-mask
	// sharpening amount, JPEG quality (0 disables compression).
	WBError float32
	Sharpen float32
	Quality int
}

// DefaultSensor is a small, cheap global-noise CMOS module behind a
// phone-grade ISP.
func DefaultSensor() SensorConfig {
	return SensorConfig{Exposure: 0.3, Readout: 0.5, BlurTaps: 4,
		PSF: 0.45, Vignette: 0.35, ChromAb: 0.35,
		AETarget: 0.18, AERate: 0.15, AEDrift: 0.08,
		FullWell: 1500, ReadNoise: 3, DSNU: 1.5, PRNU: 0.012, Bits: 10,
		WBError: 0.04, Sharpen: 0.35, Quality: 70}
}

// Exposure is the auto-exposure state the two eyes of a rig share (stereo
// modules sync exposure so both eyes see the same brightness).
type Exposure struct {
	Gain   float32 // multiplier on scene radiance
	target float32
	drift  float32
}

// Step adapts the gain towards the target from the metered mean.
func (e *Exposure) Step(cfg SensorConfig, mean float32, rng *rand.Rand) {
	if e.Gain == 0 {
		e.Gain, e.target = cfg.AETarget/float32(math.Max(float64(mean), 1e-4)), cfg.AETarget
	}
	e.drift = 0.95*e.drift + cfg.AEDrift*0.3*float32(rng.NormFloat64())
	target := cfg.AETarget * float32(math.Exp(float64(e.drift)))
	err := float32(math.Log(float64(target) / math.Max(float64(mean*e.Gain), 1e-6)))
	e.Gain *= float32(math.Exp(float64(cfg.AERate * err)))
}

// Sensor is one eye's camera: fixed-pattern noise and white balance are
// drawn once per sensor; shot and read noise are fresh every frame, as
// on a real sensor.
type Sensor struct {
	Cfg        SensorConfig
	W, H       int
	prnu, dsnu []float32
	wb         [3]float32
	rng        *rand.Rand
	a, b       []RGB // scratch
}

// NewSensor makes a w×h sensor; the seed fixes its fixed-pattern noise.
func NewSensor(cfg SensorConfig, w, h int, seed int64) *Sensor {
	s := &Sensor{Cfg: cfg, W: w, H: h, rng: rand.New(rand.NewSource(seed)),
		prnu: make([]float32, w*h*3), dsnu: make([]float32, w*h*3), a: make([]RGB, w*h), b: make([]RGB, w*h)}
	for i := range s.prnu {
		s.prnu[i] = 1 + cfg.PRNU*float32(s.rng.NormFloat64())
		s.dsnu[i] = cfg.DSNU * float32(s.rng.NormFloat64())
	}
	for c := range s.wb {
		s.wb[c] = 1 + cfg.WBError*float32(2*s.rng.Float64()-1)
	}
	return s
}

// Meter is the scene mean the auto-exposure sees (centre-weighted).
func Meter(img []RGB, w, h int) float32 {
	var s, n float32
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dx, dy := (float32(x)+0.5)/float32(w)-0.5, (float32(y)+0.5)/float32(h)-0.5
			wt := 1 - 1.2*(dx*dx+dy*dy)
			s += wt * luma(img[y*w+x])
			n += wt
		}
	}
	return s / n
}

// Optics applies motion blur and rolling shutter along the exact
// backward flow (frame t → t−1, i.e. −1 frame of motion), then the lens
// (chromatic aberration, PSF, vignetting). Rows are read out top to
// bottom, centred on the frame time, and each row integrates over the
// open shutter, so a pixel at row y sees the image at frame times
// s ∈ (y/H − ½)·Readout ± Exposure/2, approximated by shifting the
// rendered frame along the flow (content moves by −back per frame).
// Invalid flow (disocclusions) uses no motion.
func (s *Sensor) Optics(rad []RGB, back []Flow, out []RGB) {
	c := s.Cfg
	w, h := s.W, s.H
	taps := max(1, c.BlurTaps)
	tmp := s.a
	for y := 0; y < h; y++ {
		row := (float32(y)+0.5)/float32(h) - 0.5
		for x := 0; x < w; x++ {
			k := y*w + x
			var dx, dy float32
			if back != nil && back[k].Valid {
				dx, dy = back[k].DX, back[k].DY
			}
			var acc RGB
			for i := 0; i < taps; i++ {
				ts := row*c.Readout + c.Exposure*((float32(i)+0.5)/float32(taps)-0.5)
				// Content at time t+ts: I_t(p − v·ts) with v = −back.
				acc = acc.add(bilinearRGB(rad, w, h, float32(x)+0.5+dx*ts, float32(y)+0.5+dy*ts))
			}
			tmp[k] = acc.scale(1 / float32(taps))
		}
	}
	// Lens: lateral chromatic aberration (red out, blue in), then the PSF.
	cx, cy := float32(w)/2, float32(h)/2
	rmax := float32(math.Hypot(float64(cx), float64(cy)))
	ab := s.b
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			k := y*w + x
			px, py := float32(x)+0.5-cx, float32(y)+0.5-cy
			sc := c.ChromAb / rmax // shift of ChromAb·r/rmax px along the radius
			red := bilinearRGB(tmp, w, h, cx+px*(1+sc), cy+py*(1+sc))
			blue := bilinearRGB(tmp, w, h, cx+px*(1-sc), cy+py*(1-sc))
			ab[k] = RGB{red[0], tmp[k][1], blue[2]}
		}
	}
	gaussBlur(ab, out, w, h, c.PSF)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			px, py := (float32(x)+0.5)/float32(w)-0.5, (float32(y)+0.5)/float32(h)-0.5
			r2 := (px*px + py*py) * 2 // 1 at the corners
			k := y*w + x
			out[k] = out[k].scale(1 - c.Vignette*r2*r2)
		}
	}
}

// Capture turns lens-plane radiance into the sensor's output image
// (sRGB-encoded, 0..1): exposure gain, white balance, photon shot noise,
// fixed-pattern and read noise, ADC quantisation, tone curve,
// sharpening and JPEG-style compression.
func (s *Sensor) Capture(img []RGB, gain float32, out []RGB) {
	c := s.Cfg
	levels := float32(int(1)<<c.Bits - 1)
	fw := c.FullWell
	// Higher gain means fewer photons per output level: more shot noise.
	e := fw / float32(math.Max(1, float64(gain)/0.5))
	for k, v := range img {
		var o RGB
		for ch := 0; ch < 3; ch++ {
			x := v[ch] * gain * s.wb[ch]
			el := float32(math.Max(0, float64(x*e*s.prnu[3*k+ch])))
			el += float32(math.Sqrt(float64(el))) * float32(s.rng.NormFloat64())
			el += c.ReadNoise*float32(s.rng.NormFloat64()) + s.dsnu[3*k+ch]
			q := float32(math.Round(float64(clamp01(el/e) * levels)))
			o[ch] = q / levels
		}
		out[k] = o
	}
	for k := range out {
		for ch := 0; ch < 3; ch++ {
			out[k][ch] = toneCurve(out[k][ch])
		}
	}
	if c.Sharpen > 0 {
		gaussBlur(out, s.a, s.W, s.H, 1)
		for k := range out {
			for ch := 0; ch < 3; ch++ {
				out[k][ch] = clamp01(out[k][ch] + c.Sharpen*(out[k][ch]-s.a[k][ch]))
			}
		}
	}
	if c.Quality > 0 {
		JPEG(out, s.W, s.H, c.Quality)
	}
}

// toneCurve is a filmic shoulder followed by the sRGB transfer curve.
func toneCurve(x float32) float32 {
	x = x * 1.6 / (1 + x*0.6) // soft highlight roll-off, 1 → 1
	return srgb(x)
}

// srgb is the sRGB transfer curve, from a 4096-entry table with linear
// interpolation.
func srgb(x float32) float32 {
	x = clamp01(x) * (srgbN - 1)
	i := int(x)
	if i >= srgbN-1 {
		return srgbLUT[srgbN-1]
	}
	f := x - float32(i)
	return srgbLUT[i]*(1-f) + srgbLUT[i+1]*f
}

const srgbN = 4096

var srgbLUT = func() (t [srgbN]float32) {
	for i := range t {
		x := float64(i) / (srgbN - 1)
		if x <= 0.0031308 {
			t[i] = float32(12.92 * x)
		} else {
			t[i] = float32(1.055*math.Pow(x, 1/2.4) - 0.055)
		}
	}
	return
}()

func luma(c RGB) float32 { return 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2] }

// gaussBlur is a separable Gaussian of std sigma (radius 2σ, at least 1).
func gaussBlur(src, dst []RGB, w, h int, sigma float32) {
	if sigma <= 0 {
		copy(dst, src)
		return
	}
	r := max(1, int(math.Ceil(float64(2*sigma))))
	k := make([]float32, 2*r+1)
	var sum float32
	for i := -r; i <= r; i++ {
		k[i+r] = float32(math.Exp(-float64(i*i) / (2 * float64(sigma*sigma))))
		sum += k[i+r]
	}
	for i := range k {
		k[i] /= sum
	}
	tmp := make([]RGB, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var a RGB
			for i := -r; i <= r; i++ {
				a = a.add(src[y*w+clampi(x+i, 0, w-1)].scale(k[i+r]))
			}
			tmp[y*w+x] = a
		}
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var a RGB
			for i := -r; i <= r; i++ {
				a = a.add(tmp[clampi(y+i, 0, h-1)*w+x].scale(k[i+r]))
			}
			dst[y*w+x] = a
		}
	}
}

// Stereo is the rig's camera pair with shared auto-exposure.
type Stereo struct {
	Exp          Exposure
	Left, Right  *Sensor
	rng          *rand.Rand
	lensL, lensR []RGB
}

// NewStereo makes a synced pair of sensors.
func NewStereo(cfg SensorConfig, w, h int, seed int64) *Stereo {
	return &Stereo{Left: NewSensor(cfg, w, h, seed), Right: NewSensor(cfg, w, h, seed+7919),
		rng: rand.New(rand.NewSource(seed + 104729)), lensL: make([]RGB, w*h), lensR: make([]RGB, w*h)}
}

// Process runs both eyes: optics with each eye's backward flow (nil for
// the first frame), shared exposure metered on the left eye, then each
// sensor. outL and outR receive sRGB images.
func (st *Stereo) Process(radL, radR []RGB, backL, backR []Flow, outL, outR []RGB) {
	st.Left.Optics(radL, backL, st.lensL)
	st.Right.Optics(radR, backR, st.lensR)
	st.Exp.Step(st.Left.Cfg, Meter(st.lensL, st.Left.W, st.Left.H), st.rng)
	st.Left.Capture(st.lensL, st.Exp.Gain, outL)
	st.Right.Capture(st.lensR, st.Exp.Gain, outR)
}
