package env

import (
	"math"
	"math/rand"

	"mars/sim/model"
)

// roughen gives a floor random bumps up to maxBump high (a few long
// swells and some short ripples), a random softness (gym mat to concrete)
// and patches of better and worse grip, so the feet have something to
// feel and the policy learns to read it.
func roughen(g *model.Ground, rng *rand.Rand, maxBump float64) {
	u := func(lo, hi float64) float64 { return lo + rng.Float64()*(hi-lo) }
	wave := func(amp, wavelength float64) model.Wave {
		a := rng.Float64() * 2 * math.Pi
		k := 2 * math.Pi / wavelength
		return model.Wave{A: amp, Kx: k * math.Cos(a), Kz: k * math.Sin(a), Phase: rng.Float64() * 2 * math.Pi}
	}
	total := u(0, maxBump)
	for i := 0; i < 3; i++ {
		g.Bumps = append(g.Bumps, wave(total/3*u(0.5, 1.5), u(0.25, 2)))
	}
	soft := u(0.25, 1.5)
	g.K *= soft
	g.C *= math.Sqrt(soft)
	for i := 0; i < 2; i++ {
		g.MuWaves = append(g.MuWaves, wave(u(0, 0.2), u(0.5, 3)))
	}
}

// senseNoise makes the sole pads and tendon gauges behave like cheap
// sensors: some are dead for the whole episode, each reads with its own
// gain and offset error, all lag behind the truth and jitter. A policy
// trained this way does not need lab-grade sensors.
type senseNoise struct {
	dead   []bool
	gain   []float64
	offset []float64
	alpha  float64     // low-pass factor per policy step
	state  [][]float64 // filtered readings
}

// Features of a sensor token that are readings (the rest is geometry).
var readings = []int{5, 6, 7, 8, 9}

func newSenseNoise(rng *rand.Rand, n int) senseNoise {
	tau := 0.005 + 0.025*rng.Float64()
	s := senseNoise{alpha: 1 - math.Exp(-PolicyDt/tau)}
	for i := 0; i < n; i++ {
		s.dead = append(s.dead, rng.Float64() < 0.15)
		s.gain = append(s.gain, 0.75+0.5*rng.Float64())
		s.offset = append(s.offset, 0.04*(rng.Float64()-0.5))
	}
	s.state = make([][]float64, n)
	return s
}

// apply returns the sensor tokens as the robot's cheap sensors would
// report them, without the dead ones.
func (s *senseNoise) apply(rows [][]float64, rng *rand.Rand) [][]float64 {
	var out [][]float64
	for i, r := range rows {
		if i >= len(s.dead) {
			out = append(out, r)
			continue
		}
		if s.state[i] == nil {
			s.state[i] = make([]float64, len(r))
			copy(s.state[i], r)
		}
		for _, f := range readings {
			s.state[i][f] += s.alpha * (r[f] - s.state[i][f])
		}
		if s.dead[i] {
			continue
		}
		y := append([]float64(nil), r...)
		for _, f := range readings {
			y[f] = s.gain[i]*s.state[i][f] + 0.01*rng.NormFloat64()
		}
		y[5] = math.Max(0, y[5]+s.offset[i])
		out = append(out, y)
	}
	return out
}
