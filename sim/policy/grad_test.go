package policy

import (
	"math"
	"math/rand"
	"testing"
)

// checkGrad compares analytic gradients with central differences on a
// sample of parameters.
func checkGrad(t *testing.T, name string, params, grads [][]float64, loss func() float64, rng *rand.Rand) {
	t.Helper()
	const h = 1e-6
	for k := 0; k < 60; k++ {
		a := rng.Intn(len(params))
		if len(params[a]) == 0 {
			continue
		}
		j := rng.Intn(len(params[a]))
		x := params[a][j]
		params[a][j] = x + h
		lp := loss()
		params[a][j] = x - h
		lm := loss()
		params[a][j] = x
		fd := (lp - lm) / (2 * h)
		if g := grads[a][j]; math.Abs(fd-g) > 1e-6*math.Max(1, math.Abs(fd)) {
			t.Errorf("%s param %d[%d]: analytic %.8f, finite difference %.8f", name, a, j, g, fd)
		}
	}
}

func randObs(rng *rand.Rand, n, np, tok, part, glob, priv int) *Obs {
	o := &Obs{Tokens: randTokens(rng, n, tok), Parts: randTokens(rng, np, part), Global: randTokens(rng, 1, glob)[0]}
	if priv > 0 {
		o.Priv = randTokens(rng, 1, priv)[0]
	}
	return o
}

// shake gives the input statistics non-trivial values so the gradient
// checks also cover normalisation.
func shake(rng *rand.Rand, zs ...*Normalizer) {
	for _, z := range zs {
		for i := range z.Mean {
			z.Mean[i], z.Var[i] = 0.3*rng.NormFloat64(), 0.5+rng.Float64()
		}
		z.Count = 10
	}
}

func TestPolicyGradient(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	p := New(rng, 6, 4, 5, 8, 12)
	shake(rng, p.TokNorm, p.PartNorm, p.GloNorm)
	o := randObs(rng, 4, 3, 6, 4, 5, 0)
	w := randTokens(rng, 1, 4)[0]
	loss := func() float64 {
		s := 0.0
		for i, m := range p.Forward(o).Mu {
			s += w[i] * m
		}
		return s
	}
	g := p.Clone(true)
	p.Backward(p.Forward(o), w, g)
	checkGrad(t, "policy", p.Params(), g.Params(), loss, rng)
}

func TestValueGradient(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	v := NewValue(rng, 6, 4, 5+3, 8, 12)
	shake(rng, v.TokNorm, v.PartNorm, v.GloNorm)
	o := randObs(rng, 3, 5, 6, 4, 5, 3)
	loss := func() float64 { return v.Forward(o).V }
	g := v.Clone(true)
	v.Backward(v.Forward(o), 1, g)
	checkGrad(t, "value", v.Params(), g.Params(), loss, rng)
}

func TestNormalizerMatchesBatchStatistics(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	rows := randTokens(rng, 300, 3)
	for _, r := range rows {
		r[0] = 5 + 2*r[0]
	}
	z := NewNormalizer(3)
	z.Update(rows[:120])
	z.Update(rows[120:])
	all := NewNormalizer(3)
	all.Update(rows)
	for i := range z.Mean {
		if math.Abs(z.Mean[i]-all.Mean[i]) > 1e-9 || math.Abs(z.Var[i]-all.Var[i]) > 1e-9 {
			t.Fatalf("feature %d: merged %v/%v, whole batch %v/%v", i, z.Mean[i], z.Var[i], all.Mean[i], all.Var[i])
		}
	}
}
