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

func TestPolicyGradient(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	p := New(rng, 6, 5, 8, 12)
	tokens, global := randTokens(rng, 4, 6), randTokens(rng, 1, 5)[0]
	w := randTokens(rng, 1, 4)[0]
	loss := func() float64 {
		s := 0.0
		for i, m := range p.Forward(tokens, global).Mu {
			s += w[i] * m
		}
		return s
	}
	g := p.Clone(true)
	p.Backward(p.Forward(tokens, global), w, g)
	checkGrad(t, "policy", p.Params(), g.Params(), loss, rng)

	// Forward must agree with Act.
	act, mu := p.Act(tokens, global), p.Forward(tokens, global).Mu
	for i := range act {
		if math.Abs(act[i]-mu[i]) > 1e-12 {
			t.Fatalf("Forward %v differs from Act %v", mu, act)
		}
	}
}

func TestValueGradient(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	v := NewValue(rng, 6, 5, 8, 12)
	tokens, global := randTokens(rng, 3, 6), randTokens(rng, 1, 5)[0]
	loss := func() float64 { return v.Forward(tokens, global).V }
	g := v.Clone(true)
	v.Backward(v.Forward(tokens, global), 1, g)
	checkGrad(t, "value", v.Params(), g.Params(), loss, rng)
}
