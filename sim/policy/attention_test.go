package policy

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

// Adding attention to a trained policy changes none of its outputs (Wo
// starts at zero), so a run can resume with attention from any checkpoint.
func TestAttentionStartsAsIdentity(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	p := New(rng, 6, 4, 3, 5, 8, 12)
	o := randObs(rng, 5, 3, 6, 4, 5, 0)
	before := p.Act(o)
	p.AddAttention(rng, 2, 4)
	near(t, "mu", p.Act(o), before)
}

func TestAttentionPolicyGradient(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	p := New(rng, 6, 4, 3, 5, 8, 12)
	p.AddAttention(rng, 2, 3)
	for i := range p.Attn.Wo {
		p.Attn.Wo[i] = 0.4 * rng.NormFloat64() // trained-looking, so every path carries gradient
	}
	shake(rng, p.TokNorm, p.PartNorm, p.SensNorm, p.GloNorm)
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
	checkGrad(t, "attention policy", p.Params(), g.Params(), loss, rng)
	// The sample above is random; check the attention weights explicitly.
	n := len(p.Params())
	checkGrad(t, "attention weights", p.Params()[n-4:], g.Params()[n-4:], loss, rng)
}

// Attention keeps the network permutation-equivariant over actuators.
func TestAttentionPermutationEquivariant(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	p := New(rng, 6, 4, 3, 5, 8, 12)
	p.AddAttention(rng, 2, 4)
	for i := range p.Attn.Wo {
		p.Attn.Wo[i] = 0.4 * rng.NormFloat64()
	}
	o := randObs(rng, 5, 3, 6, 4, 5, 0)
	out := p.Act(o)
	perm := []int{3, 0, 4, 1, 2}
	sh := *o
	sh.Tokens = make([][]float64, len(o.Tokens))
	for i, j := range perm {
		sh.Tokens[i] = o.Tokens[j]
	}
	got := p.Act(&sh)
	for i, j := range perm {
		if math.Abs(got[i]-out[j]) > 1e-12 {
			t.Fatalf("actuator %d: %v != %v", j, got[i], out[j])
		}
	}
}

func TestAttentionCheckpointRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(10))
	p := New(rng, 6, 4, 3, 5, 8, 12)
	p.AddAttention(rng, 2, 4)
	p.Attn.Wo[3] = 0.7
	var buf bytes.Buffer
	if err := p.Save(&buf); err != nil {
		t.Fatal(err)
	}
	q, err := Load(&buf)
	if err != nil {
		t.Fatal(err)
	}
	o := randObs(rng, 4, 3, 6, 4, 5, 0)
	near(t, "mu", q.Act(o), p.Act(o))
}

// The view's attention weights are a softmax per listener, and Mixed is
// what ρ reads.
func TestAttentionActivations(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	p := New(rng, 6, 4, 3, 5, 8, 12)
	p.AddAttention(rng, 2, 4)
	for i := range p.Attn.Wo {
		p.Attn.Wo[i] = 0.4 * rng.NormFloat64()
	}
	o := randObs(rng, 4, 3, 6, 4, 5, 0)
	a := p.Activations(p.Forward(o))
	if len(a.Attn) != 2 || len(a.Attn[0]) != 4 || len(a.AttnGain) != 4 {
		t.Fatalf("attention shapes %d heads, %d rows, %d gains", len(a.Attn), len(a.Attn[0]), len(a.AttnGain))
	}
	for _, head := range a.Attn {
		for _, row := range head {
			s := 0.0
			for _, v := range row {
				s += v
			}
			if math.Abs(s-1) > 1e-12 {
				t.Fatalf("attention row sums to %v", s)
			}
		}
	}
	for i := range a.Mixed {
		rho := p.Rho.fwd(concat(a.Mixed[i], a.TokenPool, a.Global, a.PartPool, a.SensorPool))
		near(t, "pre", []float64{a.Pre[i]}, []float64{rho[len(rho)-1][0]})
	}
}
