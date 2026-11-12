package policy

import (
	"math/rand"
	"testing"
)

// Forward plus backward at the trainer's sizes (22 cylinders, 17 parts,
// 20 sensors, embed 64, hidden 128), with and without attention.
func benchPolicy(b *testing.B, attn bool) {
	rng := rand.New(rand.NewSource(1))
	p := New(rng, 20, 19, 11, 19, 64, 128)
	if attn {
		p.AddAttention(rng, 4, 8)
	}
	o := &Obs{Tokens: randTokens(rng, 22, 20), Parts: randTokens(rng, 17, 19), Sensors: randTokens(rng, 20, 11), Global: randTokens(rng, 1, 19)[0]}
	g := p.Clone(true)
	d := make([]float64, 22)
	for i := range d {
		d[i] = 0.1
	}
	if attn {
		for i := range p.Attn.Wo {
			p.Attn.Wo[i] = 0.01 // trained, so every gradient path is live
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.Backward(p.Forward(o), d, g)
	}
}

func BenchmarkPolicyPlain(b *testing.B)     { benchPolicy(b, false) }
func BenchmarkPolicyAttention(b *testing.B) { benchPolicy(b, true) }
