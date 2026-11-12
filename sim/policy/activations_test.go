package policy

import (
	"math"
	"math/rand"
	"testing"
)

// plainSet runs net on [row, shared] for every row (no shared-input
// shortcut) and returns the tanh embeddings, the last hidden layers and
// their mean.
func plainSet(net *MLP, rows [][]float64, shared []float64, E int) (emb, hid [][]float64, pool []float64) {
	pool = make([]float64, E)
	for _, r := range rows {
		acts := net.fwd(concat(r, shared))
		h := tanhOf(acts[len(acts)-1])
		for j := range h {
			pool[j] += h[j] / float64(len(rows))
		}
		emb, hid = append(emb, h), append(hid, acts[len(acts)-2])
	}
	return emb, hid, pool
}

func near(t *testing.T, what string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", what, len(got), len(want))
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Fatalf("%s[%d] = %v, want %v", what, i, got[i], want[i])
		}
	}
}

// The exposed activations are what a plain, literal forward pass of the
// Deep Sets network computes, and reading them changes no output.
func TestActivationsMatchPlainForward(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	p := New(rng, 10, 7, 4, 8, 16, 24)
	o := &Obs{Tokens: randTokens(rng, 5, 10), Parts: randTokens(rng, 4, 7), Sensors: randTokens(rng, 3, 4), Global: randTokens(rng, 1, 8)[0]}
	p.UpdateNorms([]*Obs{o, {Tokens: randTokens(rng, 5, 10), Parts: randTokens(rng, 4, 7), Sensors: randTokens(rng, 3, 4), Global: randTokens(rng, 1, 8)[0]}})
	before := p.Act(o)
	a := p.Activations(p.Forward(o))
	near(t, "mu", a.Mu, before)

	g := p.GloNorm.Apply(o.Global)
	near(t, "global", a.Global, g)
	partEmb, _, partPool := plainSet(&p.Part, p.PartNorm.applyAll(o.Parts), g, p.Embed)
	sensEmb, _, sensPool := plainSet(&p.Sensor, p.SensNorm.applyAll(o.Sensors), g, p.Embed)
	g2 := concat(g, partPool, sensPool)
	toks := p.TokNorm.applyAll(o.Tokens)
	tokEmb, phiHid, tokPool := plainSet(&p.Phi, toks, g2, p.Embed)
	near(t, "partPool", a.PartPool, partPool)
	near(t, "sensorPool", a.SensorPool, sensPool)
	near(t, "tokenPool", a.TokenPool, tokPool)
	for i := range o.Parts {
		near(t, "parts", a.Parts[i], p.PartNorm.Apply(o.Parts[i]))
		near(t, "partEmb", a.PartEmb[i], partEmb[i])
	}
	for i := range o.Sensors {
		near(t, "sensors", a.Sensors[i], p.SensNorm.Apply(o.Sensors[i]))
		near(t, "sensorEmb", a.SensorEmb[i], sensEmb[i])
	}
	for i := range o.Tokens {
		near(t, "tokens", a.Tokens[i], toks[i])
		near(t, "phiHid", a.PhiHid[i], phiHid[i])
		near(t, "tokenEmb", a.TokenEmb[i], tokEmb[i])
		rho := p.Rho.fwd(concat(tokEmb[i], tokPool, g2))
		if len(a.RhoHid[i]) != len(rho)-2 {
			t.Fatalf("actuator %d: %d hidden layers, want %d", i, len(a.RhoHid[i]), len(rho)-2)
		}
		for l := range a.RhoHid[i] {
			near(t, "rhoHid", a.RhoHid[i][l], rho[l+1])
		}
		pre := rho[len(rho)-1][0]
		near(t, "pre/mu", []float64{a.Pre[i], a.Mu[i]}, []float64{pre, math.Tanh(pre)})
	}
	if a.LogStd != p.LogStd[0] {
		t.Fatalf("logStd %v, want %v", a.LogStd, p.LogStd[0])
	}
	near(t, "mu after", p.Act(o), before)
}
