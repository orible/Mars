package policy

import (
	"math"
	"math/rand"
)

// This file holds what training needs: forward passes that keep their
// intermediate values, hand-written backpropagation, the value network,
// and the Adam optimiser. Everything is plain float64 loops; the networks
// are small, and rollouts and gradients run on many goroutines.

// fwd runs the MLP and returns every layer's input plus the final linear
// output: acts[0] is x, acts[i] (0 < i < len) is layer i-1's output after
// tanh, and the last entry is the linear output.
func (m *MLP) fwd(x []float64) [][]float64 {
	acts := make([][]float64, 0, len(m.Layers)+1)
	acts = append(acts, x)
	for i := range m.Layers {
		y := m.Layers[i].forward(x)
		if i < len(m.Layers)-1 {
			for j := range y {
				y[j] = math.Tanh(y[j])
			}
		}
		acts = append(acts, y)
		x = y
	}
	return acts
}

// bwd backpropagates dout (gradient of the loss with respect to the linear
// output) through the MLP, accumulating weight gradients into g, and
// returns the gradient with respect to the input.
func (m *MLP) bwd(acts [][]float64, dout []float64, g *MLP) []float64 {
	last := len(m.Layers) - 1
	d := append([]float64(nil), dout...)
	for i := last; i >= 0; i-- {
		if i < last {
			for j := range d {
				y := acts[i+1][j]
				d[j] *= 1 - y*y
			}
		}
		L, G, x := &m.Layers[i], &g.Layers[i], acts[i]
		dx := make([]float64, L.In)
		for o := 0; o < L.Out; o++ {
			do := d[o]
			if do == 0 {
				continue
			}
			G.B[o] += do
			row, grow := L.W[o*L.In:(o+1)*L.In], G.W[o*L.In:(o+1)*L.In]
			for k, xk := range x {
				grow[k] += do * xk
				dx[k] += row[k] * do
			}
		}
		d = dx
	}
	return d
}

func (m *MLP) clone(zero bool) MLP {
	c := MLP{Layers: make([]Dense, len(m.Layers))}
	for i, l := range m.Layers {
		c.Layers[i] = Dense{In: l.In, Out: l.Out, W: make([]float64, len(l.W)), B: make([]float64, len(l.B))}
		if !zero {
			copy(c.Layers[i].W, l.W)
			copy(c.Layers[i].B, l.B)
		}
	}
	return c
}

func (m *MLP) params() [][]float64 {
	var p [][]float64
	for i := range m.Layers {
		p = append(p, m.Layers[i].W, m.Layers[i].B)
	}
	return p
}

// DefaultLogStd is the initial exploration noise: σ ≈ 0.5 of full valve
// opening.
const DefaultLogStd = -0.7

// Std is the exploration noise on each valve command.
func (p *SetPolicy) Std() float64 { return math.Exp(p.LogStd[0]) }

// Clone returns a deep copy (zero = true gives a zeroed copy for
// gradients).
func (p *SetPolicy) Clone(zero bool) *SetPolicy {
	c := &SetPolicy{TokenDim: p.TokenDim, GlobalDim: p.GlobalDim, Embed: p.Embed,
		Phi: p.Phi.clone(zero), Rho: p.Rho.clone(zero), LogStd: []float64{0}}
	if !zero {
		c.LogStd[0] = p.LogStd[0]
	}
	return c
}

// Params lists every trainable array, in a fixed order shared by Clone.
func (p *SetPolicy) Params() [][]float64 {
	return append(append(p.Phi.params(), p.Rho.params()...), p.LogStd)
}

// PolicyCache holds one forward pass for backpropagation.
type PolicyCache struct {
	phi, rho [][][]float64
	h        [][]float64
	Mu       []float64 // mean command per actuator, in [-1, 1]
}

// Forward returns the mean command per actuator and a cache for Backward.
func (p *SetPolicy) Forward(tokens [][]float64, global []float64) *PolicyCache {
	n := len(tokens)
	c := &PolicyCache{phi: make([][][]float64, n), rho: make([][][]float64, n), h: make([][]float64, n), Mu: make([]float64, n)}
	ctx := make([]float64, p.Embed)
	for i, tok := range tokens {
		in := append(append(make([]float64, 0, p.TokenDim+p.GlobalDim), tok...), global...)
		c.phi[i] = p.Phi.fwd(in)
		e := c.phi[i][len(c.phi[i])-1]
		h := make([]float64, len(e))
		for j := range e {
			h[j] = math.Tanh(e[j])
			ctx[j] += h[j] / float64(n)
		}
		c.h[i] = h
	}
	for i := range tokens {
		in := make([]float64, 0, 2*p.Embed+p.GlobalDim)
		in = append(append(append(in, c.h[i]...), ctx...), global...)
		c.rho[i] = p.Rho.fwd(in)
		c.Mu[i] = math.Tanh(c.rho[i][len(c.rho[i])-1][0])
	}
	return c
}

// Backward accumulates into g the gradient for dMu (loss gradient with
// respect to each actuator's mean command).
func (p *SetPolicy) Backward(c *PolicyCache, dMu []float64, g *SetPolicy) {
	n := len(c.Mu)
	E := p.Embed
	dh := make([][]float64, n)
	dctx := make([]float64, E)
	for i := 0; i < n; i++ {
		dm := dMu[i] * (1 - c.Mu[i]*c.Mu[i])
		din := p.Rho.bwd(c.rho[i], []float64{dm}, &g.Rho)
		dh[i] = din[:E]
		for j := 0; j < E; j++ {
			dctx[j] += din[E+j]
		}
	}
	for i := 0; i < n; i++ {
		de := make([]float64, E)
		for j := 0; j < E; j++ {
			h := c.h[i][j]
			de[j] = (dh[i][j] + dctx[j]/float64(n)) * (1 - h*h)
		}
		p.Phi.bwd(c.phi[i], de, &g.Phi)
	}
}

// SetValue is the critic: a Deep Sets network over the same tokens that
// estimates the return. Only training uses it.
type SetValue struct {
	TokenDim  int `json:"tokenDim"`
	GlobalDim int `json:"globalDim"`
	Embed     int `json:"embed"`
	Phi       MLP `json:"phi"` // [token, global] -> embedding
	Rho       MLP `json:"rho"` // [mean embedding, global] -> value
}

// ValueScale maps the network's output to returns (rewards ≈ 1 per step).
const ValueScale = 20

func NewValue(rng *rand.Rand, tokenDim, globalDim, embed, hidden int) *SetValue {
	return &SetValue{TokenDim: tokenDim, GlobalDim: globalDim, Embed: embed,
		Phi: newMLP(rng, tokenDim+globalDim, hidden, embed),
		Rho: newMLP(rng, embed+globalDim, hidden, 1)}
}

func (v *SetValue) Clone(zero bool) *SetValue {
	return &SetValue{TokenDim: v.TokenDim, GlobalDim: v.GlobalDim, Embed: v.Embed, Phi: v.Phi.clone(zero), Rho: v.Rho.clone(zero)}
}

func (v *SetValue) Params() [][]float64 { return append(v.Phi.params(), v.Rho.params()...) }

type ValueCache struct {
	phi [][][]float64
	h   [][]float64
	rho [][]float64
	V   float64
}

func (v *SetValue) Forward(tokens [][]float64, global []float64) *ValueCache {
	n := len(tokens)
	c := &ValueCache{phi: make([][][]float64, n), h: make([][]float64, n)}
	pool := make([]float64, v.Embed)
	for i, tok := range tokens {
		in := append(append(make([]float64, 0, v.TokenDim+v.GlobalDim), tok...), global...)
		c.phi[i] = v.Phi.fwd(in)
		e := c.phi[i][len(c.phi[i])-1]
		h := make([]float64, len(e))
		for j := range e {
			h[j] = math.Tanh(e[j])
			pool[j] += h[j] / float64(n)
		}
		c.h[i] = h
	}
	c.rho = v.Rho.fwd(append(pool, global...))
	c.V = ValueScale * c.rho[len(c.rho)-1][0]
	return c
}

// Backward accumulates into g the gradient for dV (loss gradient with
// respect to the value).
func (v *SetValue) Backward(c *ValueCache, dV float64, g *SetValue) {
	n := len(c.h)
	din := v.Rho.bwd(c.rho, []float64{dV * ValueScale}, &g.Rho)
	for i := 0; i < n; i++ {
		de := make([]float64, v.Embed)
		for j := range de {
			h := c.h[i][j]
			de[j] = din[j] / float64(n) * (1 - h*h)
		}
		v.Phi.bwd(c.phi[i], de, &g.Phi)
	}
}

// Adam is the Adam optimiser over a fixed list of parameter arrays.
type Adam struct {
	LR, B1, B2, Eps float64
	m, v            [][]float64
	t               int
}

func NewAdam(params [][]float64, lr float64) *Adam {
	a := &Adam{LR: lr, B1: 0.9, B2: 0.999, Eps: 1e-8}
	for _, p := range params {
		a.m = append(a.m, make([]float64, len(p)))
		a.v = append(a.v, make([]float64, len(p)))
	}
	return a
}

// Step applies grads to params, first scaling grads down so their global
// norm is at most maxNorm (if maxNorm > 0). It returns the norm before
// clipping.
func (a *Adam) Step(params, grads [][]float64, maxNorm float64) float64 {
	norm := 0.0
	for _, g := range grads {
		for _, x := range g {
			norm += x * x
		}
	}
	norm = math.Sqrt(norm)
	scale := 1.0
	if maxNorm > 0 && norm > maxNorm {
		scale = maxNorm / norm
	}
	a.t++
	c1, c2 := 1-math.Pow(a.B1, float64(a.t)), 1-math.Pow(a.B2, float64(a.t))
	for i, p := range params {
		m, v, g := a.m[i], a.v[i], grads[i]
		for j := range p {
			gj := g[j] * scale
			m[j] = a.B1*m[j] + (1-a.B1)*gj
			v[j] = a.B2*v[j] + (1-a.B2)*gj*gj
			p[j] -= a.LR * (m[j] / c1) / (math.Sqrt(v[j]/c2) + a.Eps)
		}
	}
	return norm
}

// Zero clears gradient arrays.
func Zero(grads [][]float64) {
	for _, g := range grads {
		for j := range g {
			g[j] = 0
		}
	}
}

// AddInto adds src arrays into dst arrays of the same shapes.
func AddInto(dst, src [][]float64) {
	for i := range dst {
		for j := range dst[i] {
			dst[i][j] += src[i][j]
		}
	}
}
