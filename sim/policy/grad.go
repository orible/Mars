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

// Inputs shared by every token (the global vector, the contexts) enter
// each set network's first layer the same way for every token, so their
// share of that layer is computed once per observation: sharedPre gives
// W[:, off:]·s + b, fwdWith runs the network on a token's own features x
// on top of it, bwdWith backpropagates the token part and sums the first
// layer's output gradient into dpre, and sharedBwd turns that sum into
// the shared weights' gradient and the shared input's gradient. Same
// result as feeding [x, s] to every token, a fraction of the work.

func (m *MLP) sharedPre(off int, sh []float64) []float64 {
	L := &m.Layers[0]
	pre := append([]float64(nil), L.B...)
	for o := 0; o < L.Out; o++ {
		row := L.W[o*L.In+off : (o+1)*L.In]
		acc := 0.0
		for k, v := range sh {
			acc += row[k] * v
		}
		pre[o] += acc
	}
	return pre
}

func (m *MLP) fwdWith(x, pre []float64) [][]float64 {
	acts := make([][]float64, 0, len(m.Layers)+1)
	acts = append(acts, x)
	L := &m.Layers[0]
	y := make([]float64, L.Out)
	for o := 0; o < L.Out; o++ {
		row := L.W[o*L.In : o*L.In+len(x)]
		acc := pre[o]
		for k, v := range x {
			acc += row[k] * v
		}
		y[o] = acc
	}
	for i := 0; i < len(m.Layers); i++ {
		if i > 0 {
			y = m.Layers[i].forward(y)
		}
		if i < len(m.Layers)-1 {
			for j := range y {
				y[j] = math.Tanh(y[j])
			}
		}
		acts = append(acts, y)
	}
	return acts
}

func (m *MLP) bwdWith(acts [][]float64, dout []float64, g *MLP, dpre []float64) []float64 {
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
		dx := make([]float64, len(x))
		for o := 0; o < L.Out; o++ {
			do := d[o]
			if do == 0 {
				continue
			}
			row, grow := L.W[o*L.In:o*L.In+len(x)], G.W[o*L.In:o*L.In+len(x)]
			for k, xk := range x {
				grow[k] += do * xk
				dx[k] += row[k] * do
			}
			if i == 0 {
				dpre[o] += do
			} else {
				G.B[o] += do
			}
		}
		d = dx
	}
	return d
}

func (m *MLP) sharedBwd(off int, sh, dpre []float64, g *MLP) []float64 {
	L, G := &m.Layers[0], &g.Layers[0]
	dsh := make([]float64, len(sh))
	for o := 0; o < L.Out; o++ {
		do := dpre[o]
		if do == 0 {
			continue
		}
		G.B[o] += do
		row, grow := L.W[o*L.In+off:(o+1)*L.In], G.W[o*L.In+off:(o+1)*L.In]
		for k, v := range sh {
			grow[k] += do * v
			dsh[k] += row[k] * do
		}
	}
	return dsh
}

func concat(xs ...[]float64) []float64 {
	n := 0
	for _, x := range xs {
		n += len(x)
	}
	out := make([]float64, 0, n)
	for _, x := range xs {
		out = append(out, x...)
	}
	return out
}

// encoding is one pass of the two set encoders: body parts into a
// body-shape context, then actuators into per-actuator embeddings and
// their mean.
type encoding struct {
	g        []float64
	partActs [][][]float64
	ph       [][]float64 // tanh part embeddings
	g2       []float64   // global vector with the body-shape context appended
	phiActs  [][][]float64
	h        [][]float64 // tanh actuator embeddings
	ctx      []float64   // their mean
}

func encode(part, phi *MLP, E int, toks, parts [][]float64, g []float64) *encoding {
	c := &encoding{g: g}
	shape := make([]float64, E)
	var ppre []float64
	if len(parts) > 0 {
		ppre = part.sharedPre(len(parts[0]), g)
	}
	for _, pt := range parts {
		a := part.fwdWith(pt, ppre)
		h := tanhOf(a[len(a)-1])
		for j := range h {
			shape[j] += h[j] / float64(len(parts))
		}
		c.partActs, c.ph = append(c.partActs, a), append(c.ph, h)
	}
	c.g2 = concat(g, shape)
	c.ctx = make([]float64, E)
	var tpre []float64
	if len(toks) > 0 {
		tpre = phi.sharedPre(len(toks[0]), c.g2)
	}
	for _, t := range toks {
		a := phi.fwdWith(t, tpre)
		h := tanhOf(a[len(a)-1])
		for j := range h {
			c.ctx[j] += h[j] / float64(len(toks))
		}
		c.phiActs, c.h = append(c.phiActs, a), append(c.h, h)
	}
	return c
}

func tanhOf(x []float64) []float64 {
	y := make([]float64, len(x))
	for i, v := range x {
		y[i] = math.Tanh(v)
	}
	return y
}

// backward takes the loss gradient with respect to each actuator
// embedding (dh, may be nil), their mean (dctx) and the extended global
// vector (dg2), and accumulates the encoders' weight gradients.
func (c *encoding) backward(part, phi, gPart, gPhi *MLP, E int, dh [][]float64, dctx, dg2 []float64) {
	dg2 = append([]float64(nil), dg2...)
	n := float64(len(c.h))
	if len(c.h) > 0 {
		dpre := make([]float64, phi.Layers[0].Out)
		for i, h := range c.h {
			de := make([]float64, E)
			for j := range de {
				d := dctx[j] / n
				if dh != nil {
					d += dh[i][j]
				}
				de[j] = d * (1 - h[j]*h[j])
			}
			phi.bwdWith(c.phiActs[i], de, gPhi, dpre)
		}
		off := len(c.phiActs[0][0])
		for j, v := range phi.sharedBwd(off, c.g2, dpre, gPhi) {
			dg2[j] += v
		}
	}
	if len(c.ph) == 0 {
		return
	}
	dshape := dg2[len(dg2)-E:]
	np := float64(len(c.ph))
	dpre := make([]float64, part.Layers[0].Out)
	for i, h := range c.ph {
		de := make([]float64, E)
		for j := range de {
			de[j] = dshape[j] / np * (1 - h[j]*h[j])
		}
		part.bwdWith(c.partActs[i], de, gPart, dpre)
	}
	part.sharedBwd(len(c.partActs[0][0]), c.g, dpre, gPart)
}

// DefaultLogStd is the initial exploration noise: σ ≈ 0.5 of full valve
// opening.
const DefaultLogStd = -0.7

// Std is the exploration noise on each valve command.
func (p *SetPolicy) Std() float64 { return math.Exp(p.LogStd[0]) }

// Clone returns a deep copy (zero = true gives a zeroed copy for
// gradients, without input statistics).
func (p *SetPolicy) Clone(zero bool) *SetPolicy {
	c := &SetPolicy{ObsVersion: p.ObsVersion, TokenDim: p.TokenDim, PartDim: p.PartDim, GlobalDim: p.GlobalDim, Embed: p.Embed,
		Part: p.Part.clone(zero), Phi: p.Phi.clone(zero), Rho: p.Rho.clone(zero), LogStd: []float64{0}}
	if !zero {
		c.LogStd[0] = p.LogStd[0]
		c.TokNorm, c.PartNorm, c.GloNorm = p.TokNorm.clone(), p.PartNorm.clone(), p.GloNorm.clone()
	}
	return c
}

// Params lists every trainable array, in a fixed order shared by Clone.
func (p *SetPolicy) Params() [][]float64 {
	return append(append(append(p.Part.params(), p.Phi.params()...), p.Rho.params()...), p.LogStd)
}

// PolicyCache holds one forward pass for backpropagation.
type PolicyCache struct {
	enc    *encoding
	shared []float64 // [context, global, body shape]: ρ's input shared by every actuator
	rho    [][][]float64
	Mu  []float64 // mean command per actuator, in [-1, 1]
}

// Forward returns the mean command per actuator and a cache for Backward.
func (p *SetPolicy) Forward(o *Obs) *PolicyCache {
	g := p.GloNorm.Apply(o.Global)
	enc := encode(&p.Part, &p.Phi, p.Embed, p.TokNorm.applyAll(o.Tokens), p.PartNorm.applyAll(o.Parts), g)
	c := &PolicyCache{enc: enc, rho: make([][][]float64, len(enc.h)), Mu: make([]float64, len(enc.h))}
	c.shared = concat(enc.ctx, enc.g2)
	pre := p.Rho.sharedPre(p.Embed, c.shared)
	for i, h := range enc.h {
		c.rho[i] = p.Rho.fwdWith(h, pre)
		c.Mu[i] = math.Tanh(c.rho[i][len(c.rho[i])-1][0])
	}
	return c
}

// Backward accumulates into g the gradient for dMu (loss gradient with
// respect to each actuator's mean command).
func (p *SetPolicy) Backward(c *PolicyCache, dMu []float64, g *SetPolicy) {
	E := p.Embed
	n := len(c.Mu)
	dh := make([][]float64, n)
	dpre := make([]float64, p.Rho.Layers[0].Out)
	for i := 0; i < n; i++ {
		dm := dMu[i] * (1 - c.Mu[i]*c.Mu[i])
		dh[i] = p.Rho.bwdWith(c.rho[i], []float64{dm}, &g.Rho, dpre)
	}
	ds := p.Rho.sharedBwd(E, c.shared, dpre, &g.Rho)
	c.enc.backward(&p.Part, &p.Phi, &g.Part, &g.Phi, E, dh, ds[:E], ds[E:])
}

// SetValue is the critic: the same two set encoders over the same tokens,
// with the privileged features appended to the global vector. Only
// training uses it.
type SetValue struct {
	TokenDim  int         `json:"tokenDim"`
	PartDim   int         `json:"partDim"`
	GlobalDim int         `json:"globalDim"` // including the privileged features
	Embed     int         `json:"embed"`
	Part      MLP         `json:"part"`
	Phi       MLP         `json:"phi"`
	Rho       MLP         `json:"rho"` // [mean embedding, global, body shape] -> value
	TokNorm   *Normalizer `json:"tokNorm"`
	PartNorm  *Normalizer `json:"partNorm"`
	GloNorm   *Normalizer `json:"globalNorm"`
}

// ValueScale maps the network's output to returns (rewards ≈ 1 per step).
const ValueScale = 20

// NewValue creates a critic; globalDim includes the privileged features.
func NewValue(rng *rand.Rand, tokenDim, partDim, globalDim, embed, hidden int) *SetValue {
	g2 := globalDim + embed
	return &SetValue{TokenDim: tokenDim, PartDim: partDim, GlobalDim: globalDim, Embed: embed,
		Part:    newMLP(rng, partDim+globalDim, hidden, embed),
		Phi:     newMLP(rng, tokenDim+g2, hidden, embed),
		Rho:     newMLP(rng, embed+g2, hidden, 1),
		TokNorm: NewNormalizer(tokenDim), PartNorm: NewNormalizer(partDim), GloNorm: NewNormalizer(globalDim)}
}

func (v *SetValue) Clone(zero bool) *SetValue {
	c := &SetValue{TokenDim: v.TokenDim, PartDim: v.PartDim, GlobalDim: v.GlobalDim, Embed: v.Embed,
		Part: v.Part.clone(zero), Phi: v.Phi.clone(zero), Rho: v.Rho.clone(zero)}
	if !zero {
		c.TokNorm, c.PartNorm, c.GloNorm = v.TokNorm.clone(), v.PartNorm.clone(), v.GloNorm.clone()
	}
	return c
}

func (v *SetValue) Params() [][]float64 {
	return append(append(v.Part.params(), v.Phi.params()...), v.Rho.params()...)
}

// UpdateNorms folds a batch of observations into the input statistics.
func (v *SetValue) UpdateNorms(batch []*Obs) {
	var toks, parts, globs [][]float64
	for _, o := range batch {
		toks = append(toks, o.Tokens...)
		parts = append(parts, o.Parts...)
		globs = append(globs, concat(o.Global, o.Priv))
	}
	v.TokNorm.Update(toks)
	v.PartNorm.Update(parts)
	v.GloNorm.Update(globs)
}

type ValueCache struct {
	enc *encoding
	rho [][]float64
	V   float64
}

func (v *SetValue) Forward(o *Obs) *ValueCache {
	g := v.GloNorm.Apply(concat(o.Global, o.Priv))
	enc := encode(&v.Part, &v.Phi, v.Embed, v.TokNorm.applyAll(o.Tokens), v.PartNorm.applyAll(o.Parts), g)
	c := &ValueCache{enc: enc}
	c.rho = v.Rho.fwd(concat(enc.ctx, enc.g2))
	c.V = ValueScale * c.rho[len(c.rho)-1][0]
	return c
}

// Backward accumulates into g the gradient for dV (loss gradient with
// respect to the value).
func (v *SetValue) Backward(c *ValueCache, dV float64, g *SetValue) {
	din := v.Rho.bwd(c.rho, []float64{dV * ValueScale}, &g.Rho)
	E := v.Embed
	c.enc.backward(&v.Part, &v.Phi, &g.Part, &g.Phi, E, nil, din[:E], din[E:])
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
