package policy

import (
	"fmt"
	"math"
	"math/rand"
)

// Attention lets the actuator tokens signal each other before ρ reads
// them. Mean pooling only tells every actuator the average of all the
// others; attention lets each one pick out the few it cares about (a knee
// listening to the other hip), with weights learned from the tokens'
// contents, so it still works for any body and any actuator order.
//
// It is multi-head scaled dot-product self-attention over the actuator
// embeddings, added as a residual: out_i = h_i + Wo·Σ_j α_ij·Wv·h_j,
// α_i = softmax_j(Wq·h_i · Wk·h_j / √d) per head. Wo starts at zero, so
// adding attention to a trained policy changes none of its outputs until
// training moves Wo.
type Attention struct {
	Heads int       `json:"heads"`
	Dim   int       `json:"dim"`   // per head
	Embed int       `json:"embed"` // token width (the policy's Embed)
	Wq    []float64 `json:"wq"`    // (Heads·Dim)×Embed, row-major
	Wk    []float64 `json:"wk"`
	Wv    []float64 `json:"wv"`
	Wo    []float64 `json:"wo"` // Embed×(Heads·Dim)
}

// NewAttention returns attention with random query, key and value maps
// and a zero output map (an identity until trained).
func NewAttention(rng *rand.Rand, embed, heads, dim int) *Attention {
	hd := heads * dim
	a := &Attention{Heads: heads, Dim: dim, Embed: embed,
		Wq: make([]float64, hd*embed), Wk: make([]float64, hd*embed), Wv: make([]float64, hd*embed), Wo: make([]float64, embed*hd)}
	limit := math.Sqrt(6 / float64(embed+hd))
	for _, w := range [][]float64{a.Wq, a.Wk, a.Wv} {
		for i := range w {
			w[i] = (rng.Float64()*2 - 1) * limit
		}
	}
	return a
}

// AddAttention gives p attention over its actuator tokens (no-op if it
// already has it). Outputs are unchanged until training moves Wo.
func (p *SetPolicy) AddAttention(rng *rand.Rand, heads, dim int) {
	if p.Attn == nil {
		p.Attn = NewAttention(rng, p.Embed, heads, dim)
	}
}

func (a *Attention) check(embed int) error {
	hd := a.Heads * a.Dim
	if a.Embed != embed || hd == 0 || len(a.Wq) != hd*embed || len(a.Wk) != hd*embed || len(a.Wv) != hd*embed || len(a.Wo) != embed*hd {
		return fmt.Errorf("attention sizes do not match the embedding")
	}
	return nil
}

func (a *Attention) clone(zero bool) *Attention {
	if a == nil {
		return nil
	}
	c := &Attention{Heads: a.Heads, Dim: a.Dim, Embed: a.Embed}
	for _, pr := range [][2]*[]float64{{&c.Wq, &a.Wq}, {&c.Wk, &a.Wk}, {&c.Wv, &a.Wv}, {&c.Wo, &a.Wo}} {
		*pr[0] = make([]float64, len(*pr[1]))
		if !zero {
			copy(*pr[0], *pr[1])
		}
	}
	return c
}

func (a *Attention) params() [][]float64 {
	if a == nil {
		return nil
	}
	return [][]float64{a.Wq, a.Wk, a.Wv, a.Wo}
}

// attnPass is one forward pass, kept for backpropagation.
type attnPass struct {
	x       [][]float64   // inputs (the actuator embeddings)
	q, k, v [][]float64   // per token, Heads·Dim
	alpha   [][][]float64 // [head][i][j]
	o       [][]float64   // per token, Heads·Dim: Σ_j α_ij v_j
	out     [][]float64   // x + Wo·o
}

// matVec returns W·x for W rows×len(x).
func matVec(w []float64, rows int, x []float64) []float64 {
	n := len(x)
	y := make([]float64, rows)
	for r := 0; r < rows; r++ {
		row := w[r*n : (r+1)*n]
		s := 0.0
		for c, xc := range x {
			s += row[c] * xc
		}
		y[r] = s
	}
	return y
}

// matTVecAdd adds Wᵀ·d into dx and d⊗x into gw (W is len(d)×len(x)).
func matTVecAdd(w, gw []float64, x, d, dx []float64) {
	n := len(x)
	for r, dr := range d {
		if dr == 0 {
			continue
		}
		row, grow := w[r*n:(r+1)*n], gw[r*n:(r+1)*n]
		for c, xc := range x {
			grow[c] += dr * xc
			dx[c] += row[c] * dr
		}
	}
}

func (a *Attention) forward(x [][]float64) *attnPass {
	n, hd := len(x), a.Heads*a.Dim
	c := &attnPass{x: x, q: make([][]float64, n), k: make([][]float64, n), v: make([][]float64, n),
		alpha: make([][][]float64, a.Heads), o: make([][]float64, n), out: make([][]float64, n)}
	for i, xi := range x {
		c.q[i], c.k[i], c.v[i] = matVec(a.Wq, hd, xi), matVec(a.Wk, hd, xi), matVec(a.Wv, hd, xi)
		c.o[i] = make([]float64, hd)
	}
	scale := 1 / math.Sqrt(float64(a.Dim))
	for h := 0; h < a.Heads; h++ {
		off := h * a.Dim
		c.alpha[h] = make([][]float64, n)
		for i := 0; i < n; i++ {
			s := make([]float64, n)
			top := math.Inf(-1)
			for j := 0; j < n; j++ {
				d := 0.0
				for t := off; t < off+a.Dim; t++ {
					d += c.q[i][t] * c.k[j][t]
				}
				s[j] = d * scale
				top = math.Max(top, s[j])
			}
			sum := 0.0
			for j := range s {
				s[j] = math.Exp(s[j] - top)
				sum += s[j]
			}
			for j := range s {
				s[j] /= sum
				for t := off; t < off+a.Dim; t++ {
					c.o[i][t] += s[j] * c.v[j][t]
				}
			}
			c.alpha[h][i] = s
		}
	}
	for i, xi := range x {
		z := matVec(a.Wo, a.Embed, c.o[i])
		for e := range z {
			z[e] += xi[e]
		}
		c.out[i] = z
	}
	return c
}

// backward takes the loss gradient with respect to each output token,
// accumulates the weight gradients into g and returns the gradient with
// respect to each input token.
func (a *Attention) backward(c *attnPass, dout [][]float64, g *Attention) [][]float64 {
	n, hd := len(c.x), a.Heads*a.Dim
	dx := make([][]float64, n)
	do := make([][]float64, n)
	dq, dk, dv := make([][]float64, n), make([][]float64, n), make([][]float64, n)
	for i := range dx {
		dx[i] = append([]float64(nil), dout[i]...) // residual path
		do[i] = make([]float64, hd)
		dq[i], dk[i], dv[i] = make([]float64, hd), make([]float64, hd), make([]float64, hd)
		matTVecAdd(a.Wo, g.Wo, c.o[i], dout[i], do[i])
	}
	scale := 1 / math.Sqrt(float64(a.Dim))
	da := make([]float64, n)
	for h := 0; h < a.Heads; h++ {
		off := h * a.Dim
		for i := 0; i < n; i++ {
			al := c.alpha[h][i]
			dot := 0.0
			for j := 0; j < n; j++ {
				d := 0.0
				for t := off; t < off+a.Dim; t++ {
					d += do[i][t] * c.v[j][t]
					dv[j][t] += al[j] * do[i][t]
				}
				da[j] = d
				dot += al[j] * d
			}
			for j := 0; j < n; j++ {
				ds := al[j] * (da[j] - dot) * scale
				if ds == 0 {
					continue
				}
				for t := off; t < off+a.Dim; t++ {
					dq[i][t] += ds * c.k[j][t]
					dk[j][t] += ds * c.q[i][t]
				}
			}
		}
	}
	for i, xi := range c.x {
		matTVecAdd(a.Wq, g.Wq, xi, dq[i], dx[i])
		matTVecAdd(a.Wk, g.Wk, xi, dk[i], dx[i])
		matTVecAdd(a.Wv, g.Wv, xi, dv[i], dx[i])
	}
	return dx
}
