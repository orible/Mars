// Package policy holds the morphology-agnostic movement network.
//
// Each actuator is a token. A shared encoder φ embeds every token together
// with the global observation; embeddings are mean-pooled into a body
// context; a shared decoder ρ reads each token's embedding plus the context
// and outputs that actuator's valve command. The weights therefore do not
// depend on the number of actuators, and the network is
// permutation-equivariant over actuators (a Deep Sets architecture).
//
// Mean pooling is the starting point; attention over actuator tokens can
// replace it later without changing Act's signature.
package policy

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
)

// Dense is a fully connected layer. W is row-major, Out×In.
type Dense struct {
	In  int       `json:"in"`
	Out int       `json:"out"`
	W   []float64 `json:"w"`
	B   []float64 `json:"b"`
}

func newDense(rng *rand.Rand, in, out int) Dense {
	d := Dense{In: in, Out: out, W: make([]float64, in*out), B: make([]float64, out)}
	limit := math.Sqrt(6 / float64(in+out)) // Glorot uniform
	for i := range d.W {
		d.W[i] = (rng.Float64()*2 - 1) * limit
	}
	return d
}

func (d *Dense) forward(x []float64) []float64 {
	y := make([]float64, d.Out)
	for o := 0; o < d.Out; o++ {
		s := d.B[o]
		row := d.W[o*d.In : (o+1)*d.In]
		for i, xi := range x {
			s += row[i] * xi
		}
		y[o] = s
	}
	return y
}

// MLP applies tanh between layers and leaves the last layer linear.
type MLP struct {
	Layers []Dense `json:"layers"`
}

func newMLP(rng *rand.Rand, sizes ...int) MLP {
	var m MLP
	for i := 0; i+1 < len(sizes); i++ {
		m.Layers = append(m.Layers, newDense(rng, sizes[i], sizes[i+1]))
	}
	return m
}

func (m *MLP) forward(x []float64) []float64 {
	for i := range m.Layers {
		x = m.Layers[i].forward(x)
		if i < len(m.Layers)-1 {
			for j := range x {
				x[j] = math.Tanh(x[j])
			}
		}
	}
	return x
}

// SetPolicy is the movement network. Shapes depend only on the per-token
// and global feature sizes, never on the actuator count.
type SetPolicy struct {
	TokenDim  int `json:"tokenDim"`
	GlobalDim int `json:"globalDim"`
	Embed     int `json:"embed"`
	Phi       MLP `json:"phi"` // [token, global] -> embedding
	Rho       MLP `json:"rho"` // [embedding, context, global] -> command
}

// New creates a randomly initialised policy.
func New(rng *rand.Rand, tokenDim, globalDim, embed, hidden int) *SetPolicy {
	return &SetPolicy{
		TokenDim:  tokenDim,
		GlobalDim: globalDim,
		Embed:     embed,
		Phi:       newMLP(rng, tokenDim+globalDim, hidden, embed),
		Rho:       newMLP(rng, 2*embed+globalDim, hidden, 1),
	}
}

// Act returns one valve command in [-1, 1] per actuator token.
func (p *SetPolicy) Act(tokens [][]float64, global []float64) []float64 {
	if len(tokens) == 0 {
		return nil
	}
	emb := make([][]float64, len(tokens))
	ctx := make([]float64, p.Embed)
	for i, tok := range tokens {
		in := append(append(make([]float64, 0, p.TokenDim+p.GlobalDim), tok...), global...)
		h := p.Phi.forward(in)
		for j := range h {
			h[j] = math.Tanh(h[j])
			ctx[j] += h[j] / float64(len(tokens))
		}
		emb[i] = h
	}
	out := make([]float64, len(tokens))
	for i, h := range emb {
		in := make([]float64, 0, 2*p.Embed+p.GlobalDim)
		in = append(append(append(in, h...), ctx...), global...)
		out[i] = math.Tanh(p.Rho.forward(in)[0])
	}
	return out
}

// Save writes a checkpoint. Checkpoints are independent of any body spec.
func (p *SetPolicy) Save(w io.Writer) error {
	enc := json.NewEncoder(w)
	return enc.Encode(p)
}

// Load reads a checkpoint written by Save.
func Load(r io.Reader) (*SetPolicy, error) {
	var p SetPolicy
	if err := json.NewDecoder(r).Decode(&p); err != nil {
		return nil, err
	}
	if len(p.Phi.Layers) == 0 || len(p.Rho.Layers) == 0 {
		return nil, fmt.Errorf("checkpoint has no layers")
	}
	if p.Phi.Layers[0].In != p.TokenDim+p.GlobalDim {
		return nil, fmt.Errorf("checkpoint expects %d token+global features, phi takes %d", p.TokenDim+p.GlobalDim, p.Phi.Layers[0].In)
	}
	return &p, nil
}
