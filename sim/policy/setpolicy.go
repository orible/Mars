// Package policy holds the morphology-agnostic movement network.
//
// The network reads four things. Each actuator is a token. Each body
// part (a link with real mass, or one that touches the floor) is a token
// too, describing the body's own shape: how long the part is, how big,
// how heavy, where it is and whether it is on the floor. Each sensor (a
// pressure pad on a sole, the tension in a tendon) is a token: where it
// is and what it feels. The global vector holds the IMU, the centre of
// gravity and the command.
//
// Shared encoders embed every part token and every sensor token with the
// global vector; each set is mean-pooled into a context (body shape, what
// the feet feel) appended to the global vector. A shared actuator encoder φ then embeds
// every actuator token, the embeddings are mean-pooled into a body
// context, and a shared decoder ρ reads each actuator's embedding with
// both contexts and outputs that actuator's valve command. No weight
// depends on how many actuators or parts the body has, and the network is
// permutation-equivariant over both (a Deep Sets architecture).
//
// Inputs are normalised by running statistics that are saved with the
// weights, so the same checkpoint behaves the same in training and in the
// sim server.
package policy

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
)

// Obs is one observation.
type Obs struct {
	Tokens  [][]float64 // one per actuator
	Parts   [][]float64 // one per body part
	Sensors [][]float64 // one per sensor (may be none)
	Global []float64
	// Priv holds privileged facts only the simulator knows (true
	// velocity, height, friction, harness). Only the critic reads them,
	// so the policy never depends on them.
	Priv []float64
}

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

// Normalizer standardises a feature vector with running mean and variance
// (merged batch by batch), clipping the result to ±5.
type Normalizer struct {
	Mean  []float64 `json:"mean"`
	Var   []float64 `json:"var"`
	Count float64   `json:"count"`
}

func NewNormalizer(n int) *Normalizer {
	z := &Normalizer{Mean: make([]float64, n), Var: make([]float64, n)}
	for i := range z.Var {
		z.Var[i] = 1
	}
	return z
}

// varFloor keeps features that barely vary (one body's fixed geometry)
// from being blown up.
const varFloor = 1e-3

// Apply returns the normalised copy of x.
func (z *Normalizer) Apply(x []float64) []float64 {
	y := make([]float64, len(x))
	if z == nil {
		copy(y, x)
		return y
	}
	for i, v := range x {
		y[i] = math.Max(-5, math.Min(5, (v-z.Mean[i])/math.Sqrt(z.Var[i]+varFloor)))
	}
	return y
}

func (z *Normalizer) applyAll(xs [][]float64) [][]float64 {
	out := make([][]float64, len(xs))
	for i, x := range xs {
		out[i] = z.Apply(x)
	}
	return out
}

// Update merges a batch of rows into the statistics.
func (z *Normalizer) Update(rows [][]float64) {
	if z == nil || len(rows) == 0 {
		return
	}
	n := float64(len(rows))
	for i := range z.Mean {
		bm, bv := 0.0, 0.0
		for _, r := range rows {
			bm += r[i]
		}
		bm /= n
		for _, r := range rows {
			d := r[i] - bm
			bv += d * d
		}
		bv /= n
		if z.Count == 0 {
			z.Mean[i], z.Var[i] = bm, bv
			continue
		}
		tot := z.Count + n
		d := bm - z.Mean[i]
		z.Mean[i] += d * n / tot
		z.Var[i] = (z.Var[i]*z.Count + bv*n + d*d*z.Count*n/tot) / tot
	}
	z.Count += n
}

func (z *Normalizer) clone() *Normalizer {
	if z == nil {
		return nil
	}
	return &Normalizer{Mean: append([]float64(nil), z.Mean...), Var: append([]float64(nil), z.Var...), Count: z.Count}
}

// SetPolicy is the movement network. Shapes depend only on the feature
// sizes, never on the actuator or part count.
type SetPolicy struct {
	// ObsVersion is the observation layout the weights were trained on
	// (obs.Version); a checkpoint from another layout is refused.
	ObsVersion int `json:"obsVersion"`
	TokenDim   int `json:"tokenDim"`
	PartDim    int `json:"partDim"`
	SensorDim  int `json:"sensorDim"`
	GlobalDim  int `json:"globalDim"`
	Embed      int `json:"embed"`
	Part       MLP `json:"part"`   // [part, global] -> part embedding
	Sensor     MLP `json:"sensor"` // [sensor, global] -> sensor embedding
	Phi        MLP `json:"phi"`    // [token, global, body shape, feel] -> embedding
	Rho        MLP `json:"rho"`    // [embedding, context, global, body shape, feel] -> command
	// LogStd is the log of the exploration noise added to every command
	// during training (one value shared by all actuators).
	LogStd   []float64   `json:"logStd"`
	TokNorm  *Normalizer `json:"tokNorm"`
	PartNorm *Normalizer `json:"partNorm"`
	SensNorm *Normalizer `json:"sensorNorm"`
	GloNorm  *Normalizer `json:"globalNorm"`
}

// New creates a randomly initialised policy.
func New(rng *rand.Rand, tokenDim, partDim, sensorDim, globalDim, embed, hidden int) *SetPolicy {
	g2 := globalDim + 2*embed
	return &SetPolicy{
		TokenDim: tokenDim, PartDim: partDim, SensorDim: sensorDim, GlobalDim: globalDim, Embed: embed,
		Part:     newMLP(rng, partDim+globalDim, hidden, embed),
		Sensor:   newMLP(rng, sensorDim+globalDim, hidden, embed),
		Phi:      newMLP(rng, tokenDim+g2, hidden, embed),
		Rho:      newMLP(rng, 2*embed+g2, hidden, 1),
		LogStd:   []float64{DefaultLogStd},
		TokNorm:  NewNormalizer(tokenDim),
		PartNorm: NewNormalizer(partDim),
		SensNorm: NewNormalizer(sensorDim),
		GloNorm:  NewNormalizer(globalDim),
	}
}

// Act returns one valve command in [-1, 1] per actuator token.
func (p *SetPolicy) Act(o *Obs) []float64 {
	if len(o.Tokens) == 0 {
		return nil
	}
	return p.Forward(o).Mu
}

// UpdateNorms folds a batch of observations into the input statistics.
func (p *SetPolicy) UpdateNorms(batch []*Obs) {
	var toks, parts, sens, globs [][]float64
	for _, o := range batch {
		toks = append(toks, o.Tokens...)
		parts = append(parts, o.Parts...)
		sens = append(sens, o.Sensors...)
		globs = append(globs, o.Global)
	}
	p.TokNorm.Update(toks)
	p.PartNorm.Update(parts)
	p.SensNorm.Update(sens)
	p.GloNorm.Update(globs)
}

// Save writes a checkpoint. Checkpoints are independent of any body spec.
func (p *SetPolicy) Save(w io.Writer) error {
	return json.NewEncoder(w).Encode(p)
}

// Load reads a checkpoint written by Save.
func Load(r io.Reader) (*SetPolicy, error) {
	var p SetPolicy
	if err := json.NewDecoder(r).Decode(&p); err != nil {
		return nil, err
	}
	if err := p.Check(); err != nil {
		return nil, err
	}
	return &p, nil
}

// Check rejects checkpoints whose shapes do not fit together (such as
// ones written before body-part tokens existed).
func (p *SetPolicy) Check() error {
	if len(p.Part.Layers) == 0 || len(p.Sensor.Layers) == 0 || len(p.Phi.Layers) == 0 || len(p.Rho.Layers) == 0 || p.TokNorm == nil || p.SensNorm == nil {
		return fmt.Errorf("checkpoint predates body-part and sensor inputs; train a new one")
	}
	if p.Phi.Layers[0].In != p.TokenDim+p.GlobalDim+2*p.Embed || p.Part.Layers[0].In != p.PartDim+p.GlobalDim || p.Sensor.Layers[0].In != p.SensorDim+p.GlobalDim {
		return fmt.Errorf("checkpoint layer sizes do not match its feature sizes")
	}
	if len(p.LogStd) != 1 {
		p.LogStd = []float64{DefaultLogStd}
	}
	return nil
}
