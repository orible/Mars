package policy

import "math"

// Activations is one forward pass laid out for inspection: every set's
// normalised inputs, its per-token embeddings and their pooled mean, the
// decoder's hidden layers per actuator, and the commands. It is read from
// a PolicyCache after the fact, so the forward pass (shared with
// training) does no extra work for it.
type Activations struct {
	Global []float64 `json:"global"` // normalised global vector

	Parts      [][]float64 `json:"parts"`      // normalised part tokens
	PartEmb    [][]float64 `json:"partEmb"`    // tanh part embeddings
	PartPool   []float64   `json:"partPool"`   // their mean: the body-shape context
	Sensors    [][]float64 `json:"sensors"`    // normalised sensor tokens
	SensorEmb  [][]float64 `json:"sensorEmb"`  // tanh sensor embeddings
	SensorPool []float64   `json:"sensorPool"` // their mean: what the body feels

	Tokens    [][]float64 `json:"tokens"`           // normalised actuator tokens
	PhiHid    [][]float64 `json:"phiHid,omitempty"` // φ's last hidden layer per actuator
	TokenEmb  [][]float64 `json:"tokenEmb"`         // tanh actuator embeddings
	TokenPool []float64   `json:"tokenPool"`        // their mean: the body context

	// Attention (if the policy has it): Attn[h][i][j] is how much actuator
	// i listens to actuator j in head h (rows sum to 1), Mixed[i] is the
	// embedding ρ reads (TokenEmb[i] plus what attention adds), AttnGain[i]
	// the size of that addition relative to TokenEmb[i].
	Attn     [][][]float64 `json:"attn,omitempty"`
	Mixed    [][]float64   `json:"mixed,omitempty"`
	AttnGain []float64     `json:"attnGain,omitempty"`

	// RhoHid holds ρ's hidden layers per actuator: RhoHid[i][l] is
	// actuator i's layer l after tanh.
	RhoHid [][][]float64 `json:"rhoHid"`
	Pre    []float64     `json:"pre"`    // ρ's linear output per actuator
	Mu     []float64     `json:"mu"`     // mean command per actuator, tanh(pre)
	LogStd float64       `json:"logStd"` // exploration noise (shared by all actuators)
}

// Activations unpacks c, a forward pass of p.
func (p *SetPolicy) Activations(c *PolicyCache) *Activations {
	e := c.enc
	a := &Activations{Global: e.g, Mu: c.Mu, LogStd: p.LogStd[0]}
	pool := func(s int) []float64 {
		off := len(e.g) + s*p.Embed
		return e.g2[off : off+p.Embed]
	}
	inputs := func(acts [][][]float64) [][]float64 {
		out := make([][]float64, len(acts))
		for i, ac := range acts {
			out[i] = ac[0]
		}
		return out
	}
	a.Parts, a.PartEmb, a.PartPool = inputs(e.sets[0].acts), e.sets[0].h, pool(0)
	a.Sensors, a.SensorEmb, a.SensorPool = inputs(e.sets[1].acts), e.sets[1].h, pool(1)
	a.Tokens, a.TokenEmb, a.TokenPool = inputs(e.phiActs), e.h, e.ctx
	for _, ac := range e.phiActs {
		a.PhiHid = append(a.PhiHid, ac[len(ac)-2])
	}
	if c.att != nil {
		a.Attn, a.Mixed = c.att.alpha, c.att.out
		for i, h := range e.h {
			dz, hh := 0.0, 0.0
			for j, v := range h {
				d := c.att.out[i][j] - v
				dz, hh = dz+d*d, hh+v*v
			}
			a.AttnGain = append(a.AttnGain, math.Sqrt(dz/math.Max(hh, 1e-12)))
		}
	}
	for _, ac := range c.rho {
		a.RhoHid = append(a.RhoHid, ac[1:len(ac)-1])
		a.Pre = append(a.Pre, ac[len(ac)-1][0])
	}
	return a
}
