// Package ppo trains the movement network with proximal policy
// optimisation. Each valve command is a Gaussian around the network's mean
// with one shared, learned noise level, so a sample's log-probability is a
// sum over however many cylinders that body has. Rollouts run one
// environment per goroutine; gradients are split across workers.
package ppo

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"mars/sim/env"
	"mars/sim/policy"
)

type Config struct {
	Envs      int     // parallel environments
	Steps     int     // policy steps per environment per iteration
	Gamma     float64 // discount per policy step
	Lambda    float64 // GAE λ
	Clip      float64 // PPO ratio clip
	LR, VLR   float64 // policy and value learning rates
	Epochs    int
	Minibatch int
	Entropy   float64 // entropy bonus per actuator
	TargetKL  float64 // stop the epoch loop early above this
	MaxGrad   float64 // gradient norm clip
	Workers   int
}

func Defaults() Config {
	return Config{Envs: 24, Steps: 128, Gamma: 0.99, Lambda: 0.95, Clip: 0.2, LR: 3e-4, VLR: 1e-3,
		Epochs: 5, Minibatch: 512, Entropy: 0, TargetKL: 0.02, MaxGrad: 1, Workers: runtime.GOMAXPROCS(0)}
}

type sample struct {
	tokens [][]float64
	global []float64
	act    []float64
	logp   float64
	value  float64
	ret    float64
	adv    float64
}

// Stats summarises one iteration.
type Stats struct {
	Iter                 int
	Samples              int
	Episodes, Falls      int
	MeanReturn, MeanSecs float64 // over episodes that ended this iteration
	// Episodes that started on the floor, and how many got up.
	LyingEpisodes, GotUp int
	// Episodes that started standing without the harness: the robot's
	// real ability.
	SoloEpisodes, SoloFalls int
	SoloSecs                float64
	AssistedFalls           int
	Std                     float64
	KL, ClipFrac            float64
	ValueLoss               float64
	Epochs                  int
	RolloutSec, TrainSec    float64
}

type Trainer struct {
	Cfg  Config
	Pol  *policy.SetPolicy
	Val  *policy.SetValue
	Iter int

	envs       []*env.Env
	rngs       []*rand.Rand
	epRet      []float64
	popt, vopt *policy.Adam
}

// SetAssist changes the harness strength for episodes that start from now
// on. Call it between iterations.
func (t *Trainer) SetAssist(a float64) {
	for _, e := range t.envs {
		e.Cfg.Assist = a
	}
}

func New(cfg Config, pol *policy.SetPolicy, val *policy.SetValue, envs []*env.Env, seed int64) *Trainer {
	t := &Trainer{Cfg: cfg, Pol: pol, Val: val, envs: envs, epRet: make([]float64, len(envs)),
		popt: policy.NewAdam(pol.Params(), cfg.LR), vopt: policy.NewAdam(val.Params(), cfg.VLR)}
	for i := range envs {
		t.rngs = append(t.rngs, rand.New(rand.NewSource(seed+int64(i)*7919)))
	}
	return t
}

func logProb(act, mu []float64, logStd float64) float64 {
	s := math.Exp(logStd)
	lp := 0.0
	for i := range act {
		z := (act[i] - mu[i]) / s
		lp += -0.5*z*z - logStd - 0.5*math.Log(2*math.Pi)
	}
	return lp
}

type episode struct {
	ret, secs float64
	fell      bool // fell at least once
	assisted  bool
	lying     bool // started on the floor
	gotUp     bool // got up at least once
}

// rollout runs env i for Steps policy steps and returns its samples with
// advantages and returns filled in.
func (t *Trainer) rollout(i int) ([]sample, []episode) {
	e, rng, cfg := t.envs[i], t.rngs[i], t.Cfg
	std := t.Pol.Std()
	out := make([]sample, 0, cfg.Steps)
	rew := make([]float64, 0, cfg.Steps)
	done := make([]bool, 0, cfg.Steps)
	var eps []episode
	for s := 0; s < cfg.Steps; s++ {
		tok, g := e.Observe()
		mu := t.Pol.Forward(tok, g).Mu
		act := make([]float64, len(mu))
		for k := range mu {
			act[k] = mu[k] + std*rng.NormFloat64()
		}
		sm := sample{tokens: tok, global: g, act: act, logp: logProb(act, mu, t.Pol.LogStd[0]), value: t.Val.Forward(tok, g).V}
		r, terminal, trunc := e.Step(act)
		t.epRet[i] += r
		if trunc && !terminal {
			nt, ng := e.Observe()
			r += cfg.Gamma * t.Val.Forward(nt, ng).V
		}
		if terminal || trunc {
			eps = append(eps, episode{ret: t.epRet[i], secs: e.Seconds(), fell: e.Falls > 0, assisted: e.Assisted,
				lying: e.StartedLying, gotUp: e.GetUps > 0})
			t.epRet[i] = 0
			e.Reset()
		}
		out = append(out, sm)
		rew = append(rew, r)
		done = append(done, terminal || trunc)
	}
	tok, g := e.Observe()
	next := t.Val.Forward(tok, g).V
	adv := 0.0
	for s := len(out) - 1; s >= 0; s-- {
		if done[s] {
			next, adv = 0, 0
		}
		delta := rew[s] + cfg.Gamma*next - out[s].value
		adv = delta + cfg.Gamma*cfg.Lambda*adv
		out[s].adv = adv
		out[s].ret = adv + out[s].value
		next = out[s].value
	}
	return out, eps
}

// Iterate collects one batch of experience and updates both networks.
func (t *Trainer) Iterate() Stats {
	t.Iter++
	st := Stats{Iter: t.Iter}
	t0 := time.Now()
	parts := make([][]sample, len(t.envs))
	epParts := make([][]episode, len(t.envs))
	var wg sync.WaitGroup
	for i := range t.envs {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			parts[i], epParts[i] = t.rollout(i)
		}(i)
	}
	wg.Wait()
	var batch []sample
	for i := range parts {
		batch = append(batch, parts[i]...)
		for _, ep := range epParts[i] {
			st.Episodes++
			st.MeanReturn += ep.ret
			st.MeanSecs += ep.secs
			if ep.fell {
				st.Falls++
			}
			if ep.lying {
				st.LyingEpisodes++
				if ep.gotUp {
					st.GotUp++
				}
			} else if !ep.assisted {
				st.SoloEpisodes++
				st.SoloSecs += ep.secs
				if ep.fell {
					st.SoloFalls++
				}
			} else if ep.fell {
				st.AssistedFalls++
			}
		}
	}
	if st.Episodes > 0 {
		st.MeanReturn /= float64(st.Episodes)
		st.MeanSecs /= float64(st.Episodes)
	}
	if st.SoloEpisodes > 0 {
		st.SoloSecs /= float64(st.SoloEpisodes)
	}
	st.Samples = len(batch)
	st.RolloutSec = time.Since(t0).Seconds()

	// Normalise advantages.
	mean, sq := 0.0, 0.0
	for _, s := range batch {
		mean += s.adv
	}
	mean /= float64(len(batch))
	for _, s := range batch {
		sq += (s.adv - mean) * (s.adv - mean)
	}
	sd := math.Sqrt(sq/float64(len(batch))) + 1e-8
	for i := range batch {
		batch[i].adv = (batch[i].adv - mean) / sd
	}

	t1 := time.Now()
	st.KL, st.ClipFrac, st.ValueLoss, st.Epochs = t.update(batch)
	st.Std = t.Pol.Std()
	st.TrainSec = time.Since(t1).Seconds()
	return st
}

type workerOut struct {
	gp          *policy.SetPolicy
	gv          *policy.SetValue
	kl, clipped float64
	vloss       float64
}

func (t *Trainer) update(batch []sample) (kl, clipFrac, vloss float64, epochs int) {
	cfg := t.Cfg
	W := max(1, cfg.Workers)
	outs := make([]workerOut, W)
	for w := range outs {
		outs[w] = workerOut{gp: t.Pol.Clone(true), gv: t.Val.Clone(true)}
	}
	gp, gv := t.Pol.Clone(true), t.Val.Clone(true)
	idx := rand.Perm(len(batch))
	for ep := 0; ep < cfg.Epochs; ep++ {
		epochs = ep + 1
		rand.Shuffle(len(idx), func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })
		var epKL, epClip, epV float64
		for start := 0; start < len(idx); start += cfg.Minibatch {
			mb := idx[start:min(start+cfg.Minibatch, len(idx))]
			B := float64(len(mb))
			logStd := t.Pol.LogStd[0]
			std := math.Exp(logStd)
			var wg sync.WaitGroup
			for w := 0; w < W; w++ {
				wg.Add(1)
				go func(w int) {
					defer wg.Done()
					o := &outs[w]
					policy.Zero(o.gp.Params())
					policy.Zero(o.gv.Params())
					o.kl, o.clipped, o.vloss = 0, 0, 0
					for k := w; k < len(mb); k += W {
						s := &batch[mb[k]]
						c := t.Pol.Forward(s.tokens, s.global)
						lp := logProb(s.act, c.Mu, logStd)
						ratio := math.Exp(lp - s.logp)
						o.kl += (ratio - 1) - (lp - s.logp)
						clipped := (s.adv > 0 && ratio > 1+cfg.Clip) || (s.adv < 0 && ratio < 1-cfg.Clip)
						if clipped {
							o.clipped++
						}
						n := float64(len(c.Mu))
						dlogStd := -cfg.Entropy * n / B
						if !clipped {
							gl := -s.adv * ratio / B // dLoss/dlogp
							dMu := make([]float64, len(c.Mu))
							for i := range dMu {
								z := (s.act[i] - c.Mu[i]) / std
								dMu[i] = gl * z / std
								dlogStd += gl * (z*z - 1)
							}
							t.Pol.Backward(c, dMu, o.gp)
						}
						o.gp.LogStd[0] += dlogStd
						vc := t.Val.Forward(s.tokens, s.global)
						diff := vc.V - s.ret
						o.vloss += 0.5 * diff * diff
						t.Val.Backward(vc, diff/B/policy.ValueScale, o.gv)
					}
				}(w)
			}
			wg.Wait()
			policy.Zero(gp.Params())
			policy.Zero(gv.Params())
			for w := range outs {
				policy.AddInto(gp.Params(), outs[w].gp.Params())
				policy.AddInto(gv.Params(), outs[w].gv.Params())
				epKL += outs[w].kl
				epClip += outs[w].clipped
				epV += outs[w].vloss
			}
			t.popt.Step(t.Pol.Params(), gp.Params(), cfg.MaxGrad)
			t.vopt.Step(t.Val.Params(), gv.Params(), cfg.MaxGrad)
			t.Pol.LogStd[0] = math.Max(-2.5, math.Min(0, t.Pol.LogStd[0]))
		}
		n := float64(len(batch))
		kl, clipFrac, vloss = epKL/n, epClip/n, epV/n
		if cfg.TargetKL > 0 && kl > cfg.TargetKL {
			break
		}
	}
	return kl, clipFrac, vloss, epochs
}
