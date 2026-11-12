// Package ppo trains the movement network with proximal policy
// optimisation. Each valve command is a Gaussian around the network's mean
// with one shared, learned noise level, so a sample's log-probability is a
// sum over however many cylinders that body has. Rollouts run one
// environment per goroutine; gradients are split across workers.
//
// Beyond plain PPO:
//   - inputs are normalised by running statistics, warmed up before the
//     first update and refreshed after every update;
//   - the critic sees privileged facts (true velocity, height, friction,
//     harness, time left) that the policy does not, so its advantages are
//     sharper without the policy leaning on them;
//   - a symmetry loss asks the policy to command the mirror image of the
//     body the way it commands the body, so whatever it learns on one side
//     it knows on the other;
//   - the learning rate follows the measured KL divergence, larger while
//     updates are timid and smaller when they overshoot.
package ppo

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"mars/sim/env"
	"mars/sim/obs"
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
	DesiredKL float64 // adapt the policy learning rate towards this KL per update (0 = fixed rate)
	MinLR     float64
	MaxLR     float64
	Symmetry  float64 // weight of the mirror-symmetry loss (0 = off)
	MaxGrad   float64 // gradient norm clip
	Workers   int
}

// symEvery: the symmetry loss runs on one sample in this many (it costs a
// second forward and backward pass).
const symEvery = 4

// MinLogStd is the floor on the exploration noise: σ = 0.05 of an action,
// a target 2.5% of a stroke off. It was 0.2 when actions were valve
// openings, to keep exploring; with servo targets, standing still needs
// the noise to be able to get small.
var MinLogStd = math.Log(0.05)

func Defaults() Config {
	return Config{Envs: 24, Steps: 256, Gamma: 0.99, Lambda: 0.95, Clip: 0.2, LR: 3e-4, VLR: 1e-3,
		Epochs: 5, Minibatch: 512, Entropy: 0, TargetKL: 0.03, DesiredKL: 0.01, MinLR: 5e-5, MaxLR: 1e-3,
		Symmetry: 0.5, MaxGrad: 1, Workers: runtime.GOMAXPROCS(0)}
}

type sample struct {
	o      *policy.Obs
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
	// Episodes that started on the floor unheld, and how many got up.
	LyingEpisodes, GotUp int
	// Floor starts held by the harness, and how many got up.
	HeldFloor, HeldGotUp int
	// Episodes that started standing without the harness: the robot's
	// real ability.
	SoloEpisodes, SoloFalls int
	SoloSecs                float64
	AssistedFalls           int
	Std                     float64
	KL, ClipFrac            float64
	ValueLoss               float64
	SymLoss                 float64 // mean squared command difference to the mirror image
	LR                      float64 // policy learning rate after this update
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

// SetStartLevel changes how far down floor starts go for episodes that
// start from now on. Call it between iterations.
func (t *Trainer) SetStartLevel(l float64) {
	for _, e := range t.envs {
		e.Cfg.StartLevel = l
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
		o := e.Observe()
		mu := t.Pol.Forward(o).Mu
		act := make([]float64, len(mu))
		for k := range mu {
			act[k] = mu[k] + std*rng.NormFloat64()
		}
		sm := sample{o: o, act: act, logp: logProb(act, mu, t.Pol.LogStd[0]), value: t.Val.Forward(o).V}
		r, terminal, trunc := e.Step(act)
		t.epRet[i] += r
		if trunc && !terminal {
			r += cfg.Gamma * t.Val.Forward(e.Observe()).V
		}
		if terminal || trunc {
			eps = append(eps, episode{ret: t.epRet[i], secs: e.Seconds(), fell: e.Falls > 0, assisted: e.Assisted,
				lying: e.StartedLying, gotUp: e.StoodUp > 0}) // got up and stayed up for env.StayUpSeconds
			t.epRet[i] = 0
			e.Reset()
		}
		out = append(out, sm)
		rew = append(rew, r)
		done = append(done, terminal || trunc)
	}
	next := t.Val.Forward(e.Observe()).V
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

// warmup fills empty input statistics from a short run of every
// environment, so the very first update already sees normalised inputs.
func (t *Trainer) warmup() {
	parts := make([][]*policy.Obs, len(t.envs))
	var wg sync.WaitGroup
	for i := range t.envs {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			e := t.envs[i]
			for s := 0; s < 64; s++ {
				o := e.Observe()
				parts[i] = append(parts[i], o)
				if _, terminal, trunc := e.Step(t.Pol.Forward(o).Mu); terminal || trunc {
					e.Reset()
				}
			}
		}(i)
	}
	wg.Wait()
	var all []*policy.Obs
	for _, p := range parts {
		all = append(all, p...)
	}
	t.Pol.UpdateNorms(all)
	t.Val.UpdateNorms(all)
}

// Iterate collects one batch of experience and updates both networks.
func (t *Trainer) Iterate() Stats {
	if t.Pol.TokNorm.Count == 0 {
		t.warmup()
	}
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
			if ep.lying && ep.assisted {
				st.HeldFloor++
				if ep.gotUp {
					st.HeldGotUp++
				}
			} else if ep.lying {
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
	st.KL, st.ClipFrac, st.ValueLoss, st.SymLoss, st.Epochs = t.update(batch)
	st.Std = t.Pol.Std()
	// Refresh the input statistics for the next rollout (after the update,
	// so this batch's log-probabilities stay exact while it trains).
	all := make([]*policy.Obs, len(batch))
	for i := range batch {
		all[i] = batch[i].o
	}
	t.Pol.UpdateNorms(all)
	t.Val.UpdateNorms(all)
	if c := t.Cfg; c.DesiredKL > 0 {
		// Cut hard when an update overshoots; otherwise creep back up while
		// updates stay below target. (Raising only below half the target
		// left the rate stuck wherever a few early large updates had cut
		// it, 25× below its start.)
		switch {
		case st.KL > 2*c.DesiredKL:
			t.popt.LR = math.Max(c.MinLR, t.popt.LR/1.5)
		case st.KL < c.DesiredKL/2:
			t.popt.LR = math.Min(c.MaxLR, t.popt.LR*1.5)
		case st.KL < c.DesiredKL:
			t.popt.LR = math.Min(c.MaxLR, t.popt.LR*1.1)
		}
	}
	st.LR = t.popt.LR
	st.TrainSec = time.Since(t1).Seconds()
	return st
}

type workerOut struct {
	gp          *policy.SetPolicy
	gv          *policy.SetValue
	kl, clipped float64
	vloss, sym  float64
}

// LR is the policy learning rate in use; SetLR changes it (to resume).
func (t *Trainer) LR() float64      { return t.popt.LR }
func (t *Trainer) SetLR(lr float64) { t.popt.LR = lr }

func (t *Trainer) update(batch []sample) (kl, clipFrac, vloss, symLoss float64, epochs int) {
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
		var epKL, epClip, epV, epSym float64
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
					o.kl, o.clipped, o.vloss, o.sym = 0, 0, 0, 0
					for k := w; k < len(mb); k += W {
						s := &batch[mb[k]]
						c := t.Pol.Forward(s.o)
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
						if cfg.Symmetry > 0 && mb[k]%symEvery == 0 {
							// Pull the mirror image's commands towards this
							// state's (held fixed): each token is its twin's
							// mirror, so a symmetric policy matches exactly.
							// Every symEvery-th sample, weighted up to match.
							cm := t.Pol.Forward(obs.Mirror(s.o))
							d := make([]float64, len(cm.Mu))
							for i := range d {
								diff := cm.Mu[i] - c.Mu[i]
								o.sym += symEvery * diff * diff / n
								d[i] = symEvery * cfg.Symmetry * 2 * diff / n / B
							}
							t.Pol.Backward(cm, d, o.gp)
						}
						vc := t.Val.Forward(s.o)
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
				epSym += outs[w].sym
			}
			t.popt.Step(t.Pol.Params(), gp.Params(), cfg.MaxGrad)
			t.vopt.Step(t.Val.Params(), gv.Params(), cfg.MaxGrad)
			// Keep exploring: valve noise never drops below σ = 0.2.
			t.Pol.LogStd[0] = math.Max(MinLogStd, math.Min(0, t.Pol.LogStd[0]))
		}
		n := float64(len(batch))
		kl, clipFrac, vloss, symLoss = epKL/n, epClip/n, epV/n, epSym/n
		if cfg.TargetKL > 0 && kl > cfg.TargetKL {
			break
		}
	}
	return kl, clipFrac, vloss, symLoss, epochs
}
