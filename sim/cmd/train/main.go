// Command train runs PPO for the movement network across the biped spec
// and random bodies (cylinder count, anchors, CoG), random floor friction
// and random pushes. Tasks: stand (the stop command) and walk (follow the
// commands a scripted stand-in for the LLM loop issues while heading for
// random objectives). It writes the policy checkpoint after every
// iteration (atomic rename), so a sim server started with -policy on the
// same file shows the latest weights while training runs.
//
//	go run ./cmd/train -task walk -out checkpoints/walk.json -log runs/walk.csv
//	go run ./cmd/simserver -policy checkpoints/walk.json
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"mars/sim/body"
	"mars/sim/env"
	"mars/sim/obs"
	"mars/sim/policy"
	"mars/sim/ppo"
)

func writeJSON(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if err := json.NewEncoder(f).Encode(v); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func readJSON(path string, v any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, v)
}

func main() {
	cfg := ppo.Defaults()
	bodyPath := flag.String("body", "bodies/biped.json", "base body spec")
	task := flag.String("task", "walk", "stand (always the stop command) or walk (follow the scripted LLM stand-in towards random objectives)")
	out := flag.String("out", "checkpoints/stand.json", "policy checkpoint (value network goes next to it)")
	resume := flag.Bool("resume", false, "continue from -out and its value network")
	iters := flag.Int("iters", 300, "PPO iterations")
	seed := flag.Int64("seed", 1, "random seed")
	randomFrac := flag.Float64("random-bodies", 0.5, "fraction of episodes on random bodies")
	logPath := flag.String("log", "runs/stand.csv", "per-iteration statistics (CSV)")
	flag.IntVar(&cfg.Envs, "envs", cfg.Envs, "parallel environments")
	flag.IntVar(&cfg.Steps, "steps", cfg.Steps, "policy steps per environment per iteration")
	flag.Float64Var(&cfg.LR, "lr", cfg.LR, "policy learning rate")
	assist := flag.Float64("assist", -1, "starting harness strength 0–1 (walk default 1, stand default 0); faded automatically as assisted falls drop")
	flag.Parse()

	base, err := body.Load(*bodyPath)
	if err != nil {
		log.Fatal(err)
	}
	rng := rand.New(rand.NewSource(*seed))
	pol := policy.New(rng, obs.TokenDim, obs.GlobalDim, 32, 64)
	val := policy.NewValue(rng, obs.TokenDim, obs.GlobalDim, 32, 64)
	valPath := filepath.Join(filepath.Dir(*out), "value-"+filepath.Base(*out))
	if *resume {
		if err := readJSON(*out, pol); err != nil {
			log.Fatal(err)
		}
		if err := readJSON(valPath, val); err != nil {
			log.Fatal(err)
		}
		if len(pol.LogStd) != 1 {
			pol.LogStd = []float64{policy.DefaultLogStd}
		}
		log.Printf("resumed from %s", *out)
	}
	if pol.TokenDim != obs.TokenDim || pol.GlobalDim != obs.GlobalDim {
		log.Fatalf("checkpoint features %d/%d, simulator provides %d/%d", pol.TokenDim, pol.GlobalDim, obs.TokenDim, obs.GlobalDim)
	}

	var ecfg env.Config
	switch *task {
	case "stand":
		ecfg = env.DefaultStand()
	case "walk":
		ecfg = env.DefaultWalk()
	default:
		log.Fatalf("-task must be stand or walk, not %q", *task)
	}
	ecfg.RandomBodyFrac = *randomFrac
	if *assist >= 0 {
		ecfg.Assist = *assist
	}
	level := ecfg.Assist
	envs := make([]*env.Env, cfg.Envs)
	for i := range envs {
		envs[i] = env.New(base, ecfg, *seed*1000+int64(i))
	}
	tr := ppo.New(cfg, pol, val, envs, *seed)

	if err := os.MkdirAll(filepath.Dir(*logPath), 0o755); err != nil {
		log.Fatal(err)
	}
	csv, err := os.OpenFile(*logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		log.Fatal(err)
	}
	defer csv.Close()
	if info, _ := csv.Stat(); info.Size() == 0 {
		fmt.Fprintln(csv, "iter,samples,episodes,falls,mean_return,mean_seconds,std,kl,clip_frac,value_loss,epochs,rollout_s,train_s,assist,solo_episodes,solo_falls,solo_seconds,assisted_falls")
	}

	log.Printf("training %d envs × %d steps (%.2f s of robot time each) per iteration; %.0f%% random bodies; writing %s",
		cfg.Envs, cfg.Steps, float64(cfg.Steps)*env.PolicyDt, 100**randomFrac, *out)
	start := time.Now()
	var heldEps, heldFalls []int
	for i := 0; i < *iters; i++ {
		s := tr.Iterate()
		fmt.Fprintf(csv, "%d,%d,%d,%d,%.4f,%.4f,%.4f,%.5f,%.4f,%.4f,%d,%.2f,%.2f,%.3f,%d,%d,%.3f,%d\n",
			s.Iter, s.Samples, s.Episodes, s.Falls, s.MeanReturn, s.MeanSecs, s.Std, s.KL, s.ClipFrac, s.ValueLoss, s.Epochs, s.RolloutSec, s.TrainSec,
			level, s.SoloEpisodes, s.SoloFalls, s.SoloSecs, s.AssistedFalls)
		fallRate := 0.0
		if s.Episodes > 0 {
			fallRate = float64(s.Falls) / float64(s.Episodes)
		}
		log.Printf("iter %4d  episodes %4d  fell %3.0f%%  ep %.2f s  return %7.2f  assist %.2f  solo %d/%d fell, %.1f s  σ %.3f  kl %.4f  vloss %6.2f  %.1f+%.1f s  (%s)",
			s.Iter, s.Episodes, 100*fallRate, s.MeanSecs, s.MeanReturn, level, s.SoloFalls, s.SoloEpisodes, s.SoloSecs, s.Std, s.KL, s.ValueLoss, s.RolloutSec, s.TrainSec,
			time.Since(start).Round(time.Second))

		// Fade the helping hands as the robot stops falling while held,
		// and bring them back if it starts falling again. Judged over the
		// last 20 iterations (≥ 20 held episodes): single iterations end
		// too few episodes and fade the harness on luck.
		heldEps = append(heldEps, s.Episodes-s.SoloEpisodes)
		heldFalls = append(heldFalls, s.AssistedFalls)
		if len(heldEps) > 20 {
			heldEps, heldFalls = heldEps[1:], heldFalls[1:]
		}
		eps, falls := 0, 0
		for k := range heldEps {
			eps, falls = eps+heldEps[k], falls+heldFalls[k]
		}
		if level > 0 && len(heldEps) == 20 && eps >= 20 {
			rate := float64(falls) / float64(eps)
			switch {
			case rate < 0.15:
				level = math.Max(0, level-0.01)
			case rate > 0.4:
				level = math.Min(1, level+0.01)
			}
			tr.SetAssist(level)
		}
		if err := writeJSON(*out, pol); err != nil {
			log.Fatal(err)
		}
		if err := writeJSON(valPath, val); err != nil {
			log.Fatal(err)
		}
	}
}
