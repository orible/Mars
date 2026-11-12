// Command train runs PPO for the movement network in a training scenario
// (-scenario): a built-in (getup, stand, walk, life) or a JSON file. Every
// scenario trains the same network across the biped spec and random bodies
// (cylinder count, anchors, CoG), so skills carry from one to the next:
//
//	go run ./cmd/train -scenario getup -iters 400
//	go run ./cmd/train -scenario walk -resume
//	go run ./cmd/train -scenario life -resume
//	go run ./cmd/train -scenario scenarios/getup-then-walk.json -resume
//
// It writes the policy checkpoint after every iteration (atomic rename),
// so a sim server started with -policy on the same file shows the latest
// weights while training runs.
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
	"runtime/pprof"
	"strings"
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

const csvHeader = "iter,samples,episodes,falls,mean_return,mean_seconds,std,kl,clip_frac,value_loss,epochs,rollout_s,train_s,assist,solo_episodes,solo_falls,solo_seconds,assisted_falls,lying_episodes,got_up,start_level,held_floor,held_got_up,sym_loss,lr"

// scenarioFile returns the scenario argument if it is a file, made
// absolute so another process can load it.
func scenarioFile(arg string) string {
	if !strings.HasSuffix(arg, ".json") {
		return ""
	}
	if abs, err := filepath.Abs(arg); err == nil {
		return abs
	}
	return arg
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
	scenarioArg := flag.String("scenario", "life", "what to practise: "+strings.Join(env.ScenarioNames(), ", ")+", or a scenario .json file")
	out := flag.String("out", "checkpoints/policy.json", "policy checkpoint shared by all scenarios (value network goes next to it)")
	resume := flag.Bool("resume", false, "continue from -out and its value network")
	iters := flag.Int("iters", 300, "PPO iterations")
	seed := flag.Int64("seed", 1, "random seed")
	randomFrac := flag.Float64("random-bodies", -1, "fraction of episodes on random bodies (default: the scenario's)")
	logPath := flag.String("log", "", "per-iteration statistics (CSV); default runs/<scenario>.csv")
	flag.IntVar(&cfg.Envs, "envs", cfg.Envs, "parallel environments")
	flag.IntVar(&cfg.Steps, "steps", cfg.Steps, "policy steps per environment per iteration")
	flag.Float64Var(&cfg.LR, "lr", cfg.LR, "policy learning rate")
	startLevel := flag.Float64("start-level", -1, "how far down floor starts go, 0 crouched … 1 lying flat (default: the scenario's); moved automatically in curriculum scenarios")
	assist := flag.Float64("assist", -1, "starting harness strength 0–1 (walk default 1, stand default 0); faded automatically as assisted falls drop")
	embed := flag.Int("embed", 64, "embedding width of a new network")
	hidden := flag.Int("hidden", 128, "hidden layer width of a new network")
	flag.Float64Var(&cfg.Symmetry, "symmetry", cfg.Symmetry, "weight of the left–right symmetry loss (0 = off)")
	noSensors := flag.Bool("no-foot-sensors", false, "leave the sole pads and tendon sensors out (to compare)")
	rough := flag.Float64("rough-floor", -1, "largest floor bump in m (default: the scenario's; 0 = flat)")
	flag.IntVar(&cfg.Workers, "workers", cfg.Workers, "goroutines computing gradients (default: logical CPUs; physical cores can be faster)")
	flag.IntVar(&cfg.Epochs, "epochs", cfg.Epochs, "passes over each batch per iteration")
	cpuProfile := flag.String("cpuprofile", "", "write a CPU profile here (go tool pprof) to see where training time goes")
	attn := flag.Bool("attn", false, "let the cylinder tokens attend to each other before ρ (added to a resumed network without changing its outputs; a checkpoint that has it keeps it)")
	attnHeads := flag.Int("attn-heads", 4, "attention heads (with -attn)")
	attnDim := flag.Int("attn-dim", 8, "attention width per head (with -attn)")
	flag.Parse()
	if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
			log.Fatal(err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal(err)
		}
		defer pprof.StopCPUProfile()
	}

	base, err := body.Load(*bodyPath)
	if err != nil {
		log.Fatal(err)
	}
	rng := rand.New(rand.NewSource(*seed))
	pol := policy.New(rng, obs.TokenDim, obs.PartDim, obs.SensorDim, obs.GlobalDim, *embed, *hidden)
	pol.ObsVersion = obs.Version
	val := policy.NewValue(rng, obs.TokenDim, obs.PartDim, obs.SensorDim, obs.GlobalDim+env.PrivDim, *embed, *hidden)
	valPath := filepath.Join(filepath.Dir(*out), "value-"+filepath.Base(*out))
	if *resume {
		pol = &policy.SetPolicy{}
		if err := readJSON(*out, pol); err != nil {
			log.Fatal(err)
		}
		if err := pol.Check(); err != nil {
			log.Fatalf("%s: %v", *out, err)
		}
		if pol.ObsVersion != obs.Version {
			log.Fatalf("%s was trained on observation layout %d, the simulator now provides layout %d; train a new network (drop -resume)", *out, pol.ObsVersion, obs.Version)
		}
		val = &policy.SetValue{}
		if err := readJSON(valPath, val); err != nil {
			log.Fatal(err)
		}
		if val.GlobalDim != obs.GlobalDim+env.PrivDim || val.PartDim != obs.PartDim || val.SensorDim != obs.SensorDim {
			log.Fatalf("%s does not match the critic's inputs; train a new network (drop -resume)", valPath)
		}
		pol.LogStd[0] = math.Max(ppo.MinLogStd, pol.LogStd[0]) // restore exploration if it had collapsed
		log.Printf("resumed from %s (exploration σ %.2f)", *out, pol.Std())
	}
	if *attn && pol.Attn == nil {
		pol.AddAttention(rng, *attnHeads, *attnDim)
		log.Printf("added attention over the cylinder tokens (%d heads × %d); outputs unchanged until it trains", *attnHeads, *attnDim)
	}
	if pol.TokenDim != obs.TokenDim || pol.PartDim != obs.PartDim || pol.GlobalDim != obs.GlobalDim {
		log.Fatalf("checkpoint features %d/%d/%d, simulator provides %d/%d/%d", pol.TokenDim, pol.PartDim, pol.GlobalDim, obs.TokenDim, obs.PartDim, obs.GlobalDim)
	}

	ecfg, err := env.LoadScenario(*scenarioArg)
	if err != nil {
		log.Fatal(err)
	}
	if *randomFrac >= 0 {
		ecfg.RandomBodyFrac = *randomFrac
	}
	ecfg.NoSensors = *noSensors
	if *rough >= 0 {
		ecfg.RoughFloor = *rough
	}
	if *logPath == "" {
		*logPath = filepath.Join("runs", filepath.Base(strings.TrimSuffix(ecfg.Name, ".json"))+".csv")
	}
	log.Printf("scenario %q: %s", ecfg.Name, ecfg.Description)
	if *assist >= 0 {
		ecfg.Assist = *assist
	}
	if *startLevel >= 0 {
		ecfg.StartLevel = *startLevel
	}
	level := ecfg.Assist
	floorLevel := ecfg.StartLevel
	// Resuming the same scenario carries on where it stopped: the
	// curriculum's start level and harness strength, the iteration count
	// and the learning rate (flags given on the command line still win).
	var prev env.TrainingStatus
	resumed := *resume && readJSON(env.StatusPath(*out), &prev) == nil && prev.Scenario == ecfg.Name
	if resumed {
		set := map[string]bool{}
		flag.Visit(func(f *flag.Flag) { set[f.Name] = true })
		if !set["start-level"] {
			floorLevel = prev.StartLevel
		}
		if !set["assist"] {
			level = prev.Assist
		}
		ecfg.StartLevel, ecfg.Assist = floorLevel, level
		log.Printf("resuming %q at iteration %d: start level %.2f, assist %.2f", ecfg.Name, prev.Iter, floorLevel, level)
	}
	envs := make([]*env.Env, cfg.Envs)
	for i := range envs {
		envs[i] = env.New(base, ecfg, *seed*1000+int64(i))
	}
	tr := ppo.New(cfg, pol, val, envs, *seed)
	if resumed {
		tr.Iter = prev.Iter
		if prev.LR > 0 {
			tr.SetLR(prev.LR)
		}
	}

	if err := os.MkdirAll(filepath.Dir(*logPath), 0o755); err != nil {
		log.Fatal(err)
	}
	// A log written with different columns is moved aside, not appended to.
	if old, err := os.ReadFile(*logPath); err == nil && len(old) > 0 && !strings.HasPrefix(string(old), csvHeader+"\n") {
		aside := strings.TrimSuffix(*logPath, ".csv") + time.Now().Format("-20060102-150405") + ".csv"
		if err := os.Rename(*logPath, aside); err != nil {
			log.Fatal(err)
		}
		log.Printf("%s had different columns; moved it to %s", *logPath, aside)
	}
	csv, err := os.OpenFile(*logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		log.Fatal(err)
	}
	defer csv.Close()
	if info, _ := csv.Stat(); info.Size() == 0 {
		fmt.Fprintln(csv, csvHeader)
	}

	log.Printf("training %d envs × %d steps (%.2f s of robot time each) per iteration; %.0f%% random bodies; writing %s and %s",
		cfg.Envs, cfg.Steps, float64(cfg.Steps)*env.PolicyDt, 100*ecfg.RandomBodyFrac, *out, *logPath)
	start := time.Now()
	var heldEps, heldFalls, downEps, upEps, hfEps, hfUp []int
	for i := 0; i < *iters; i++ {
		s := tr.Iterate()
		fmt.Fprintf(csv, "%d,%d,%d,%d,%.4f,%.4f,%.4f,%.5f,%.4f,%.4f,%d,%.2f,%.2f,%.3f,%d,%d,%.3f,%d,%d,%d,%.3f,%d,%d,%.5f,%.2e\n",
			s.Iter, s.Samples, s.Episodes, s.Falls, s.MeanReturn, s.MeanSecs, s.Std, s.KL, s.ClipFrac, s.ValueLoss, s.Epochs, s.RolloutSec, s.TrainSec,
			level, s.SoloEpisodes, s.SoloFalls, s.SoloSecs, s.AssistedFalls, s.LyingEpisodes, s.GotUp, floorLevel, s.HeldFloor, s.HeldGotUp, s.SymLoss, s.LR)
		fallRate := 0.0
		if s.Episodes > 0 {
			fallRate = float64(s.Falls) / float64(s.Episodes)
		}
		log.Printf("iter %4d  eps %3d  fell %3.0f%%  return %7.2f  floor-starts got up %d/%d (level %.2f), held %d/%d  assist %.2f  solo %d/%d fell  σ %.3f  kl %.4f  lr %.1e  sym %.4f  vloss %6.2f  %.1f+%.1f s  (%s)",
			s.Iter, s.Episodes, 100*fallRate, s.MeanReturn, s.GotUp, s.LyingEpisodes, floorLevel, s.HeldGotUp, s.HeldFloor, level, s.SoloFalls, s.SoloEpisodes, s.Std, s.KL, s.LR, s.SymLoss, s.ValueLoss, s.RolloutSec, s.TrainSec,
			time.Since(start).Round(time.Second))

		// Reverse curriculum for getting up: start lower once it gets up
		// from where it starts now, and higher again if it stops managing.
		downEps = append(downEps, s.LyingEpisodes)
		upEps = append(upEps, s.GotUp)
		if len(downEps) > 20 {
			downEps, upEps = downEps[1:], upEps[1:]
		}
		if ecfg.Curriculum {
			d, u := 0, 0
			for k := range downEps {
				d, u = d+downEps[k], u+upEps[k]
			}
			if len(downEps) == 20 && d >= 20 {
				// After a step, judge the new level only on episodes
				// started there: clear the windows so old successes at
				// an easier level cannot push it on (they once carried
				// it from 0.12 to 0.58 in 28 iterations, past what the
				// policy could do).
				rate := float64(u) / float64(d)
				was := floorLevel
				switch {
				case rate > 0.5:
					floorLevel = math.Min(1, floorLevel+0.02)
				case rate < 0.2:
					floorLevel = math.Max(0.05, floorLevel-0.02)
				}
				if floorLevel != was {
					downEps, upEps = downEps[:0], upEps[:0]
					hfEps, hfUp = hfEps[:0], hfUp[:0]
				}
				tr.SetStartLevel(floorLevel)
			}
		}

		// Fade the helping hands as the robot stops falling while held,
		// and bring them back if it starts falling again. Judged over the
		// last 20 iterations (≥ 20 held episodes): single iterations end
		// too few episodes and fade the harness on luck.
		heldEps = append(heldEps, s.Episodes-s.SoloEpisodes-s.LyingEpisodes-s.HeldFloor)
		heldFalls = append(heldFalls, s.AssistedFalls)
		if len(heldEps) > 20 {
			heldEps, heldFalls = heldEps[1:], heldFalls[1:]
		}
		eps, falls := 0, 0
		for k := range heldEps {
			eps, falls = eps+heldEps[k], falls+heldFalls[k]
		}
		// With held floor starts, fade on how often held get-ups succeed.
		hfEps, hfUp = append(hfEps, s.HeldFloor), append(hfUp, s.HeldGotUp)
		if len(hfEps) > 20 {
			hfEps, hfUp = hfEps[1:], hfUp[1:]
		}
		hd, hu := 0, 0
		for k := range hfEps {
			hd, hu = hd+hfEps[k], hu+hfUp[k]
		}
		if ecfg.AssistFloor && level > 0 && len(hfEps) == 20 && hd >= 20 {
			rate := float64(hu) / float64(hd)
			switch {
			case rate > 0.6:
				level = math.Max(0, level-0.01)
			case rate < 0.3:
				level = math.Min(1, level+0.01)
			}
			tr.SetAssist(level)
		} else if level > 0 && len(heldEps) == 20 && eps >= 20 {
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
		// What training is practising right now, so the sim server can
		// follow along (-follow, or "Follow training" in the observer).
		if err := writeJSON(env.StatusPath(*out), env.TrainingStatus{
			Scenario: ecfg.Name, Description: ecfg.Description, File: scenarioFile(*scenarioArg),
			Iter: s.Iter, Elapsed: time.Since(start).Round(time.Second).String(),
			StartLevel: floorLevel, Assist: level,
			Episodes: s.Episodes, Falls: s.Falls, FloorStarts: s.LyingEpisodes, GotUp: s.GotUp,
			HeldFloor: s.HeldFloor, HeldGotUp: s.HeldGotUp,
			SoloEpisodes: s.SoloEpisodes, SoloFalls: s.SoloFalls, MeanReturn: s.MeanReturn, LR: s.LR,
		}); err != nil {
			log.Fatal(err)
		}
		if err := writeJSON(valPath, val); err != nil {
			log.Fatal(err)
		}
	}
}
