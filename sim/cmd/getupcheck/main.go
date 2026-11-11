// Command getupcheck runs the current policy on unheld floor starts and
// shows how each attempt to get up went: start, best and final pelvis
// height (percent of standing), trunk lean (+ forward, − back) over time,
// and whether it ended up. Use it to see how getting up fails.
//
//	go run ./cmd/getupcheck -level 0.05 -n 12
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"

	"mars/sim/body"
	"mars/sim/env"
	"mars/sim/policy"
)

func main() {
	ckpt := flag.String("policy", "checkpoints/policy.json", "policy checkpoint")
	bodyPath := flag.String("body", "bodies/biped.json", "body spec")
	level := flag.Float64("level", 0.05, "floor start level: 0 squat … 1 lying flat")
	n := flag.Int("n", 12, "attempts")
	secs := flag.Float64("seconds", 10, "length of each attempt")
	flag.Parse()

	f, err := os.Open(*ckpt)
	if err != nil {
		log.Fatal(err)
	}
	pol, err := policy.Load(f)
	f.Close()
	if err != nil {
		log.Fatal(err)
	}
	b, err := body.Load(*bodyPath)
	if err != nil {
		log.Fatal(err)
	}
	sc, _ := env.LoadScenario("getup")
	sc.RandomBodyFrac, sc.Assist, sc.StartLevel, sc.Curriculum = 0, 0, *level, false
	sc.EpisodeSeconds = *secs
	up := 0
	for seed := 1; seed <= *n; seed++ {
		e := env.New(b, sc, int64(seed))
		w := e.W
		h0 := w.StandH()
		start, best, firstUp := w.Base.P[1]/h0, 0.0, -1.0
		trace := ""
		for s := 0; float64(s)*env.PolicyDt < *secs; s++ {
			tok, g := e.Observe()
			e.Step(pol.Act(tok, g))
			h := w.Base.P[1] / h0
			best = math.Max(best, h)
			if firstUp < 0 && w.Upright() {
				firstUp = e.Seconds()
			}
			if s%50 == 0 {
				trace += fmt.Sprintf(" %3.0f%%/%+4.0f°", 100*h, env.TrunkLean(w)*180/math.Pi)
			}
		}
		ended := w.Upright()
		if ended {
			up++
		}
		res := "down"
		if ended {
			res = fmt.Sprintf("UP (first at %.1f s)", firstUp)
		} else if firstUp >= 0 {
			res = fmt.Sprintf("was up at %.1f s, fell", firstUp)
		}
		fmt.Printf("attempt %2d: start %3.0f%% best %3.0f%% end %3.0f%%  %s\n   height/lean every 0.5 s:%s\n",
			seed, 100*start, 100*best, 100*w.Base.P[1]/h0, res, trace)
	}
	fmt.Printf("\n%d of %d attempts ended standing\n", up, *n)
}
