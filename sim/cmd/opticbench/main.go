// Command opticbench measures vision throughput the way the RL loop would
// use it: a batch of environments (each with its own scene copy, body and
// temporal state) stepped together, at 1, 2 and 4 threads. It also saves a
// still of an environment looking down at its own body.
//
//	go run ./cmd/opticbench -envs 24 -profile fast
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"runtime"
	"strconv"
	"strings"
	"time"

	"mars/sim/body"
	"mars/sim/model"
	"mars/sim/optic"
)

func main() {
	bodyPath := flag.String("body", "bodies/biped.json", "body spec")
	envs := flag.Int("envs", 24, "environments")
	steps := flag.Int("steps", 30, "timed steps per thread count")
	threads := flag.String("threads", "1,2,4", "thread counts")
	prof := flag.String("profile", "fast", "fast (64×48, RL loop) or quality (128×96, datasets)")
	w := flag.Int("w", 0, "override width per eye")
	h := flag.Int("h", 0, "override height per eye")
	still := flag.String("still", "", "save a look-down still (PNG) here")
	flag.Parse()

	b, err := body.Load(*bodyPath)
	if err != nil {
		log.Fatal(err)
	}
	m, err := model.New(b)
	if err != nil {
		log.Fatal(err)
	}
	p := optic.FastProfile()
	if *prof == "quality" {
		p = optic.QualityProfile()
	}
	if *w > 0 {
		p.Rig.W, p.Rig.H = *w, *h
	}
	scene := optic.DefaultScene(1)
	worlds := make([]*model.World, *envs)
	for i := range worlds {
		worlds[i] = model.NewWorld(m, true)
	}
	ts := make([]float64, *envs)
	pose := func(step int) {
		for i, wd := range worlds {
			ts[i] = float64(step)*0.05 + 0.37*float64(i)
			look := 0.5 + 0.5*math.Sin(0.7*ts[i]) // some envs look down at the body
			optic.ScriptedWalk(wd, ts[i], look)
		}
	}
	fmt.Printf("%s profile, %d×%d per eye, %d envs, host has %d logical CPUs\n", *prof, p.Rig.W, p.Rig.H, *envs, runtime.NumCPU())
	for _, s := range strings.Split(*threads, ",") {
		th, _ := strconv.Atoi(s)
		runtime.GOMAXPROCS(th)
		bt := optic.NewBatch(*envs, scene, m, p, th, 1)
		for k := 0; k < 2; k++ { // warm up
			pose(k)
			bt.Step(worlds, ts)
		}
		var wall time.Duration
		for k := 0; k < *steps; k++ {
			pose(k + 2)
			t0 := time.Now()
			bt.Step(worlds, ts)
			wall += time.Since(t0)
		}
		tm := bt.Timing()
		n := float64(tm.Frames)
		ms := func(d time.Duration) float64 { return 1000 * d.Seconds() / n }
		fps := float64(*envs**steps) / wall.Seconds()
		fmt.Printf("threads %d: %6.0f stereo frames/s (%.1f ms per batch step) | CPU ms per stereo frame: render %.2f truth %.2f realism %.2f camera %.2f\n",
			th, fps, 1000*wall.Seconds()/float64(*steps), ms(tm.Render), ms(tm.Truth), ms(tm.Realism), ms(tm.Camera))
	}
	if *still != "" {
		e := optic.NewEyes(scene, m, p, 5)
		var f *optic.StereoFrame
		for k := 0; k < 3; k++ {
			optic.ScriptedWalk(worlds[0], 0.6+0.05*float64(k), 1)
			f = e.Step(worlds[0], 0.6+0.05*float64(k))
		}
		W, H := p.Rig.W, p.Rig.H
		pn := func(img []optic.RGB) optic.Panel { return optic.Panel{Img: img, W: W, H: H} }
		sheet := optic.Sheet([][]optic.Panel{{pn(optic.Display(f.L.RGB, 0.45)), pn(f.OutL), pn(f.OutR)},
			{pn(optic.PartImage(f.L)), pn(optic.DepthImage(f.L)), pn(optic.FlowImage(f.BackL, 3))}}, 384/W)
		if err := optic.SavePNG(*still, sheet); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("eye token: %+v\n", f.Token)
	}
}
