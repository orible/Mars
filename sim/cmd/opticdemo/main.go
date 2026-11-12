// Command opticdemo renders a short stereo sequence from the robot's
// head as it walks a scripted path through the default optic scene, runs
// it through the realism translator and the camera model, saves a GIF and
// a contact sheet, and prints speed and consistency metrics.
//
//	go run ./cmd/opticdemo -out ../docs/optic
package main

import (
	"flag"
	"fmt"
	"image"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"

	"mars/sim/body"
	"mars/sim/model"
	"mars/sim/optic"
)

func main() {
	out := flag.String("out", "../docs/optic", "output directory")
	bodyPath := flag.String("body", "bodies/biped.json", "body spec")
	frames := flag.Int("frames", 36, "frames")
	fps := flag.Float64("fps", 20, "frame rate")
	w := flag.Int("w", 128, "width per eye")
	h := flag.Int("h", 96, "height per eye")
	seed := flag.Int64("seed", 1, "scene and sensor seed")
	scale := flag.Int("scale", 2, "GIF pixel scale")
	fast := flag.Bool("fast", false, "fast RL-loop profile (64×48, light realism and camera); no images unless -images")
	images := flag.Bool("images", true, "write the GIF and still")
	flag.Parse()
	runtime.GOMAXPROCS(2)

	b, err := body.Load(*bodyPath)
	if err != nil {
		log.Fatal(err)
	}
	m, err := model.New(b)
	if err != nil {
		log.Fatal(err)
	}
	wd := model.NewWorld(m, true)
	scene := optic.DefaultScene(*seed)
	prof := optic.QualityProfile()
	prof.Rig.W, prof.Rig.H = *w, *h
	if *fast {
		prof = optic.FastProfile()
		if !isSet("images") {
			*images = false
		}
	}
	rig := prof.Rig
	eyes := optic.NewEyes(scene, m, prof, *seed)
	pipe := eyes.Pipe
	pn := func(img []optic.RGB) optic.Panel { return optic.Panel{Img: img, W: rig.W, H: rig.H} }
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}

	// Baselines for the metrics: a naive per-frame, per-eye screen-space
	// translator, and the same conditioned on its warped history.
	rng := rand.New(rand.NewSource(*seed))
	stab := &optic.Stabiliser{Alpha: 0.8}
	const dispExp = 0.45
	maxD := int(math.Ceil(rig.Disparity(0.6)))

	var (
		flClean, flReal, flOut, flNaive, flStab, nFl                 float64
		dClean, dReal, dOut, dNaive, bClean, bReal, bOut, bNaive, nD float64
		eReal, eNoShadow, eShift, eOut, eNaive, nE                   float64
		gif                                                          []*image.RGBA
		prevClean, prevReal, prevNaive, prevStab                     []optic.RGB
	)
	dt := 1 / *fps
	for i := 0; i < *frames; i++ {
		t := float64(i) * dt
		// Walk, then look down at the feet and hands for the last third.
		look := math.Max(0, math.Min(1, 3*float64(i)/float64(*frames)-2))
		optic.ScriptedWalk(wd, t, look)
		f := eyes.Step(wd, t)
		clean := optic.Display(f.L.RGB, dispExp)
		real := optic.Display(f.RadL, dispExp)
		naiveLin := append([]optic.RGB(nil), f.RadL...)
		optic.ScreenNoise(naiveLin, rig.W, rig.H, 0.12, rng)
		naive := optic.Display(naiveLin, dispExp)
		stabLin := append([]optic.RGB(nil), naiveLin...)
		stab.Apply(stabLin, f.BackL, rig.W, rig.H)
		stabbed := optic.Display(stabLin, dispExp)
		if i > 0 {
			prev := pipe.Prev()
			flClean += optic.Flicker(clean, prevClean, f.BackL, f.L)
			flReal += optic.Flicker(real, prevReal, f.BackL, f.L)
			flOut += optic.Flicker(f.OutL, prev.OutL, f.BackL, f.L)
			flNaive += optic.Flicker(naive, prevNaive, f.BackL, f.L)
			flStab += optic.Flicker(stabbed, prevStab, f.BackL, f.L)
			nFl++
		}
		prevClean, prevReal, prevNaive, prevStab = clean, real, naive, stabbed

		gt := rig.DisparityMap(f.L)
		lc := optic.Luma(clean)
		var bmOut []float32
		if i%4 == 0 {
			mask := optic.StereoMask(f.L, f.LR, lc, maxD, 0.02)
			naiveR := append([]optic.RGB(nil), f.RadR...)
			optic.ScreenNoise(naiveR, rig.W, rig.H, 0.12, rng)
			pairs := []struct {
				l, r     []optic.RGB
				mae, bad *float64
			}{
				{clean, optic.Display(f.R.RGB, dispExp), &dClean, &bClean},
				{real, optic.Display(f.RadR, dispExp), &dReal, &bReal},
				{f.OutL, f.OutR, &dOut, &bOut},
				{naive, optic.Display(naiveR, dispExp), &dNaive, &bNaive},
			}
			for pi, p := range pairs {
				est := optic.BlockMatch(optic.Luma(p.l), optic.Luma(p.r), rig.W, rig.H, maxD)
				mae, bad := optic.DisparityError(est, gt, mask)
				*p.mae += mae
				*p.bad += bad
				if pi == 2 {
					bmOut = est
				}
			}
			nD++

			edges := optic.GeometricEdges(f.L)
			d, _ := optic.EdgeDrift(lc, optic.Luma(real), edges, rig.W, rig.H, 0)
			eReal += d
			noShadow := make([]optic.RGB, rig.W*rig.H)
			pipe.Realism.Shadows = false
			pipe.Realism.Translate(f.L, noShadow)
			pipe.Realism.Shadows = true
			d, _ = optic.EdgeDrift(lc, optic.Luma(optic.Display(noShadow, dispExp)), edges, rig.W, rig.H, 0)
			eNoShadow += d
			shifted := make([]float32, len(lc))
			for y := 0; y < rig.H; y++ {
				for x := 0; x < rig.W; x++ {
					shifted[y*rig.W+x] = lc[y*rig.W+max(0, x-1)]
				}
			}
			d, _ = optic.EdgeDrift(lc, shifted, edges, rig.W, rig.H, 0)
			eShift += d
			d, _ = optic.EdgeDrift(lc, optic.Luma(f.OutL), edges, rig.W, rig.H, 0.8)
			eOut += d
			d, _ = optic.EdgeDrift(lc, optic.Luma(naive), edges, rig.W, rig.H, 0)
			eNaive += d
			nE++
		}

		row1 := []optic.Panel{pn(clean), pn(f.OutL), pn(f.OutR)}
		row2 := []optic.Panel{pn(optic.DepthImage(f.L)), pn(optic.IDImage(f.L)), pn(optic.FlowImage(f.BackL, 3))}
		gif = append(gif, optic.Sheet([][]optic.Panel{row1, row2}, *scale))
		if i == *frames/2 {
			still := [][]optic.Panel{
				{pn(clean), pn(optic.Display(f.R.RGB, dispExp)), pn(real), pn(f.OutL), pn(f.OutR)},
				{pn(optic.DepthImage(f.L)), pn(optic.IDImage(f.L)), pn(optic.FlowImage(f.BackL, 3)),
					pn(optic.DisparityImage(gt, float32(maxD))), pn(optic.DisparityImage(orBlank(bmOut, gt), float32(maxD)))},
			}
			if !*images {
				continue
			}
			if err := optic.SavePNG(filepath.Join(*out, "still.png"), optic.Sheet(still, 3)); err != nil {
				log.Fatal(err)
			}
		}
	}
	// Noise floor: the same radiance captured twice by one sensor, no
	// motion. A real sensor flickers this much in a still scene.
	var floor float64
	{
		last := pipe.Prev()
		sen := optic.NewSensor(optic.DefaultSensor(), rig.W, rig.H, *seed+99)
		lens := make([]optic.RGB, rig.W*rig.H)
		sen.Optics(last.RadL, nil, lens)
		a, b := make([]optic.RGB, len(lens)), make([]optic.RGB, len(lens))
		sen.Capture(lens, pipe.Camera.Exp.Gain, a)
		sen.Capture(lens, pipe.Camera.Exp.Gain, b)
		still := make([]optic.Flow, len(lens))
		for k := range still {
			still[k].Valid = true
		}
		floor = optic.Flicker(a, b, still, last.L)
	}
	if *images {
		if err := optic.SaveGIF(filepath.Join(*out, "stereo.gif"), gif, int(math.Round(100/(*fps)))); err != nil {
			log.Fatal(err)
		}
	}

	tm := pipe.Timing
	ms := func(d interface{ Seconds() float64 }) float64 { return 1000 * d.Seconds() / float64(tm.Frames) }
	fmt.Printf("%d stereo frames of %d×%d per eye, %.0f° HFOV, baseline %.1f cm, GOMAXPROCS 2\n",
		tm.Frames, rig.W, rig.H, rig.HFOV*180/math.Pi, rig.Baseline*100)
	fmt.Printf("ms per stereo frame: render %.2f  ground truth (flow L+R, stereo) %.2f  realism %.2f  camera %.2f  total %.2f\n",
		ms(tm.Render), ms(tm.Truth), ms(tm.Realism), ms(tm.Camera), ms(tm.Render)+ms(tm.Truth)+ms(tm.Realism)+ms(tm.Camera))
	fmt.Printf("flicker (warped diff, 0–255 luma): clean %.2f  realism %.2f  camera output %.2f (sensor-noise floor %.2f)  | naive per-frame %.2f  naive+stabiliser %.2f\n",
		flClean/nFl, flReal/nFl, flOut/nFl, floor, flNaive/nFl, flStab/nFl)
	fmt.Printf("disparity error vs truth (block matching, MAE px / >1 px): clean %.2f/%.0f%%  realism %.2f/%.0f%%  camera output %.2f/%.0f%%  naive %.2f/%.0f%%\n",
		dClean/nD, 100*bClean/nD, dReal/nD, 100*bReal/nD, dOut/nD, 100*bOut/nD, dNaive/nD, 100*bNaive/nD)
	fmt.Printf("edge drift vs clean (px, normal to the edge): realism %.3f (without cast shadows %.3f)  camera output %.3f (both smoothed σ0.8)  naive %.3f  | control: clean shifted 1 px right %.3f\n",
		eReal/nE, eNoShadow/nE, eOut/nE, eNaive/nE, eShift/nE)
	var within, across, nw, na float64
	for i, a := range scene.Objects {
		for j := i + 1; j < len(scene.Objects); j++ {
			c := optic.Cosine(a.Appearance, scene.Objects[j].Appearance)
			if a.Kind == scene.Objects[j].Kind {
				within += c
				nw++
			} else {
				across += c
				na++
			}
		}
	}
	fmt.Printf("appearance vectors (%d-d): mean cosine within a kind %.2f, across kinds %.2f\n", scene.AppDim, within/nw, across/na)
	if *images {
		fmt.Printf("wrote %s and %s\n", filepath.Join(*out, "stereo.gif"), filepath.Join(*out, "still.png"))
	}
}

func orBlank(a, b []float32) []float32 {
	if a != nil {
		return a
	}
	return b
}

func isSet(name string) bool {
	set := false
	flag.Visit(func(f *flag.Flag) { set = set || f.Name == name })
	return set
}
