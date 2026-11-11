// Command mocapref turns a human motion capture recording (Acclaim
// ASF/AMC, as in the CMU database) into a robot motion reference.
//
// First look at the recording:
//
//	go run ./cmd/mocapref -asf mocap/cmu-114/114.asf -amc mocap/cmu-114/114_02.amc -dump 0.25
//
// then cut out the part the legs do and write keyframes:
//
//	go run ./cmd/mocapref -asf ... -amc ... -from 6.5 -to 11 -every 0.5 -out env/references/kneel-to-stand.json
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"

	"mars/sim/env"
	"mars/sim/mocap"
)

// leanSub removes the standing lean offset. Lean is signed by which way
// the subject faces (+ face down, − face up) and reaches ±180 lying
// flat, so the offset is taken off the magnitude: upright leans shrink
// towards 0 and lying stays near ±90.
func leanSub(lean, neutral float64) float64 {
	m := math.Abs(lean)
	m = math.Max(0, m-neutral*math.Max(0, 1-m/180))
	if math.Abs(lean) < neutral { // standing a little more upright than neutral
		return 0
	}
	return math.Copysign(m, lean)
}

func main() {
	asf := flag.String("asf", "", "skeleton (.asf)")
	amc := flag.String("amc", "", "motion (.amc)")
	fps := flag.Float64("fps", 120, "frames per second of the recording")
	dump := flag.Float64("dump", 0, "print the features every this many seconds and exit")
	from := flag.Float64("from", 0, "start of the excerpt (s)")
	to := flag.Float64("to", 0, "end of the excerpt (s); 0 = end of recording")
	every := flag.Float64("every", 0.5, "keyframe spacing (s)")
	ankleSign := flag.Float64("ankle-sign", 1, "+1 or −1: sign that makes foot rx (relative to standing) dorsiflexion")
	name := flag.String("name", "", "reference name")
	desc := flag.String("desc", "", "reference description")
	source := flag.String("source", "", "attribution for the recording")
	out := flag.String("out", "", "write the reference JSON here")
	auto := flag.Bool("auto", false, "find the rise from the floor automatically instead of -from/-to")
	noAnkle := flag.Bool("ignore-ankle", false, "mark the reference's ankle data as untrustworthy (left out of tracking)")
	standFrom := flag.Float64("stand-from", 0, "start of a window where the subject stands still (s)")
	standTo := flag.Float64("stand-to", 0, "end of that window (s); 0 = the last second of the recording")
	flag.Parse()

	sk, err := mocap.ReadASF(*asf)
	if err != nil {
		log.Fatal(err)
	}
	frames, err := mocap.ReadAMC(*amc, sk)
	if err != nil {
		log.Fatal(err)
	}
	if len(frames) < 2 {
		log.Fatal("recording has fewer than two frames")
	}
	dur := float64(len(frames)-1) / *fps
	// Standing calibration: the subject's angles while standing still are
	// her neutral pose (marker placement and posture make them non-zero),
	// so every angle is measured relative to their mean over the standing
	// window, and height relative to the highest pelvis in it.
	s0, s1 := *standFrom, *standTo
	if s1 <= 0 {
		// Default: ±0.15 s around the highest pelvis in the recording,
		// when the legs are straightest (subjects shift about afterwards).
		top, topY := 0, math.Inf(-1)
		for i, f := range frames {
			if y := f.Get("root", "ty"); y > topY {
				top, topY = i, y
			}
		}
		t := float64(top) / *fps
		s0, s1 = math.Max(0, t-0.15), math.Min(dur, t+0.15)
	}
	win := frames[int(s0**fps):int(math.Min(float64(len(frames)), s1**fps+1))]
	standY := 0.0
	for _, f := range win {
		standY = math.Max(standY, f.Get("root", "ty"))
	}
	feats := mocap.Extract(frames, standY, [2]float64{})
	var neutral mocap.Features
	n := 0.0
	for i := int(s0 * *fps); i < int(s0**fps)+len(win); i++ {
		x := feats[i]
		neutral.TrunkLean += x.TrunkLean
		for s := 0; s < 2; s++ {
			neutral.Hip[s] += x.Hip[s]
			neutral.Knee[s] += x.Knee[s]
			neutral.Ankle[s] += x.Ankle[s]
		}
		n++
	}
	neutral.TrunkLean /= n
	for s := 0; s < 2; s++ {
		neutral.Hip[s] /= n
		neutral.Knee[s] /= n
		neutral.Ankle[s] /= n
	}
	log.Printf("standing neutral (%.1f–%.1f s): lean %.0f, hip %.0f/%.0f, knee %.0f/%.0f, foot rx %.0f/%.0f",
		s0, s1, neutral.TrunkLean, neutral.Hip[0], neutral.Hip[1], neutral.Knee[0], neutral.Knee[1], neutral.Ankle[0], neutral.Ankle[1])
	for i := range feats {
		x := &feats[i]
		x.TrunkLean = leanSub(x.TrunkLean, neutral.TrunkLean)
		for s := 0; s < 2; s++ {
			x.Hip[s] -= neutral.Hip[s]
			x.Knee[s] -= neutral.Knee[s]
			x.Ankle[s] -= neutral.Ankle[s]
		}
	}
	log.Printf("%d frames, %.1f s; standing root height %.1f units", len(frames), dur, standY)

	if *dump > 0 {
		fmt.Println("    t  height  lean |  hipL  kneeL ankleL |  hipR  kneeR ankleR")
		for t := 0.0; t <= dur; t += *dump {
			x := feats[int(t**fps)]
			fmt.Printf("%5.2f  %5.2f %5.0f | %5.0f %6.0f %6.0f | %5.0f %6.0f %6.0f\n",
				t, x.Height, x.TrunkLean, x.Hip[0], x.Knee[0], x.Ankle[0], x.Hip[1], x.Knee[1], x.Ankle[1])
		}
		return
	}

	end := *to
	if *auto {
		// The rise: from the last moment near the floor before the
		// subject first reaches 95% of standing height, to that moment.
		up := -1
		for i, x := range feats {
			if x.Height >= 0.95 {
				up = i
				break
			}
		}
		if up < 0 {
			log.Fatal("-auto: the subject never reaches standing height")
		}
		low := math.Inf(1)
		for _, x := range feats[:up] {
			low = math.Min(low, x.Height)
		}
		start := 0
		for i := up; i >= 0; i-- {
			if feats[i].Height <= low+0.03 {
				start = i
				break
			}
		}
		*from = math.Max(0, float64(start) / *fps - 0.25)
		// Finish at the straightest moment: the highest pelvis within 2 s
		// of reaching standing height.
		top := up
		for i := up; i < len(feats) && i < up+int(2**fps); i++ {
			if feats[i].Height > feats[top].Height {
				top = i
			}
		}
		end = float64(top) / *fps
		log.Printf("-auto: rise from %.2f s (pelvis %.0f%%) to %.2f s", *from, 100*low, end)
	}
	if end <= 0 || end > dur {
		end = dur
	}
	if *from < 0 || *from >= end || *every <= 0 {
		log.Fatal("need 0 ≤ -from < -to and -every > 0")
	}
	ref := env.Reference{Name: *name, Description: *desc, IgnoreAnkle: *noAnkle, Sources: []string{*source,
		fmt.Sprintf("Extracted by cmd/mocapref from %s, %.2f–%.2f s, keyframes every %.2f s; pelvis lean and height from the root; hip = −femur rx, knee = tibia rx, ankle = %+.0f × foot rx; all relative to the subject's standing pose (%.1f–%.1f s).", *amc, *from, end, *every, *ankleSign, s0, s1)}}
	if end-*from < 2**every {
		log.Fatalf("excerpt %.2f–%.2f s is too short for keyframes every %.2f s", *from, end, *every)
	}
	for t := *from; t <= end+1e-9; t += *every {
		x := feats[int(math.Min(float64(len(feats)-1), math.Round(t**fps)))]
		ref.Keyframes = append(ref.Keyframes, env.Keyframe{
			T: (t - *from) / (end - *from), TrunkLean: x.TrunkLean, Height: x.Height,
			Legs: &[2]env.LegAngles{
				{Hip: x.Hip[0], Knee: x.Knee[0], Ankle: *ankleSign * x.Ankle[0]},
				{Hip: x.Hip[1], Knee: x.Knee[1], Ankle: *ankleSign * x.Ankle[1]},
			},
		})
	}
	data, err := json.MarshalIndent(ref, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(*out, append(data, '\n'), 0o644); err != nil {
		log.Fatal(err)
	}
	log.Printf("wrote %d keyframes to %s", len(ref.Keyframes), *out)
}
