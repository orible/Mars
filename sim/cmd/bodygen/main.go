// Command bodygen writes a humanoid body spec built from adult
// anthropometry (see body.HumanSpec) and reports how each cylinder was
// sized: stroke, bore, and the torque it gives across its joint's range.
//
//	go run ./cmd/bodygen -out bodies/biped.json           # 5 ft 10 in, 225 lb
//	go run ./cmd/bodygen -height 1.65 -mass 70 -arms=false -out bodies/small.json
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"

	"mars/sim/body"
	"mars/sim/model"
)

func main() {
	s := body.DefaultHuman()
	flag.Float64Var(&s.Height, "height", s.Height, "standing height, m")
	flag.Float64Var(&s.Mass, "mass", s.Mass, "total mass, kg")
	flag.IntVar(&s.Spine, "spine", s.Spine, "0 rigid trunk, 1 lumbar, 2 lumbar and thoracic")
	flag.BoolVar(&s.Arms, "arms", s.Arms, "shoulders and elbows")
	flag.BoolVar(&s.HipYaw, "hip-yaw", s.HipYaw, "hip yaw joints")
	flag.BoolVar(&s.HipRoll, "hip-roll", s.HipRoll, "hip roll cylinders")
	flag.BoolVar(&s.DoubleKnee, "double-knee", s.DoubleKnee, "a second knee cylinder")
	flag.Float64Var(&s.Pressure, "pressure", s.Pressure, "supply pressure, Pa")
	name := flag.String("name", "biped", "body name")
	out := flag.String("out", "", "write the spec here (default: report only)")
	flag.Parse()

	b := body.Human(s)
	b.Name = *name
	if err := b.Validate(); err != nil {
		log.Fatal(err)
	}
	m, err := model.New(b)
	if err != nil {
		log.Fatal(err)
	}
	report(b, m, s)
	if *out != "" {
		data, err := json.MarshalIndent(b, "", "  ")
		if err != nil {
			log.Fatal(err)
		}
		if err := os.WriteFile(*out, append(data, '\n'), 0o644); err != nil {
			log.Fatal(err)
		}
		log.Printf("wrote %s", *out)
	}
}

func report(b body.Body, m *model.Model, s body.HumanSpec) {
	w := model.NewWorld(m, true)
	mass := 0.0
	for _, l := range b.Links {
		mass += l.Mass
	}
	fmt.Printf("%s: %.2f m, %.1f kg (%.0f lb), %d links, %d joints, %d cylinders, %d contacts, supply %.0f bar\n",
		b.Name, s.Height, mass, mass/0.45359, len(b.Links), m.NDof(), len(b.Cylinders), len(m.Contacts), s.Pressure/1e5)
	fmt.Printf("standing pelvis height %.3f m\n\n", w.StandHeight())
	fmt.Println("cylinder            bore  stroke  retr.  joint            range (deg)    arm mm min..max   torque N·m (rod side) neutral / weakest")
	q := make([]float64, m.NDof())
	ps := m.NewPose()
	for k, c := range m.Cyl {
		// The joint it mostly drives: the spanned joint with the largest neutral arm.
		m.FK(q, ps)
		d := make([]float64, len(c.Span))
		m.CylLength(ps, k, d)
		best := 0
		for i := range d {
			if math.Abs(d[i]) > math.Abs(d[best]) {
				best = i
			}
		}
		j := c.Span[best]
		lo, hi := m.Lo[j+1], m.Hi[j+1]
		amin, amax := math.Inf(1), 0.0
		for n := 0; n <= 40; n++ {
			q[j] = lo + (hi-lo)*float64(n)/40
			m.FK(q, ps)
			m.CylLength(ps, k, d)
			a := math.Abs(d[best])
			amin, amax = math.Min(amin, a), math.Max(amax, a)
		}
		q[j] = 0
		f := b.Supply.Pressure * c.P.RodArea
		dia := 2 * math.Sqrt(c.P.BoreArea/math.Pi) * 1000
		fmt.Printf("%-18s %4.0fmm %5.3f  %5.3f  %-14s %5.0f..%-5.0f  %5.1f..%-5.1f        %6.0f / %-6.0f\n",
			c.Name, dia, c.P.Stroke, c.Retracted, m.Names[j+1], lo*180/math.Pi, hi*180/math.Pi,
			amin*1000, amax*1000, f*math.Abs(d0(m, k, best)), f*amin)
	}
	if v := m.CornerViolations(); len(v) > 0 {
		fmt.Println()
		for _, s := range v {
			fmt.Println("  ", s)
		}
	}
}

// d0 is cylinder k's neutral moment arm about its i-th spanned joint.
func d0(m *model.Model, k, i int) float64 {
	ps := m.NewPose()
	m.FK(make([]float64, m.NDof()), ps)
	d := make([]float64, len(m.Cyl[k].Span))
	m.CylLength(ps, k, d)
	return d[i]
}
