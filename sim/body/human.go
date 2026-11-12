package body

import (
	"fmt"
	"math"
	"math/rand"

	"mars/sim/geom"
	"mars/sim/hydraulics"
)

// HumanSpec describes a full-size humanoid built from adult male
// anthropometry (Drillis & Contini segment lengths, Dempster/Winter mass
// fractions and radii of gyration): legs, an optional spine and optional
// arms, every joint driven by cylinders sized for a share of the body's
// weight. Human(DefaultHuman()) is bodies/biped.json; Random draws
// variations of it for morphology randomisation.
type HumanSpec struct {
	Height float64 // m, standing height (top of the head)
	Mass   float64 // kg, total
	// Spine: 0 = rigid trunk, 1 = lumbar (pitch and lateral bend),
	// 2 = lumbar plus thoracic (twist and pitch).
	Spine      int
	Arms       bool // shoulders (flexion, abduction) and elbows
	HipYaw     bool
	HipRoll    bool
	DoubleKnee bool    // a second, hamstring-style knee cylinder
	Pressure   float64 // Pa, supply
}

// DefaultHuman is 5 ft 10 in (1.778 m) and 225 lb (102.06 kg) with a
// two-stage spine and arms, on a 210 bar (3000 psi) supply.
func DefaultHuman() HumanSpec {
	return HumanSpec{Height: 1.778, Mass: 102.06, Spine: 2, Arms: true, HipYaw: true, HipRoll: true, Pressure: 21e6}
}

// Human builds the spec with mean proportions and symmetric sides.
func Human(s HumanSpec) Body { return buildHuman(s, nil) }

// part is a piece of mass placed in the neutral pose (pelvis frame).
type part struct {
	m   float64
	com geom.V3
	i   geom.V3 // principal moments about com, along the pelvis axes
}

// merge lumps parts into one link whose frame origin is at o: total mass,
// centre of mass relative to o, and the moments about that centre
// (parallel-axis terms included, products of inertia ignored).
func merge(o geom.V3, ps ...part) (float64, geom.V3, geom.V3) {
	var m float64
	var c geom.V3
	for _, p := range ps {
		m += p.m
		c = c.Add(p.com.Scale(p.m))
	}
	c = c.Scale(1 / m)
	var in geom.V3
	for _, p := range ps {
		d := p.com.Sub(c)
		in = in.Add(p.i).Add(geom.V3{p.m * (d[1]*d[1] + d[2]*d[2]), p.m * (d[0]*d[0] + d[2]*d[2]), p.m * (d[0]*d[0] + d[1]*d[1])})
	}
	return m, c.Sub(o), in
}

func buildHuman(s HumanSpec, rng *rand.Rand) Body {
	H, M, P := s.Height, s.Mass, s.Pressure
	k := H / 1.778 // length scale against the reference man
	// vary perturbs a proportion by up to ±rel (random bodies only).
	vary := func(x, rel float64) float64 {
		if rng == nil {
			return x
		}
		return x * (1 + rel*(2*rng.Float64()-1))
	}
	u := func(lo, hi float64) float64 {
		if rng == nil {
			return (lo + hi) / 2
		}
		return lo + rng.Float64()*(hi-lo)
	}
	jit := func(v geom.V3) geom.V3 {
		if rng == nil {
			return v
		}
		e := 0.008 * k
		return v.Add(geom.V3{u(-e, e), u(-e, e), u(-e, e)})
	}

	// Lengths (m). Pelvis frame origin is the lumbar joint (L5/S1), 8 cm
	// above the hip joint centres.
	hipDrop := 0.08 * k
	hw := vary(0.09*k, 0.12)  // half the hip joint spacing
	abd := vary(0.20*k, 0.08) // lumbar to thoracic joint
	cs := vary(0.232*k, 0.06) // thoracic joint to shoulder joint height
	headTop := cs + 0.324*k   // thoracic joint to top of head
	sw := vary(0.19*k, 0.08)  // half the shoulder joint spacing
	T := geom.V3{0, abd, 0}   // thoracic joint, pelvis frame

	// Mass fractions: pelvis .142, abdomen .139, thorax .216, head and neck
	// .081, upper arm .028, forearm .016, hand .006, thigh .100, shank
	// .0465, foot .0145. Light joint brackets come out of their
	// neighbours.
	mf := func(f float64) float64 { return vary(f*M, 0.08) }
	box := func(m, x, y, z float64) geom.V3 { // solid box, extents along x, y, z
		return geom.V3{m * (y*y + z*z) / 12, m * (x*x + z*z) / 12, m * (x*x + y*y) / 12}
	}
	pelM := mf(0.142)
	pelvisP := part{pelM, geom.V3{0, -0.03 * k, 0}, box(pelM, 0.2*k, 0.2*k, 0.34*k)}
	abdM := mf(0.134)
	abdP := part{abdM, geom.V3{0, 0.5 * abd, 0}, box(abdM, 0.2*k, abd, 0.30*k)}
	thM, headM := mf(0.211), mf(0.081)
	chestCom := geom.V3{u(-0.03, 0.04) * k, 0, u(-0.02, 0.02) * k} // payload off centre on random bodies
	thoraxP := part{thM, T.Add(geom.V3{0.01 * k, 0.12 * k, 0}).Add(chestCom), box(thM, 0.22*k, 0.30*k, 0.36*k)}
	hr := 0.10 * k
	headP := part{headM, T.Add(geom.V3{0.02 * k, cs + 0.23*k, 0}), geom.V3{0.4 * headM * hr * hr, 0.4 * headM * hr * hr, 0.4 * headM * hr * hr}}
	bracket := 0.005 * M // a light link that carries one joint of a pair
	if s.Spine < 1 {
		abdP.m += bracket
	}
	if s.Spine < 2 {
		thoraxP.m += bracket
	}

	b := Body{
		Name:   "human",
		Supply: hydraulics.Supply{Pressure: P, BulkModulus: 2e8},
	}
	if rng != nil {
		b.Supply.BulkModulus = u(1e8, 4e8)
	}
	origin := map[string]geom.V3{} // link frame origins in the neutral pose, pelvis frame
	link := func(name, parent string, at, axis geom.V3, lo, hi, mass float64, com, inertia geom.V3) {
		b.Links = append(b.Links, Link{Name: name, Parent: parent, Mass: mass, Com: com, Inertia: inertia,
			Joint: &Joint{Origin: at.Sub(origin[parent]), Axis: axis, Limits: [2]float64{lo, hi}, Damping: 1.0 * M / 100}})
		origin[name] = at
	}
	small := func(m float64) geom.V3 { return geom.V3{m * 0.004 * k * k, m * 0.004 * k * k, m * 0.004 * k * k} }
	contacts := func(pts ...geom.V3) { // pelvis-frame neutral points onto the last link
		l := &b.Links[len(b.Links)-1]
		for _, p := range pts {
			l.Contacts = append(l.Contacts, p.Sub(origin[l.Name]))
		}
	}

	// Trunk: whatever the spine leaves rigid is lumped together.
	trunkParts := map[int][]part{0: {pelvisP}}
	switch s.Spine {
	case 0:
		trunkParts[0] = append(trunkParts[0], abdP, thoraxP, headP)
	case 1:
		trunkParts[1] = []part{abdP, thoraxP, headP}
	default:
		trunkParts[1] = []part{abdP}
		trunkParts[2] = []part{thoraxP, headP}
	}
	var armParts []part // folded into the trunk when there are no arms
	ua, fa := vary(0.186*H, 0.05), vary(0.146*H, 0.05)
	handL := 0.108 * H
	reach := fa + handL // elbow to fingertip
	uaM, faM := mf(0.025), mf(0.022)
	if !s.Arms {
		for _, z := range []float64{-1, 1} {
			sh := geom.V3{0, abd + cs, z * sw}
			armParts = append(armParts,
				part{uaM + 0.003*M, sh.Add(geom.V3{0, -0.44 * ua, 0}), geom.V3{0.03, 0.003, 0.03}},
				part{faM, sh.Add(geom.V3{0, -ua - 0.4*reach, 0}), geom.V3{0.035, 0.002, 0.035}})
		}
		last := s.Spine
		trunkParts[last] = append(trunkParts[last], armParts...)
	}
	pm, pc, pi := merge(geom.V3{}, trunkParts[0]...)
	b.Links = append(b.Links, Link{Name: "pelvis", Mass: pm, Com: pc, Inertia: pi})
	origin["pelvis"] = geom.V3{}
	// Pelvis box corners: front, back (buttocks), bottom and top.
	for _, x := range []float64{0.10 * k, -0.12 * k} {
		for _, y := range []float64{-0.15 * k, 0.03 * k} {
			for _, z := range []float64{-0.16 * k, 0.16 * k} {
				contacts(geom.V3{x, y, z})
			}
		}
	}
	abdContacts := []geom.V3{{0.11 * k, 0.5 * abd, 0}, {-0.12 * k, 0.5 * abd, 0}, {0, 0.5 * abd, -0.15 * k}, {0, 0.5 * abd, 0.15 * k}}
	chestContacts := []geom.V3{}
	for _, z := range []float64{-1, 1} {
		chestContacts = append(chestContacts,
			T.Add(geom.V3{-0.12 * k, 0.03 * k, z * 0.11 * k}), T.Add(geom.V3{-0.12 * k, 0.22 * k, z * 0.11 * k}), // back
			T.Add(geom.V3{0.12 * k, 0.08 * k, z * 0.09 * k}),  // front
			T.Add(geom.V3{0, cs + 0.04*k, z * (sw + 0.02*k)}), // shoulder tops
			T.Add(geom.V3{0, cs + 0.23*k, z * 0.08 * k}))      // sides of the head
	}
	chestContacts = append(chestContacts, T.Add(geom.V3{0, headTop, 0}), T.Add(geom.V3{-0.10 * k, cs + 0.23*k, 0}), T.Add(geom.V3{0.10 * k, cs + 0.21*k, 0}))
	trunk := "pelvis"
	switch s.Spine {
	case 0:
		contacts(abdContacts...)
		contacts(chestContacts...)
	case 1:
		link("lumbar", "pelvis", geom.V3{}, geom.V3{0, 0, 1}, -0.87, 0.44, bracket, geom.V3{}, small(bracket))
		m, c, in := merge(geom.V3{}, trunkParts[1]...)
		link("abdomen", "lumbar", geom.V3{}, geom.V3{1, 0, 0}, -0.44, 0.44, m, c, in)
		contacts(abdContacts...)
		contacts(chestContacts...)
		trunk = "abdomen"
	default:
		link("lumbar", "pelvis", geom.V3{}, geom.V3{0, 0, 1}, -0.87, 0.44, bracket, geom.V3{}, small(bracket))
		m, c, in := merge(geom.V3{}, trunkParts[1]...)
		link("abdomen", "lumbar", geom.V3{}, geom.V3{1, 0, 0}, -0.44, 0.44, m, c, in)
		contacts(abdContacts...)
		link("thoracic", "abdomen", T, geom.V3{0, 1, 0}, -0.61, 0.61, bracket, geom.V3{}, small(bracket))
		m, c, in = merge(T, trunkParts[2]...)
		link("chest", "thoracic", T, geom.V3{0, 0, 1}, -0.35, 0.26, m, c, in)
		contacts(chestContacts...)
		trunk = "chest"
	}

	type cylReq struct {
		name      string
		base, rod Anchor
		joint     string  // the joint it is sized for (its link name)
		torque    float64 // N·m wanted from the weaker (rod) side at neutral
		speed     float64 // rad/s of that joint at full valve opening
		cables    []Cable // a tendon drive's runs (none for a plain cylinder)
		pressure  bool    // force-controlled valve
	}
	var reqs []cylReq
	var tendons []Tendon
	tq := M / 102.06 * k // torque scale against the reference man
	// at returns an anchor on link for a point given in the neutral pelvis frame.
	at := func(l string, p geom.V3) Anchor { return Anchor{l, jit(p).Sub(origin[l])} }
	cyl := func(name string, base, rod Anchor, joint string, torque, speed float64) {
		reqs = append(reqs, cylReq{name: name, base: base, rod: rod, joint: joint, torque: torque * tq, speed: speed})
	}

	// Legs: the anchor layout found for the small biped (human-aligned
	// ranges with the most leverage), scaled to the segment lengths.
	th, sh := vary(0.245*H, 0.05), vary(0.246*H, 0.05)
	ah, fl := 0.039*H, vary(0.152*H, 0.05)
	for _, side := range []struct {
		s string
		z float64
	}{{"l", -1}, {"r", 1}} {
		n, z := side.s, side.z
		sl, ss := th/0.25, sh/0.25
		rollLo, rollHi := -0.52, 0.79 // 30° adduction, 45° abduction
		yawLo, yawHi := -0.70, 0.79   // 40° internal, 45° external rotation
		if z > 0 {
			rollLo, rollHi = -0.79, 0.52
			yawLo, yawHi = -0.79, 0.70
		}
		hip := geom.V3{0, -hipDrop, z * hw}
		mount := "pelvis"
		thighM := mf(0.100) - 0.008*M
		if s.HipYaw {
			link("hip_yaw_"+n, "pelvis", hip, geom.V3{0, 1, 0}, yawLo, yawHi, 0.005*M, geom.V3{}, small(0.005*M))
			mount = "hip_yaw_" + n
		} else {
			thighM += 0.005 * M
		}
		link("hip_roll_"+n, mount, hip, geom.V3{1, 0, 0}, rollLo, rollHi, 0.003*M, geom.V3{}, small(0.003*M))
		link("thigh_"+n, "hip_roll_"+n, hip, geom.V3{0, 0, 1}, -0.35, 1.92, thighM,
			geom.V3{0, -0.433 * th, 0}, geom.V3{thighM * math.Pow(0.323*th, 2), thighM * math.Pow(0.06*k, 2), thighM * math.Pow(0.323*th, 2)})
		knee := hip.Add(geom.V3{0, -th, 0})
		contacts(knee.Add(geom.V3{0.07 * k, th / 2, 0}), knee.Add(geom.V3{-0.07 * k, th / 2, 0}), knee.Add(geom.V3{0, th / 2, z * 0.08 * k}))
		shinM := mf(0.0465)
		link("shin_"+n, "thigh_"+n, knee, geom.V3{0, 0, 1}, -1.57, 0.05, shinM,
			geom.V3{0, -0.433 * sh, 0}, geom.V3{shinM * math.Pow(0.302*sh, 2), shinM * math.Pow(0.04*k, 2), shinM * math.Pow(0.302*sh, 2)})
		contacts(knee.Add(geom.V3{0.05 * k, 0, 0}), knee.Add(geom.V3{0.05 * k, -sh / 2, 0}), knee.Add(geom.V3{-0.06 * k, -sh / 2, 0}), knee.Add(geom.V3{0, -sh / 2, z * 0.06 * k}))
		// Foot: talus (ankle pitch) → heel (subtalar: tilting the sole in
		// and out) → midfoot (a hinge across the arch, held by a ligament
		// spring and the plantar fascia) → toes (a sprung toe joint at the
		// ball of the foot). Every sole contact carries a pressure pad.
		ankle := knee.Add(geom.V3{0, -sh, 0})
		footM := mf(0.0145)
		heel, toe, half := -0.07*k, fl-0.055*k, vary(0.05*k, 0.1)
		mtpX := heel + 0.78*(toe-heel) // ball of the foot
		midX := heel + 0.45*(toe-heel) // the arch hinge
		sole := -ah
		in := -z // towards the body's midline
		padAll := func() {
			l := &b.Links[len(b.Links)-1]
			for i := range l.Contacts {
				l.Sensors = append(l.Sensors, i)
			}
		}
		// spring sets a foot joint's passive stiffness (N·m/rad) and its
		// soft-tissue damping (N·m·s/rad).
		spring := func(kNm, c float64) {
			j := b.Links[len(b.Links)-1].Joint
			j.Stiffness, j.Damping = kNm, c
		}
		link("foot_"+n, "shin_"+n, ankle, geom.V3{0, 0, 1}, -0.79, 0.61, 0.1*footM, geom.V3{}, small(0.1*footM))
		invLo, invHi := -0.26, 0.52 // 15° eversion, 30° inversion (left: + turns the sole inwards)
		if z > 0 {
			invLo, invHi = -0.52, 0.26
		}
		hm, mm, tm := 0.4*footM, 0.35*footM, 0.15*footM
		link("heel_"+n, "foot_"+n, ankle, geom.V3{1, 0, 0}, invLo, invHi, hm,
			geom.V3{0.5 * (heel + midX), 0.6 * sole, 0}, box(hm, midX-heel, ah, 2*half))
		spring(0.5*M, 0.05*M)
		contacts(ankle.Add(geom.V3{heel, sole, -0.6 * half}), ankle.Add(geom.V3{heel, sole, 0.6 * half}), ankle.Add(geom.V3{midX - 0.02*k, sole, -in * half}))
		padAll()
		midO := ankle.Add(geom.V3{midX, sole + 0.04*k, 0}) // the arch peaks about 4 cm up
		link("midfoot_"+n, "heel_"+n, midO, geom.V3{0, 0, 1}, -0.26, 0.17, mm,
			geom.V3{0.5 * (mtpX - midX), -0.02 * k, 0}, box(mm, mtpX-midX, 0.035*k, 2*half))
		spring(2.5*M, 0.4*M)
		contacts(ankle.Add(geom.V3{mtpX - 0.01*k, sole, -half}), ankle.Add(geom.V3{mtpX - 0.01*k, sole, half}), ankle.Add(geom.V3{0.5 * (midX + mtpX), sole, -in * 0.9 * half}))
		padAll()
		mtpO := ankle.Add(geom.V3{mtpX, sole + 0.012*k, 0})
		link("toes_"+n, "midfoot_"+n, mtpO, geom.V3{0, 0, 1}, -0.35, 1.05, tm,
			geom.V3{0.5 * (toe - mtpX), -0.006 * k, 0}, box(tm, toe-mtpX, 0.02*k, 2*half))
		spring(0.1*M, 0.02*M)
		contacts(ankle.Add(geom.V3{toe, sole, -0.6 * half}), ankle.Add(geom.V3{toe, sole, 0.6 * half}), ankle.Add(geom.V3{mtpX + 0.025*k, sole, 0}))
		padAll()
		// Plantar fascia: heel, under the ball of the foot, to the toes.
		// Bending the toes up winds it round the ball and lifts the arch
		// (the windlass), so the foot stiffens for push-off by itself.
		fascia := []Anchor{at("heel_"+n, ankle.Add(geom.V3{heel + 0.02*k, sole + 0.015*k, 0})),
			at("midfoot_"+n, ankle.Add(geom.V3{mtpX, sole + 0.004*k, 0})),
			at("toes_"+n, ankle.Add(geom.V3{mtpX + 0.025*k, sole + 0.004*k, 0}))}
		tendons = append(tendons, Tendon{Name: "plantar_" + n, Path: fascia, Stiffness: 2e5 * M / 102.06, Damping: 300 * M / 102.06})

		if s.HipYaw {
			// Horizontal, across the pelvis: from the front of the far side
			// (inside the pelvis, which spans x −0.12k..0.10k) back to a
			// lever behind the yaw axis on the hip bracket. Anchor search:
			// covers 40° in to 45° out with a 74 mm stroke and a 38 mm
			// minimum lever. The pelvis is too short front to back for a
			// cylinder that stays on its own side, so left and right cross
			// in the middle. In hardware stack them about 3 cm apart
			// vertically (one above hip height, one below); moving both pins
			// of a horizontal cylinder up or down by the same amount does not
			// change its yaw lever. The sim keeps both at hip height so the
			// body stays an exact mirror image (training relies on that), and
			// cylinders do not collide in the sim.
			// (It used to be mounted 20 cm behind the hip: 8 cm behind the
			// pelvis, attached to nothing.)
			cyl("hip_yaw_"+n, at("pelvis", hip.Add(geom.V3{0.085 * k, 0, -z * 0.21 * k})), at(mount, hip.Add(geom.V3{-0.045 * k, 0, -z * 0.025 * k})), "hip_yaw_"+n, 80, 5)
		}
		cyl("hip_pitch_"+n, at(mount, hip.Add(geom.V3{0.035 * sl, 0.05 * sl, 0})), at("thigh_"+n, hip.Add(geom.V3{0.07 * sl, -0.96 * th, 0})), "thigh_"+n, 300, 8)
		if s.HipRoll {
			cyl("hip_roll_"+n, at(mount, hip.Add(geom.V3{0, -0.005 * sl, -z * 0.045 * sl})), at("thigh_"+n, hip.Add(geom.V3{0, -0.64 * th, -z * 0.04 * sl})), "hip_roll_"+n, 200, 6)
		}
		// Knee: a "quad" down the front of the thigh to the front of the
		// shin. Ankle: a "calf" down the back of the shin to the heel.
		cyl("knee_"+n, at("thigh_"+n, hip.Add(geom.V3{0.07 * sl, -0.38 * th, 0})), at("shin_"+n, knee.Add(geom.V3{0.065 * ss, -0.005 * ss, 0})), "shin_"+n, 400, 8)
		if s.DoubleKnee {
			cyl("knee_back_"+n, at("thigh_"+n, hip.Add(geom.V3{-0.03 * sl, -0.12 * th, 0})), at("shin_"+n, knee.Add(geom.V3{0.045 * ss, -0.035 * ss, 0})), "shin_"+n, 150, 8)
		}
		cyl("ankle_"+n, at("shin_"+n, knee.Add(geom.V3{-0.06 * ss, -0.22 * sh, 0})), at("heel_"+n, ankle.Add(geom.V3{-0.05 * sl, -0.01 * sl, 0})), "foot_"+n, 250, 8)
		// Stirrup: a force-controlled cylinder on the back of the shin pulls
		// a cable loop. One run passes behind the inner ankle bone to the
		// inside of the heel, the other behind the outer one to the
		// outside, so one valve tilts the sole either way and sets how
		// firmly it is held (tibialis posterior and peroneus longus).
		body0, body1 := at("shin_"+n, knee.Add(geom.V3{-0.05 * k, -0.25 * sh, 0})), at("shin_"+n, knee.Add(geom.V3{-0.05 * k, -0.55 * sh, 0}))
		run := func(side float64) []Anchor {
			return []Anchor{body1,
				at("shin_"+n, ankle.Add(geom.V3{-0.02 * k, 0.03 * k, side * 0.035 * k})),
				at("heel_"+n, ankle.Add(geom.V3{0.02 * k, -0.045 * k, side * 0.04 * k}))}
		}
		reqs = append(reqs, cylReq{name: "stirrup_" + n, base: body0, rod: body1, joint: "heel_" + n, torque: 80 * tq, speed: 6, pressure: true,
			cables: []Cable{{Sign: 0.5, Path: run(in)}, {Sign: -0.5, Path: run(-in)}}})
	}

	// Spine: two erector cylinders behind the lumbar joint (together they
	// pitch the trunk, against each other they bend it sideways), and at
	// the thoracic joint one pitch cylinder behind and one horizontal
	// twist cylinder.
	if s.Spine >= 1 {
		for _, z := range []float64{-1, 1} {
			n := map[float64]string{-1: "l", 1: "r"}[z]
			cyl("erector_"+n, at("pelvis", geom.V3{-0.10 * k, -0.15 * k, z * 0.07 * k}), at("abdomen", geom.V3{-0.10 * k, 0.85 * abd, z * 0.07 * k}), "lumbar", 350, 4)
		}
	}
	if s.Spine >= 2 {
		cyl("thoracic_pitch", at("abdomen", geom.V3{-0.11 * k, 0.25 * abd, 0}), at("chest", T.Add(geom.V3{-0.11 * k, 0.15 * k, 0})), "chest", 150, 4)
		cyl("thoracic_yaw", at("abdomen", T.Add(geom.V3{-0.12 * k, 0, -0.10 * k})), at("thoracic", T.Add(geom.V3{-0.03 * k, 0, 0.04 * k})), "thoracic", 60, 4)
	}

	// Arms: shoulder flexion 40° back to 90° forward, abduction 10° in to
	// 80° out, elbow straight to 115°. The shoulder pitch cylinder is the
	// hip's layout turned back 20°, the abduction cylinder the hip roll's,
	// and the elbow a "triceps" behind the upper arm to the olecranon.
	if s.Arms {
		ra := ua / 0.331 * 0.76 // arm anchors against the thigh layout
		for _, side := range []struct {
			s string
			z float64
		}{{"l", -1}, {"r", 1}} {
			n, z := side.s, side.z
			sh := geom.V3{0, abd + cs, z * sw}
			rollLo, rollHi := -0.17, 1.40
			if z > 0 {
				rollLo, rollHi = -1.40, 0.17
			}
			link("shoulder_"+n, trunk, sh, geom.V3{0, 0, 1}, -0.70, 1.57, 0.003*M, geom.V3{}, small(0.003*M))
			link("upper_arm_"+n, "shoulder_"+n, sh, geom.V3{1, 0, 0}, rollLo, rollHi, uaM,
				geom.V3{0, -0.436 * ua, 0}, geom.V3{uaM * math.Pow(0.322*ua, 2), uaM * math.Pow(0.03*k, 2), uaM * math.Pow(0.322*ua, 2)})
			elbow := sh.Add(geom.V3{0, -ua, 0})
			contacts(elbow.Add(geom.V3{0.045 * k, ua / 2, 0}), elbow.Add(geom.V3{-0.045 * k, ua / 2, 0}), elbow.Add(geom.V3{0, ua / 2, z * 0.045 * k}))
			link("forearm_"+n, "upper_arm_"+n, elbow, geom.V3{0, 0, 1}, -0.03, 2.0, faM,
				geom.V3{0, -0.396 * reach, 0}, geom.V3{faM * math.Pow(0.28*reach, 2), faM * math.Pow(0.025*k, 2), faM * math.Pow(0.28*reach, 2)})
			contacts(elbow.Add(geom.V3{-0.04 * k, 0, 0}), // elbow tip
				elbow.Add(geom.V3{0.035 * k, -0.4 * fa, 0}), elbow.Add(geom.V3{-0.035 * k, -0.4 * fa, 0}),
				elbow.Add(geom.V3{0, -reach, 0}),                                                                        // fingertips
				elbow.Add(geom.V3{0.035 * k, -fa - 0.45*handL, 0}), elbow.Add(geom.V3{-0.035 * k, -fa - 0.45*handL, 0}), // palm, back of the hand
				elbow.Add(geom.V3{0, -fa - 0.45*handL, z * 0.04 * k}))

			cyl("shoulder_pitch_"+n, at(trunk, sh.Add(geom.V3{0.066 * ra, 0.046 * ra, 0})), at("upper_arm_"+n, sh.Add(geom.V3{0.093 * ra, -0.96 * ua, 0})), "shoulder_"+n, 110, 8)
			cyl("shoulder_roll_"+n, at(trunk, sh.Add(geom.V3{0, -0.007 * ra, -z * 0.059 * ra})), at("upper_arm_"+n, sh.Add(geom.V3{0, -0.64 * ua, -z * 0.053 * ra})), "upper_arm_"+n, 90, 8)
			cyl("elbow_"+n, at("upper_arm_"+n, sh.Add(geom.V3{-0.045 * ra, -0.15 * ua, 0})), at("forearm_"+n, elbow.Add(geom.V3{-0.03 * ra, 0.03 * ra, 0})), "forearm_"+n, 90, 8)
		}
	}

	for _, r := range reqs {
		c := Cylinder{Name: r.name, Base: r.base, Rod: r.rod, Cables: r.cables}
		b.Cylinders = append(b.Cylinders, b.fitCylinder(c, r.joint, r.torque, r.speed, P, r.pressure, rng))
	}
	for _, t := range tendons {
		t.RestLength = 0.998 * b.NeutralLength([]Cable{{Sign: 1, Path: t.Path}}) // a little pretension
		b.Tendons = append(b.Tendons, t)
	}
	if rng != nil {
		b.Name = fmt.Sprintf("random-human-%d", len(b.Cylinders))
	}
	return b
}

// frames returns every link's origin and orientation in the pelvis frame
// for joint angles q (indexed by link; q[0] is unused).
func (b Body) frames(q []float64) ([]geom.V3, []geom.M3) {
	ps := make([]geom.V3, len(b.Links))
	rs := make([]geom.M3, len(b.Links))
	rs[0] = geom.Identity()
	for i := 1; i < len(b.Links); i++ {
		p := b.LinkIndex(b.Links[i].Parent)
		j := b.Links[i].Joint
		rs[i] = rs[p].Mul(geom.AxisAngle(j.Axis, q[i]))
		ps[i] = ps[p].Add(rs[p].MulV(j.Origin))
	}
	return ps, rs
}

// routesLength is a cylinder's effective length at joint angles q.
func (b Body) routesLength(q []float64, routes []Cable) float64 {
	ps, rs := b.frames(q)
	pa := func(a Anchor) geom.V3 {
		i := b.LinkIndex(a.Link)
		return ps[i].Add(rs[i].MulV(a.Point))
	}
	l := 0.0
	for _, r := range routes {
		for i := 1; i < len(r.Path); i++ {
			l += r.Sign * pa(r.Path[i]).Sub(pa(r.Path[i-1])).Norm()
		}
	}
	return l
}

// Spanned lists the links whose joints lie between two links in the tree.
func (b Body) Spanned(a, c string) []int {
	up := func(name string) []int {
		var out []int
		for i := b.LinkIndex(name); i >= 0; i = b.LinkIndex(b.Links[i].Parent) {
			out = append(out, i)
			if i == 0 {
				break
			}
		}
		return out
	}
	pa, pc := up(a), up(c)
	inC := map[int]bool{}
	for _, i := range pc {
		inC[i] = true
	}
	common := -1
	for _, i := range pa {
		if inC[i] {
			common = i
			break
		}
	}
	var out []int
	for _, chain := range [][]int{pa, pc} {
		for _, i := range chain {
			if i == common {
				break
			}
			out = append(out, i)
		}
	}
	return out
}

// fitCylinder sizes a cylinder between two anchors: its stroke and
// retracted length cover every combination of the spanned joints' ranges,
// its bore gives the wanted torque about joint from the rod side at
// supply pressure P (at the neutral pose's moment arm), and its valve
// moves that joint at speed rad/s when fully open.
func (b Body) fitCylinder(c Cylinder, joint string, torque, speed, P float64, pressure bool, rng *rand.Rand) Cylinder {
	routes := c.Routes()
	var span []int
	seen := map[int]bool{}
	for _, r := range routes {
		for _, a := range r.Path[1:] {
			for _, i := range b.Spanned(r.Path[0].Link, a.Link) {
				if !seen[i] {
					seen[i] = true
					span = append(span, i)
				}
			}
		}
	}
	q := make([]float64, len(b.Links))
	lo, hi := math.Inf(1), math.Inf(-1)
	const samples = 12
	combos := 1
	for range span {
		combos *= samples + 1
	}
	for n := 0; n < combos; n++ {
		for i, v := 0, n; i < len(span); i, v = i+1, v/(samples+1) {
			lim := b.Links[span[i]].Joint.Limits
			q[span[i]] = lim[0] + (lim[1]-lim[0])*float64(v%(samples+1))/samples
		}
		l := b.routesLength(q, routes)
		lo, hi = math.Min(lo, l), math.Max(hi, l)
	}
	for _, i := range span {
		q[i] = 0
	}
	// Moment arm about the sized joint at neutral.
	ji := b.LinkIndex(joint)
	const eps = 1e-4
	q[ji] = eps
	lp := b.routesLength(q, routes)
	q[ji] = -eps
	lm := b.routesLength(q, routes)
	q[ji] = 0
	arm := math.Max(math.Abs(lp-lm)/(2*eps), 0.01)
	jit := func(lo, hi float64) float64 {
		if rng == nil {
			return 1
		}
		return lo + rng.Float64()*(hi-lo)
	}
	const rodRatio = 0.6 // rod-side annulus over bore
	bore := torque / (rodRatio * P * arm) * jit(0.8, 1.25)
	v := speed * arm // m/s at full opening
	gain := bore * v / math.Sqrt(P/2) * jit(0.7, 1.3)
	margin := 0.003
	c.RetractedLength = lo - margin
	c.Params = hydraulics.Params{
		BoreArea: bore, RodArea: rodRatio * bore, Stroke: hi - lo + 2*margin,
		DeadVolume: 3e-5 * jit(0.6, 1.6),
		Viscous:    0.03 * P * bore / v * jit(0.5, 2),
		ValveGain:  gain, ValveTau: 0.01 * jit(0.5, 2),
		PressureControl: pressure,
	}
	return c
}

// Random draws a humanoid for morphology randomisation: height 1.55–1.95
// m, build from lean to heavy, segment proportions, payload CoG, supply,
// cylinder sizes and anchors all vary; some bodies have no arms, a
// shorter spine or a rigid trunk, no hip yaw or roll, or a second knee
// cylinder.
func Random(rng *rand.Rand) Body {
	h := 1.55 + 0.40*rng.Float64()
	bmi := 20 + 13*rng.Float64()
	s := HumanSpec{
		Height: h, Mass: bmi * h * h,
		Spine:      []int{0, 1, 2, 2, 2}[rng.Intn(5)],
		Arms:       rng.Float64() < 0.8,
		HipYaw:     rng.Float64() < 0.8,
		HipRoll:    rng.Float64() < 0.85,
		DoubleKnee: rng.Float64() < 0.2,
		Pressure:   14e6 + 10e6*rng.Float64(),
	}
	return buildHuman(s, rng)
}
