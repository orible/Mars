package model

import (
	"fmt"
	"math"
)

// ReachRow says how far one joint can travel before one cylinder that spans
// it hits an end stop, with every other joint held at 0. It is the quick
// check for a pelvis or limb layout: if a cylinder's stroke cannot cover
// the joint range you need, move its anchors before printing parts.
type ReachRow struct {
	Cylinder, Joint  string
	Lo, Hi           float64 // rad, reachable interval containing 0
	LimitLo, LimitHi float64 // rad, the joint's own limits
}

// Full reports whether the cylinder reaches the joint's whole range.
func (r ReachRow) Full() bool { return r.Lo <= r.LimitLo && r.Hi >= r.LimitHi }

// CornerViolations checks cylinders that span several joints at every
// combination of those joints' limits (and 0), and describes each
// combination where the cylinder would be past an end stop. Reach alone
// moves one joint at a time and can miss these.
func (m *Model) CornerViolations() []string {
	q := make([]float64, m.NDof())
	ps := m.NewPose()
	var out []string
	for k, c := range m.Cyl {
		if len(c.Span) < 2 {
			continue
		}
		combos := 1
		for range c.Span {
			combos *= 3
		}
		for n := 0; n < combos; n++ {
			desc := ""
			for i, v := 0, n; i < len(c.Span); i, v = i+1, v/3 {
				l := c.Span[i] + 1
				q[c.Span[i]] = [3]float64{m.Lo[l], 0, m.Hi[l]}[v%3]
				desc += fmt.Sprintf(" %s=%.1f°", m.Names[l], q[c.Span[i]]*180/math.Pi)
			}
			m.FK(q, ps)
			l := m.CylLength(ps, k, nil)
			if l < c.Retracted-1e-9 || l > c.Retracted+c.P.Stroke+1e-9 {
				out = append(out, fmt.Sprintf("%s past its end stop at%s (length %.4f, range %.4f–%.4f m)",
					c.Name, desc, l, c.Retracted, c.Retracted+c.P.Stroke))
			}
		}
		for _, j := range c.Span {
			q[j] = 0
		}
	}
	return out
}

// Reach sweeps each spanned joint in steps of range/samples.
func (m *Model) Reach(samples int) []ReachRow {
	q := make([]float64, m.NDof())
	ps := m.NewPose()
	var rows []ReachRow
	for k, c := range m.Cyl {
		fits := func(j int, v float64) bool {
			q[j] = v
			m.FK(q, ps)
			l := m.CylLength(ps, k, nil)
			q[j] = 0
			return l >= c.Retracted-1e-9 && l <= c.Retracted+c.P.Stroke+1e-9
		}
		for _, j := range c.Span {
			l := j + 1
			step := (m.Hi[l] - m.Lo[l]) / float64(samples)
			lo, hi := 0.0, 0.0
			for v := -step; v >= m.Lo[l]-1e-12 && fits(j, v); v -= step {
				lo = v
			}
			for v := step; v <= m.Hi[l]+1e-12 && fits(j, v); v += step {
				hi = v
			}
			if m.Lo[l]-lo > -step && fits(j, m.Lo[l]) {
				lo = m.Lo[l]
			}
			if hi-m.Hi[l] > -step && fits(j, m.Hi[l]) {
				hi = m.Hi[l]
			}
			rows = append(rows, ReachRow{Cylinder: c.Name, Joint: m.Names[l], Lo: lo, Hi: hi, LimitLo: m.Lo[l], LimitHi: m.Hi[l]})
		}
	}
	return rows
}
