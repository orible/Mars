package optic

import (
	"math"
)

// Display maps linear radiance to sRGB at a fixed exposure, so clean and
// translated renders are compared on the same scale.
func Display(lin []RGB, exposure float32) []RGB {
	out := make([]RGB, len(lin))
	for k, c := range lin {
		out[k] = RGB{srgb(c[0] * exposure), srgb(c[1] * exposure), srgb(c[2] * exposure)}
	}
	return out
}

// Luma is the Rec. 709 luma plane of an image.
func Luma(img []RGB) []float32 {
	out := make([]float32, len(img))
	for k, c := range img {
		out[k] = luma(c)
	}
	return out
}

// interior marks pixels whose backward flow is valid and whose
// 4-neighbours show the same object: warping there samples one surface.
func interior(f *Frame, fl []Flow) []bool {
	out := make([]bool, len(fl))
	for y := 1; y < f.H-1; y++ {
		for x := 1; x < f.W-1; x++ {
			k := y*f.W + x
			id := f.ID[k]
			out[k] = fl[k].Valid && f.ID[k-1] == id && f.ID[k+1] == id && f.ID[k-f.W] == id && f.ID[k+f.W] == id
		}
	}
	return out
}

// Flicker is the warped-difference metric: the mean absolute luma
// difference (in 0..255 sRGB levels) between frame t and frame t−1
// warped to it along the exact backward flow, over pixels whose surface
// was visible in both frames and away from silhouettes. A clean render
// scores near 0 (only resampling error); a translator adds its own
// frame-to-frame change on top.
func Flicker(cur, prev []RGB, back []Flow, f *Frame) float64 {
	m := interior(f, back)
	var s float64
	n := 0
	for y := 0; y < f.H; y++ {
		for x := 0; x < f.W; x++ {
			k := y*f.W + x
			if !m[k] {
				continue
			}
			p := bilinearRGB(prev, f.W, f.H, float32(x)+0.5+back[k].DX, float32(y)+0.5+back[k].DY)
			s += math.Abs(float64(luma(cur[k]) - luma(p)))
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return 255 * s / float64(n)
}

// BlockMatch estimates left-eye disparity by zero-mean SAD over 7×7
// windows along the rectified row, with parabolic sub-pixel refinement.
// It is a deliberately plain matcher: the point is to see whether
// translation makes the pair harder or easier to match. Pixels it cannot
// match (border) are −1.
func BlockMatch(l, r []float32, w, h, maxD int) []float32 {
	const rad = 3
	out := make([]float32, w*h)
	cost := make([]float64, maxD+1)
	mean := func(p []float32, x, y int) float64 {
		var s float64
		for dy := -rad; dy <= rad; dy++ {
			for dx := -rad; dx <= rad; dx++ {
				s += float64(p[(y+dy)*w+x+dx])
			}
		}
		return s / float64((2*rad+1)*(2*rad+1))
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			k := y*w + x
			out[k] = -1
			if y < rad || y >= h-rad || x < rad || x >= w-rad {
				continue
			}
			ml := mean(l, x, y)
			best, bd := math.Inf(1), -1
			for d := 0; d <= maxD; d++ {
				xr := x - d
				if xr < rad {
					cost[d] = math.Inf(1)
					continue
				}
				mr := mean(r, xr, y)
				var c float64
				for dy := -rad; dy <= rad; dy++ {
					for dx := -rad; dx <= rad; dx++ {
						c += math.Abs(float64(l[(y+dy)*w+x+dx]) - ml - float64(r[(y+dy)*w+xr+dx]) + mr)
					}
				}
				cost[d] = c
				if c < best {
					best, bd = c, d
				}
			}
			if bd < 0 {
				continue
			}
			sub := float64(bd)
			if bd > 0 && bd < maxD && !math.IsInf(cost[bd-1], 1) && !math.IsInf(cost[bd+1], 1) {
				a, b, c := cost[bd-1], cost[bd], cost[bd+1]
				if den := a - 2*b + c; den > 0 {
					sub += 0.5 * (a - c) / den
				}
			}
			out[k] = float32(sub)
		}
	}
	return out
}

// StereoMask selects the left-eye pixels where disparity error is
// meaningful: seen by both eyes (exact correspondence valid), not sky,
// inside the matcher's reach, and textured enough in the clean render to
// be matchable at all (7×7 luma std above minStd).
func StereoMask(left *Frame, lr []Flow, cleanLuma []float32, maxD int, minStd float64) []bool {
	w, h := left.W, left.H
	out := make([]bool, w*h)
	for y := 3; y < h-3; y++ {
		for x := 3 + maxD; x < w-3; x++ {
			k := y*w + x
			if left.ID[k] == 0 || !lr[k].Valid {
				continue
			}
			var s, s2 float64
			for dy := -3; dy <= 3; dy++ {
				for dx := -3; dx <= 3; dx++ {
					v := float64(cleanLuma[(y+dy)*w+x+dx])
					s += v
					s2 += v * v
				}
			}
			s /= 49
			if math.Sqrt(math.Max(0, s2/49-s*s)) >= minStd {
				out[k] = true
			}
		}
	}
	return out
}

// DisparityError compares estimated disparity with ground truth on mask:
// mean absolute error (px) and the fraction off by more than 1 px.
func DisparityError(est, gt []float32, mask []bool) (mae, bad float64) {
	n := 0
	for k, ok := range mask {
		if !ok {
			continue
		}
		e := math.Abs(float64(est[k] - gt[k]))
		if est[k] < 0 {
			e = float64(gt[k])
		}
		mae += e
		if e > 1 {
			bad++
		}
		n++
	}
	if n == 0 {
		return 0, 0
	}
	return mae / float64(n), bad / float64(n)
}

// sobel returns gradient x, y and magnitude planes.
func sobel(p []float32, w, h int) (gx, gy, gm []float32) {
	gx, gy, gm = make([]float32, w*h), make([]float32, w*h), make([]float32, w*h)
	at := func(x, y int) float32 { return p[clampi(y, 0, h-1)*w+clampi(x, 0, w-1)] }
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			a := at(x+1, y-1) + 2*at(x+1, y) + at(x+1, y+1) - at(x-1, y-1) - 2*at(x-1, y) - at(x-1, y+1)
			b := at(x-1, y+1) + 2*at(x, y+1) + at(x+1, y+1) - at(x-1, y-1) - 2*at(x, y-1) - at(x+1, y-1)
			k := y*w + x
			gx[k], gy[k] = a, b
			gm[k] = float32(math.Hypot(float64(a), float64(b)))
		}
	}
	return
}

// EdgeDrift measures how far image edges move. At every geometric edge
// pixel (GeometricEdges) where the reference image has a clear gradient,
// it takes the luma profile across the edge (along the reference
// gradient, ±1.5 px) and finds the shift of the other image's profile
// (within ±1.5 px, 1/8 px steps) that correlates best with it (normalised
// cross-correlation, so contrast and brightness changes do not count).
// The mean absolute shift is the drift in pixels. Images may be smoothed
// by sigma first so sensor noise does not register as drift.
func EdgeDrift(ref, img []float32, edges []bool, w, h int, sigma float32) (drift float64, n int) {
	if sigma > 0 {
		ref, img = blurF(ref, w, h, sigma), blurF(img, w, h, sigma)
	}
	gx, gy, gr := sobel(ref, w, h)
	var maxG float32
	for _, g := range gr {
		maxG = max(maxG, g)
	}
	const half = 6 // profile ±1.5 px in 1/4 px steps
	var pr, pi [2*half + 1]float64
	for y := 4; y < h-4; y++ {
		for x := 4; x < w-4; x++ {
			k := y*w + x
			if !edges[k] || gr[k] < 0.15*maxG {
				continue
			}
			dx, dy := gx[k]/gr[k], gy[k]/gr[k]
			cx, cy := float32(x)+0.5, float32(y)+0.5
			for i := range pr {
				t := float32(i-half) * 0.25
				pr[i] = float64(sampleF(ref, w, h, cx+dx*t, cy+dy*t))
			}
			best, bs := math.Inf(-1), 0.0
			for si := -12; si <= 12; si++ {
				s := float32(si) * 0.125
				for i := range pi {
					t := float32(i-half)*0.25 + s
					pi[i] = float64(sampleF(img, w, h, cx+dx*t, cy+dy*t))
				}
				if c := ncc(pr[:], pi[:]); c > best {
					best, bs = c, float64(s)
				}
			}
			drift += math.Abs(bs)
			n++
		}
	}
	if n == 0 {
		return 0, 0
	}
	return drift / float64(n), n
}

func ncc(a, b []float64) float64 {
	var ma, mb float64
	for i := range a {
		ma += a[i]
		mb += b[i]
	}
	ma /= float64(len(a))
	mb /= float64(len(b))
	var ab, aa, bb float64
	for i := range a {
		x, y := a[i]-ma, b[i]-mb
		ab += x * y
		aa += x * x
		bb += y * y
	}
	return ab / math.Sqrt(aa*bb+1e-12)
}

func blurF(p []float32, w, h int, sigma float32) []float32 {
	in := make([]RGB, len(p))
	for k, v := range p {
		in[k] = RGB{v, v, v}
	}
	out := make([]RGB, len(p))
	gaussBlur(in, out, w, h, sigma)
	r := make([]float32, len(p))
	for k := range out {
		r[k] = out[k][0]
	}
	return r
}
