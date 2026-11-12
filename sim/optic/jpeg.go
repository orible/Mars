package optic

import "math"

// Standard JPEG luminance and chrominance quantisation tables (ITU T.81
// Annex K), scaled by quality the way libjpeg does.
var qLuma = [64]float32{
	16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
	14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
	18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
	49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99}
var qChroma = [64]float32{
	17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
	24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99}

var dctC [8][8]float32

func init() {
	for u := 0; u < 8; u++ {
		a := math.Sqrt(2.0 / 8)
		if u == 0 {
			a = math.Sqrt(1.0 / 8)
		}
		for x := 0; x < 8; x++ {
			dctC[u][x] = float32(a * math.Cos(float64(2*x+1)*float64(u)*math.Pi/16))
		}
	}
}

func scaledTable(t *[64]float32, quality int) [64]float32 {
	q := max(1, min(100, quality))
	s := 5000 / q
	if q >= 50 {
		s = 200 - 2*q
	}
	var o [64]float32
	for i, v := range t {
		o[i] = float32(math.Max(1, math.Floor((float64(v)*float64(s)+50)/100)))
	}
	return o
}

// JPEG round-trips an sRGB image (0..1) through baseline JPEG's lossy
// steps in place: YCbCr, 4:2:0 chroma subsampling, 8×8 DCT and
// quantisation at the given quality (entropy coding is lossless and
// skipped). It produces real blocking and ringing artefacts.
func JPEG(img []RGB, w, h, quality int) {
	ql, qc := scaledTable(&qLuma, quality), scaledTable(&qChroma, quality)
	Y := make([]float32, w*h)
	cw, ch := (w+1)/2, (h+1)/2
	cb, cr := make([]float32, cw*ch), make([]float32, cw*ch)
	cn := make([]float32, cw*ch)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := img[y*w+x]
			r, g, b := 255*c[0], 255*c[1], 255*c[2]
			Y[y*w+x] = 0.299*r + 0.587*g + 0.114*b - 128
			j := (y/2)*cw + x/2
			cb[j] += -0.168736*r - 0.331264*g + 0.5*b
			cr[j] += 0.5*r - 0.418688*g - 0.081312*b
			cn[j]++
		}
	}
	for j := range cb {
		cb[j] /= cn[j]
		cr[j] /= cn[j]
	}
	blocks(Y, w, h, &ql)
	blocks(cb, cw, ch, &qc)
	blocks(cr, cw, ch, &qc)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			// Bilinear chroma upsampling (centred siting).
			fx, fy := (float32(x)+0.5)/2, (float32(y)+0.5)/2
			b := sampleF(cb, cw, ch, fx, fy)
			r := sampleF(cr, cw, ch, fx, fy)
			yy := Y[y*w+x] + 128
			img[y*w+x] = RGB{clamp01((yy + 1.402*r) / 255), clamp01((yy - 0.344136*b - 0.714136*r) / 255), clamp01((yy + 1.772*b) / 255)}
		}
	}
}

func sampleF(p []float32, w, h int, x, y float32) float32 {
	x -= 0.5
	y -= 0.5
	fx, fy := floorf(x), floorf(y)
	ax, ay := x-fx, y-fy
	x0, y0 := clampi(int(fx), 0, w-1), clampi(int(fy), 0, h-1)
	x1, y1 := clampi(int(fx)+1, 0, w-1), clampi(int(fy)+1, 0, h-1)
	return (p[y0*w+x0]*(1-ax)+p[y0*w+x1]*ax)*(1-ay) + (p[y1*w+x0]*(1-ax)+p[y1*w+x1]*ax)*ay
}

// blocks quantises plane p in 8×8 blocks (edge blocks replicate the
// border, as encoders pad).
func blocks(p []float32, w, h int, q *[64]float32) {
	var b, t [64]float32
	for by := 0; by < h; by += 8 {
		for bx := 0; bx < w; bx += 8 {
			for y := 0; y < 8; y++ {
				for x := 0; x < 8; x++ {
					b[y*8+x] = p[clampi(by+y, 0, h-1)*w+clampi(bx+x, 0, w-1)]
				}
			}
			dct2(&b, &t, false)
			for i := range b {
				b[i] = float32(math.Round(float64(b[i]/q[i]))) * q[i]
			}
			dct2(&b, &t, true)
			for y := 0; y < 8 && by+y < h; y++ {
				for x := 0; x < 8 && bx+x < w; x++ {
					p[(by+y)*w+bx+x] = b[y*8+x]
				}
			}
		}
	}
}

// dct2 is the orthonormal 2D DCT-II of an 8×8 block in place (inverse:
// DCT-III).
func dct2(b, t *[64]float32, inverse bool) {
	for y := 0; y < 8; y++ { // rows
		for u := 0; u < 8; u++ {
			var s float32
			for x := 0; x < 8; x++ {
				if inverse {
					s += dctC[x][u] * b[y*8+x]
				} else {
					s += dctC[u][x] * b[y*8+x]
				}
			}
			t[y*8+u] = s
		}
	}
	for x := 0; x < 8; x++ { // columns
		for v := 0; v < 8; v++ {
			var s float32
			for y := 0; y < 8; y++ {
				if inverse {
					s += dctC[y][v] * t[y*8+x]
				} else {
					s += dctC[v][y] * t[y*8+x]
				}
			}
			b[v*8+x] = s
		}
	}
}
