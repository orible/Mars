// Package optic is the robot's simulated vision: a tiny stereo software
// rasteriser of low-poly textured geometry, exact ground truth for every
// frame (depth, segmentation, optical flow, stereo disparity, per-object
// appearance vectors), and a realism translator that turns the clean
// render into a real-camera-like frame without breaking temporal or
// stereo consistency.
//
// Frames: world X forward, Y up, Z right (as geom). A camera looks along
// its +X with +Y up and +Z right; pixel (u, v) has u to the right, v down,
// and pixel centres at integer + 0.5.
package optic

import (
	"math"
	"math/rand"
)

// RGB is a linear-light colour.
type RGB [3]float32

func (a RGB) scale(s float32) RGB { return RGB{a[0] * s, a[1] * s, a[2] * s} }
func (a RGB) add(b RGB) RGB       { return RGB{a[0] + b[0], a[1] + b[1], a[2] + b[2]} }
func (a RGB) mul(b RGB) RGB       { return RGB{a[0] * b[0], a[1] * b[1], a[2] * b[2]} }

// Texture is a square, power-of-two, tiling RGB texture with a mip chain.
type Texture struct {
	Size int     // level-0 width and height
	Mips [][]RGB // Mips[l] is (Size>>l)² texels, row-major
}

// NewTexture builds the mip chain of a Size×Size level 0 by box filtering.
func NewTexture(size int, px []RGB) *Texture {
	t := &Texture{Size: size, Mips: [][]RGB{px}}
	for s := size; s > 1; s /= 2 {
		prev, h := t.Mips[len(t.Mips)-1], s/2
		next := make([]RGB, h*h)
		for y := 0; y < h; y++ {
			for x := 0; x < h; x++ {
				a, b := prev[2*y*s+2*x], prev[2*y*s+2*x+1]
				c, d := prev[(2*y+1)*s+2*x], prev[(2*y+1)*s+2*x+1]
				next[y*h+x] = a.add(b).add(c).add(d).scale(0.25)
			}
		}
		t.Mips = append(t.Mips, next)
	}
	return t
}

// Sample is a bilinear, wrapping lookup at (u, v) in texture repeats on
// mip level lod (fractional levels are blended).
func (t *Texture) Sample(u, v, lod float32) RGB {
	if lod <= 0 {
		return t.bilinear(0, u, v)
	}
	top := float32(len(t.Mips) - 1)
	if lod >= top {
		return t.Mips[len(t.Mips)-1][0]
	}
	l := int(lod)
	f := lod - float32(l)
	return t.bilinear(l, u, v).scale(1 - f).add(t.bilinear(l+1, u, v).scale(f))
}

func (t *Texture) bilinear(l int, u, v float32) RGB {
	s := t.Size >> l
	px := t.Mips[l]
	x := u*float32(s) - 0.5
	y := v*float32(s) - 0.5
	fx, fy := float32(math.Floor(float64(x))), float32(math.Floor(float64(y)))
	ax, ay := x-fx, y-fy
	m := s - 1 // s is a power of two
	x0, y0 := int(fx)&m, int(fy)&m
	x1, y1 := (x0+1)&m, (y0+1)&m
	a, b := px[y0*s+x0], px[y0*s+x1]
	c, d := px[y1*s+x0], px[y1*s+x1]
	top := a.scale(1 - ax).add(b.scale(ax))
	bot := c.scale(1 - ax).add(d.scale(ax))
	return top.scale(1 - ay).add(bot.scale(ay))
}

// Pattern is a procedural texture recipe. Textures are baked once from a
// pattern, two colours and a seed; objects of one kind share a pattern
// and differ only by a small tint, so their identity is ambiguous.
type Pattern int

const (
	PatTiles Pattern = iota // floor tiles with grout
	PatWood                 // grain stripes
	PatBrick
	PatStripes
	PatDots
	PatMarble // smooth noise
	PatCardboard
	NumPatterns
)

// Bake renders pattern p into a size×size texture with base colour a and
// detail colour b.
func Bake(p Pattern, size int, a, b RGB, seed int64) *Texture {
	rng := rand.New(rand.NewSource(seed))
	n := newNoise2(rng, 8)
	px := make([]RGB, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			u, v := (float64(x)+0.5)/float64(size), (float64(y)+0.5)/float64(size)
			var t float64 // 0 = a, 1 = b
			switch p {
			case PatTiles:
				gu, gv := math.Mod(u*4, 1), math.Mod(v*4, 1)
				if gu < 0.06 || gv < 0.06 {
					t = 1
				} else {
					t = 0.25 * n.fbm(u*8, v*8, 3)
				}
			case PatWood:
				t = 0.5 + 0.5*math.Sin(2*math.Pi*(v*6+0.6*n.fbm(u*2, v*2, 3)))
				t = t * t
			case PatBrick:
				row := int(v * 8)
				off := 0.0
				if row%2 == 1 {
					off = 0.125
				}
				bu, bv := math.Mod(u*4+off*4, 1), math.Mod(v*8, 1)
				if bu < 0.07 || bv < 0.12 {
					t = 1
				} else {
					t = 0.3 * n.fbm(u*6, v*6, 2)
				}
			case PatStripes:
				if math.Mod(u*4, 1) < 0.5 {
					t = 1
				}
				t = 0.85*t + 0.15*n.fbm(u*8, v*8, 2)
			case PatDots:
				du, dv := math.Mod(u*4, 1)-0.5, math.Mod(v*4, 1)-0.5
				if du*du+dv*dv < 0.09 {
					t = 1
				}
			case PatMarble:
				t = 0.5 + 0.5*math.Sin(6*u+5*n.fbm(u*3, v*3, 4))
			case PatCardboard:
				t = 0.35*n.fbm(u*10, v*3, 3) + 0.15*(0.5+0.5*math.Sin(2*math.Pi*u*16))
			}
			t = math.Max(0, math.Min(1, t))
			px[y*size+x] = a.scale(float32(1 - t)).add(b.scale(float32(t)))
		}
	}
	return NewTexture(size, px)
}

// noise2 is tiling 2D value noise on a g×g lattice.
type noise2 struct {
	g int
	v []float64
}

func newNoise2(rng *rand.Rand, g int) *noise2 {
	n := &noise2{g: g, v: make([]float64, g*g)}
	for i := range n.v {
		n.v[i] = rng.Float64()
	}
	return n
}

func smooth(t float64) float64 { return t * t * (3 - 2*t) }

func (n *noise2) at(x, y float64) float64 {
	fx, fy := math.Floor(x), math.Floor(y)
	ax, ay := smooth(x-fx), smooth(y-fy)
	g := n.g
	x0, y0 := ((int(fx)%g)+g)%g, ((int(fy)%g)+g)%g
	x1, y1 := (x0+1)%g, (y0+1)%g
	a, b := n.v[y0*g+x0], n.v[y0*g+x1]
	c, d := n.v[y1*g+x0], n.v[y1*g+x1]
	return (a*(1-ax)+b*ax)*(1-ay) + (c*(1-ax)+d*ax)*ay
}

// fbm sums oct octaves; the result is roughly in [0, 1].
func (n *noise2) fbm(x, y float64, oct int) float64 {
	s, amp, tot := 0.0, 0.5, 0.0
	for o := 0; o < oct; o++ {
		s += amp * n.at(x, y)
		tot += amp
		x, y, amp = x*2, y*2, amp*0.5
	}
	return s / tot
}

// noise3 is 3D value noise on an integer lattice, hashed so it needs no
// table and never repeats; it is how the translator anchors realism
// detail to surfaces.
type noise3 struct{ seed uint32 }

func hash3(x, y, z int32, seed uint32) float32 {
	h := uint32(x)*0x8da6b343 ^ uint32(y)*0xd8163841 ^ uint32(z)*0xcb1ab31f ^ seed*0x165667b1
	h ^= h >> 13
	h *= 0x5bd1e995
	h ^= h >> 15
	return float32(h&0xffffff) / float32(0xffffff)
}

func (n noise3) at(x, y, z float32) float32 {
	fx, fy, fz := floorf(x), floorf(y), floorf(z)
	ix, iy, iz := int32(fx), int32(fy), int32(fz)
	ax, ay, az := smoothf(x-fx), smoothf(y-fy), smoothf(z-fz)
	s := n.seed
	l := func(dx, dy, dz int32) float32 { return hash3(ix+dx, iy+dy, iz+dz, s) }
	x00 := l(0, 0, 0)*(1-ax) + l(1, 0, 0)*ax
	x10 := l(0, 1, 0)*(1-ax) + l(1, 1, 0)*ax
	x01 := l(0, 0, 1)*(1-ax) + l(1, 0, 1)*ax
	x11 := l(0, 1, 1)*(1-ax) + l(1, 1, 1)*ax
	y0 := x00*(1-ay) + x10*ay
	y1 := x01*(1-ay) + x11*ay
	return y0*(1-az) + y1*az
}

// fbm is centred on 0 with amplitude about ±0.5.
func (n noise3) fbm(x, y, z float32, oct int) float32 {
	s, amp, tot := float32(0), float32(0.5), float32(0)
	for o := 0; o < oct; o++ {
		s += amp * n.at(x, y, z)
		tot += amp
		x, y, z, amp = x*2.03, y*2.03, z*2.03, amp*0.5
	}
	return s/tot - 0.5
}

func floorf(x float32) float32  { return float32(math.Floor(float64(x))) }
func smoothf(t float32) float32 { return t * t * (3 - 2*t) }
