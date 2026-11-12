package optic

import (
	"image"
	"image/color"
	"image/color/palette"
	"image/draw"
	"image/gif"
	"image/png"
	"math"
	"os"
)

// Panel is one tile of a contact sheet: an sRGB image of w×h.
type Panel struct {
	Img  []RGB
	W, H int
}

// Sheet lays panels out in rows (all panels the same size) and scales
// them up by an integer factor with nearest-neighbour pixels.
func Sheet(rows [][]Panel, scale int) *image.RGBA {
	pw, ph := rows[0][0].W, rows[0][0].H
	cols := 0
	for _, r := range rows {
		cols = max(cols, len(r))
	}
	gap := 2
	W := cols*(pw*scale+gap) - gap
	H := len(rows)*(ph*scale+gap) - gap
	im := image.NewRGBA(image.Rect(0, 0, W, H))
	draw.Draw(im, im.Bounds(), image.NewUniform(color.RGBA{20, 20, 20, 255}), image.Point{}, draw.Src)
	for ri, r := range rows {
		for ci, p := range r {
			ox, oy := ci*(pw*scale+gap), ri*(ph*scale+gap)
			for y := 0; y < p.H*scale; y++ {
				for x := 0; x < p.W*scale; x++ {
					c := p.Img[(y/scale)*p.W+x/scale]
					im.SetRGBA(ox+x, oy+y, color.RGBA{b8(c[0]), b8(c[1]), b8(c[2]), 255})
				}
			}
		}
	}
	return im
}

func b8(x float32) uint8 { return uint8(clamp01(x)*255 + 0.5) }

// SavePNG writes an image.
func SavePNG(path string, im image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, im)
}

// SaveGIF writes frames as an animated GIF at delay hundredths of a second
// per frame (Plan 9 palette, Floyd–Steinberg dithering).
func SaveGIF(path string, frames []*image.RGBA, delay int) error {
	g := &gif.GIF{}
	for _, fr := range frames {
		p := image.NewPaletted(fr.Bounds(), palette.Plan9)
		draw.FloydSteinberg.Draw(p, fr.Bounds(), fr, image.Point{})
		g.Image = append(g.Image, p)
		g.Delay = append(g.Delay, delay)
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gif.EncodeAll(f, g)
}

// DepthImage shows depth as inverse-depth grey (near bright, sky black).
func DepthImage(f *Frame) []RGB {
	out := make([]RGB, len(f.Depth))
	for k, d := range f.Depth {
		if math.IsInf(float64(d), 1) {
			continue
		}
		v := clamp01(0.6 / d)
		out[k] = RGB{v, v, v}
	}
	return out
}

// IDImage colours each object id with a fixed hashed colour.
func IDImage(f *Frame) []RGB {
	out := make([]RGB, len(f.ID))
	for k, id := range f.ID {
		if id == 0 {
			continue
		}
		h := uint32(id) * 2654435761
		out[k] = RGB{0.25 + 0.75*float32(h&255)/255, 0.25 + 0.75*float32(h>>8&255)/255, 0.25 + 0.75*float32(h>>16&255)/255}
	}
	return out
}

// FlowImage colours flow by direction (hue) and length (saturation, full
// at maxLen px); invalid pixels are black.
func FlowImage(fl []Flow, maxLen float32) []RGB {
	out := make([]RGB, len(fl))
	for k, f := range fl {
		if !f.Valid {
			continue
		}
		a := math.Atan2(float64(f.DY), float64(f.DX))
		m := clamp01(float32(math.Hypot(float64(f.DX), float64(f.DY))) / maxLen)
		h := (a + math.Pi) / (2 * math.Pi) * 6
		var c RGB
		for i := 0; i < 3; i++ {
			d := math.Abs(math.Mod(h-float64(2*i)+6, 6) - 3) // 0..3
			c[i] = clamp01(float32(d - 1))
		}
		out[k] = RGB{1 - m + m*c[0], 1 - m + m*c[1], 1 - m + m*c[2]}
	}
	return out
}

// DisparityImage shows a disparity map scaled so maxD is white
// (negative = unmatched, red).
func DisparityImage(d []float32, maxD float32) []RGB {
	out := make([]RGB, len(d))
	for k, v := range d {
		if v < 0 {
			out[k] = RGB{0.6, 0, 0}
			continue
		}
		g := clamp01(v / maxD)
		out[k] = RGB{g, g, g}
	}
	return out
}

// PartImage colours the robot's own links (hashed colour per link) and
// leaves everything else black.
func PartImage(f *Frame) []RGB {
	out := make([]RGB, len(f.Part))
	for k, l := range f.Part {
		if l < 0 {
			continue
		}
		h := uint32(l+1) * 2246822519
		out[k] = RGB{0.3 + 0.7*float32(h&255)/255, 0.3 + 0.7*float32(h>>8&255)/255, 0.3 + 0.7*float32(h>>16&255)/255}
	}
	return out
}
