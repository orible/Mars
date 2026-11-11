// Package mocap reads Acclaim ASF/AMC motion capture (the format of the
// CMU Graphics Lab Motion Capture Database) and turns a human recording
// into the joint-angle features the robot's motion references use: trunk
// (pelvis) lean, pelvis height, and per-leg hip flexion, knee flexion and
// ankle dorsiflexion.
//
// Downloaded recordings are data, never code: this package only parses
// numbers out of them.
package mocap

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

// Skeleton is the part of an ASF file needed to read its AMC frames: the
// degrees of freedom of each bone, in file order.
type Skeleton struct {
	RootOrder []string            // e.g. TX TY TZ RX RY RZ
	Dofs      map[string][]string // bone name → its dofs, e.g. rx ry rz
	Length    float64             // :units length (multiply by 1/Length for inches)
}

// ReadASF parses the bone dof lists from an ASF file.
func ReadASF(path string) (*Skeleton, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	sk := &Skeleton{Dofs: map[string][]string{}, Length: 1}
	sc := bufio.NewScanner(f)
	var bone, section string
	for sc.Scan() {
		fs := strings.Fields(sc.Text())
		if len(fs) == 0 || strings.HasPrefix(fs[0], "#") {
			continue
		}
		if strings.HasPrefix(fs[0], ":") {
			section = fs[0]
			continue
		}
		switch {
		case section == ":units" && fs[0] == "length" && len(fs) > 1:
			sk.Length, _ = strconv.ParseFloat(fs[1], 64)
		case section == ":root" && fs[0] == "order":
			sk.RootOrder = fs[1:]
		case section == ":bonedata" && fs[0] == "name" && len(fs) > 1:
			bone = fs[1]
		case section == ":bonedata" && fs[0] == "dof":
			sk.Dofs[bone] = fs[1:]
		}
	}
	if len(sk.RootOrder) == 0 {
		return nil, fmt.Errorf("%s: no root order", path)
	}
	return sk, sc.Err()
}

// Frame is one AMC frame: each bone's values keyed by dof name (tx, ry...).
type Frame map[string]map[string]float64

// Get returns a bone's dof value (0 if absent).
func (f Frame) Get(bone, dof string) float64 { return f[bone][dof] }

// ReadAMC parses every frame of an AMC file using the skeleton's dof lists.
func ReadAMC(path string, sk *Skeleton) ([]Frame, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var frames []Frame
	var cur Frame
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<16), 1<<20)
	for line := 1; sc.Scan(); line++ {
		fs := strings.Fields(sc.Text())
		if len(fs) == 0 || strings.HasPrefix(fs[0], "#") || strings.HasPrefix(fs[0], ":") {
			continue
		}
		if _, err := strconv.Atoi(fs[0]); err == nil && len(fs) == 1 {
			cur = Frame{}
			frames = append(frames, cur)
			continue
		}
		if cur == nil {
			return nil, fmt.Errorf("%s:%d: bone data before the first frame number", path, line)
		}
		dofs := sk.Dofs[fs[0]]
		if fs[0] == "root" {
			dofs = sk.RootOrder
		}
		vals := map[string]float64{}
		for i, d := range dofs {
			if i+1 >= len(fs) {
				break
			}
			v, err := strconv.ParseFloat(fs[i+1], 64)
			if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, fmt.Errorf("%s:%d: bad number %q", path, line, fs[i+1])
			}
			vals[strings.ToLower(d)] = v
		}
		cur[fs[0]] = vals
	}
	return frames, sc.Err()
}
