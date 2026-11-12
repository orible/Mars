package obs

import (
	"fmt"

	"mars/sim/model"
)

// Feature names, in the order Build writes them (for inspection views).
var (
	TokenFeatures = []string{"stroke pos", "rod speed", "cap press", "rod press", "spool", "strength", "area ratio",
		"base x", "base y", "base z", "rod x", "rod y", "rod z",
		"twist1 x", "twist1 y", "twist1 z", "twist2 x", "twist2 y", "twist2 z", "force ctl", "target err"}
	PartFeatures = []string{"pos x", "pos y", "pos z", "up x", "up y", "up z", "fwd x", "fwd y", "fwd z",
		"vel x", "vel y", "vel z", "mass", "reach", "extent x", "extent y", "extent z", "load", "height"}
	SensorFeatures = []string{"pos x", "pos y", "pos z", "local x", "local z", "pressure", "shear/press",
		"shear x", "shear z", "slip", "tendon"}
	GlobalFeatures = []string{"log mass", "cog x", "cog y", "cog z", "pitch", "roll", "gyro x", "gyro y", "gyro z",
		"cmd cos", "cmd sin", "cmd speed", "cylinders", "parts", "height-1", "floor load", "cog vel x", "cog vel y", "cog vel z"}
)

// Labels names the tokens Build makes for w's body, in the same order:
// cylinders, body parts and sensors.
func Labels(w *model.World) (tokens, parts, sensors []string) {
	m := w.M
	for _, c := range m.Cyl {
		tokens = append(tokens, c.Name)
	}
	_, mass := w.CoG()
	touches := make([]bool, len(m.Names))
	for _, ct := range m.Contacts {
		touches[ct.Link] = true
	}
	for i, name := range m.Names {
		if m.Mass[i] >= PartMinMass*mass || touches[i] {
			parts = append(parts, name)
		}
	}
	for k, sn := range m.Sensors {
		sensors = append(sensors, fmt.Sprintf("%s pad %d", m.Names[m.Contacts[sn.Contact].Link], k))
	}
	for _, tn := range m.Tendons {
		sensors = append(sensors, tn.Name)
	}
	return tokens, parts, sensors
}
