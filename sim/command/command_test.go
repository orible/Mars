package command

import (
	"math/rand"
	"strings"
	"testing"
)

func TestParseClampsLLMReply(t *testing.T) {
	c, err := Parse(`{"heading_deg": 400, "speed": 3}`)
	if err != nil {
		t.Fatal(err)
	}
	if c.HeadingDeg != 40 || c.Speed != MaxSpeed {
		t.Fatalf("got %+v", c)
	}
	if _, err := Parse(`walk left please`); err == nil {
		t.Fatal("free text accepted as a command")
	}
}

func TestPromptSaysLeftAndRight(t *testing.T) {
	p := Perception{Visible: true, DistanceM: 4.8, BearingDeg: -22, Objectives: 1}
	if s := p.Prompt(); !strings.Contains(s, "22° to your left") || !strings.Contains(s, "4.8 m") {
		t.Fatalf("prompt: %s", s)
	}
}

func TestLoopHasLatency(t *testing.T) {
	l := NewLoop(Scripted{Cruise: 0.3, SlowDown: 1.5}, rand.New(rand.NewSource(1)))
	l.AngleNoise, l.DistNoise = 0, 0
	l.Reset(0)
	p := Perception{Visible: true, DistanceM: 5, BearingDeg: 30, Objectives: 1}
	if c := l.Update(0, p); c != Stop {
		t.Fatalf("command in force before the LLM has answered: %+v", c)
	}
	var c Command
	for t := 0.0; t < 1; t += 0.01 {
		c = l.Update(t, p)
	}
	if c.HeadingDeg != 30 || c.Speed != 0.3 {
		t.Fatalf("after the latency: %+v", c)
	}
	near := Perception{Visible: true, DistanceM: 0.75, BearingDeg: 0, Objectives: 1}
	for t := 1.0; t < 4; t += 0.01 {
		c = l.Update(t, near)
	}
	if c.Speed != 0.15 {
		t.Fatalf("should slow as the objective grows near: %+v", c)
	}
}
