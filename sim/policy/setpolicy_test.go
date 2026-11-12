package policy

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

func randTokens(rng *rand.Rand, n, dim int) [][]float64 {
	t := make([][]float64, n)
	for i := range t {
		t[i] = make([]float64, dim)
		for j := range t[i] {
			t[i][j] = rng.NormFloat64()
		}
	}
	return t
}

func TestArbitraryActuatorCount(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	p := New(rng, 10, 7, 4, 8, 32, 64)
	global := randTokens(rng, 1, 8)[0]
	for _, n := range []int{1, 3, 6, 13} {
		out := p.Act(&Obs{Tokens: randTokens(rng, n, 10), Parts: randTokens(rng, n+2, 7), Sensors: randTokens(rng, n%3, 4), Global: global})
		if len(out) != n {
			t.Fatalf("n=%d: got %d outputs", n, len(out))
		}
		for _, v := range out {
			if v < -1 || v > 1 || math.IsNaN(v) {
				t.Fatalf("n=%d: output %v out of range", n, v)
			}
		}
	}
}

func TestPermutationEquivariant(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	p := New(rng, 10, 7, 4, 8, 32, 64)
	global := randTokens(rng, 1, 8)[0]
	tokens := randTokens(rng, 5, 10)
	parts := randTokens(rng, 4, 7)
	out := p.Act(&Obs{Tokens: tokens, Parts: parts, Global: global})

	perm := []int{3, 0, 4, 1, 2}
	shuffled := make([][]float64, len(tokens))
	for i, j := range perm {
		shuffled[i] = tokens[j]
	}
	// Reordering the body parts must change nothing either.
	outShuffled := p.Act(&Obs{Tokens: shuffled, Parts: [][]float64{parts[2], parts[0], parts[3], parts[1]}, Global: global})
	for i, j := range perm {
		if math.Abs(outShuffled[i]-out[j]) > 1e-12 {
			t.Fatalf("actuator %d: %v != %v", j, outShuffled[i], out[j])
		}
	}
}

func TestCheckpointRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	p := New(rng, 10, 7, 4, 8, 16, 32)
	p.TokNorm.Update(randTokens(rng, 20, 10))
	var buf bytes.Buffer
	if err := p.Save(&buf); err != nil {
		t.Fatal(err)
	}
	q, err := Load(&buf)
	if err != nil {
		t.Fatal(err)
	}
	o := &Obs{Tokens: randTokens(rng, 4, 10), Parts: randTokens(rng, 3, 7), Global: randTokens(rng, 1, 8)[0]}
	a, b := p.Act(o), q.Act(o)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("output %d differs after reload: %v vs %v", i, a[i], b[i])
		}
	}
}
