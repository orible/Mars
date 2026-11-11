package model

import (
	"fmt"
	"testing"
)

// BenchmarkStanding times one control tick (20 physics steps, 1 ms of
// simulated time) of the biped standing on the floor. ns/op divided by
// 1e6 is the cost per simulated second relative to real time.
func BenchmarkStanding(b *testing.B) {
	for _, every := range []int{1, 5, 20} {
		b.Run(fmt.Sprintf("massEvery=%d", every), func(b *testing.B) {
			bp, err := loadBipedB()
			if err != nil {
				b.Fatal(err)
			}
			w := NewWorld(bp, true)
			w.MassEvery = every
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for s := 0; s < 20; s++ {
					w.Step(50e-6)
				}
			}
		})
	}
}
