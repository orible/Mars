# Mars: hydraulic biped

Simulation-first stack for a small hydraulic biped. The full plan is in [docs/architecture.md](docs/architecture.md).

**Phase 1 goal:** one movement network that walks in the simulator on a body with any number of cylinders, any mounting geometry and any centre of gravity, and keeps working as the body spec is corrected to match real hardware.

## Layout

| Path | What it is |
| --- | --- |
| `sim/geom` | 3D vectors and rotations (X forward, Y up, Z right) |
| `sim/body` | Body spec: link tree with hinge joints, cylinders pinned between two anchors, supply; validation and random bipeds |
| `sim/model` | Kinematics, cylinder length and moment arms per spanned joint, inverse dynamics, the coupled world step, reach report |
| `sim/hydraulics` | Double-acting cylinder and servo-valve pressure model (also a standalone bench model) |
| `sim/obs` | Per-cylinder tokens and a global vector; CoG computed from the links |
| `sim/policy` | Morphology-agnostic set network (shared φ per cylinder → mean pool → shared ρ), JSON checkpoints |
| `sim/cmd/simserver` | Runs the sim, hot-reloads the body spec, streams frames on `ws://localhost:8080/ws` |
| `sim/bodies` | Body specs; edit these as hardware measurements come in |
| `observer` | Blazor Server app: side and front views, cylinder pressures and valve commands, CoG |

## Run

Requires Go 1.22+ and the .NET 10 SDK.

```bash
cd sim && go test ./... && go run ./cmd/simserver
```

```bash
dotnet run --project observer
```

Open http://localhost:5217. Edit `sim/bodies/biped.json` and the server reloads it within a second, still running the same weights. An invalid edit is rejected with a reason, and the previous body keeps running.

Check how much joint range each cylinder's stroke allows for a spec:

```bash
cd sim && go run ./cmd/simserver -check
```

Cycle through random bodies (pelvis width, segment lengths, anchors, cylinder count and sizes, CoG):

```bash
cd sim && go run ./cmd/simserver -random 5s
```

## Status

- Done: 3D body spec with two-anchor cylinders; joint spans and pose-dependent moment arms (checked against finite differences); fixed-pelvis test stand dynamics coupled to the hydraulics (energy-conservation test); spec validation and reach report; network that handles any cylinder count; hot reload; observer.
- Placeholder: the network has random weights; there is no ground contact or leg self-collision.
- Next: floating base with foot contacts, then PPO training with body and CoG randomisation until it walks to commanded targets.
