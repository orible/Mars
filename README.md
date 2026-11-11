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
| `sim/policy` | Morphology-agnostic set network (shared φ per cylinder → mean pool → shared ρ), value network, backprop, Adam, JSON checkpoints |
| `sim/command` | The LLM loop interface: perception text in, JSON command out; the scripted LLM stand-in with latency and misjudgement |
| `sim/world` | World files (friction, objectives), random worlds, the tracker that produces the perception |
| `sim/worlds` | Hand-made worlds |
| `sim/env` | Training environment: control timing, randomisation (body, CoG, friction, pushes), command-following reward |
| `sim/ppo` | PPO with parallel rollouts and parallel gradients |
| `sim/cmd/train` | Trains the standing task and writes `checkpoints/stand.json` every iteration |
| `sim/cmd/simserver` | Runs the sim, hot-reloads the body spec and the policy checkpoint, streams frames on `ws://localhost:8080/ws` |
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

The robot stands on the floor by default. With the untrained policy it falls within a couple of seconds and is put back upright 1 s later. Close every valve so the cylinders lock the joints and it stands still:

```bash
cd sim && go run ./cmd/simserver -hold
```

Bolt the pelvis to the world instead (test stand, legs swing freely):

```bash
cd sim && go run ./cmd/simserver -stand
```

## Train

Train walking: the movement network follows commands from a scripted stand-in for the LLM loop, which heads for random objectives. Training uses the biped spec plus random bodies, random floor friction and random pushes. Statistics go to `runs/walk.csv`, and the checkpoint is rewritten after every iteration:

```bash
cd sim && go run ./cmd/train -task walk -out checkpoints/walk.json -log runs/walk.csv
```

Watch the latest weights in a hand-made world while training runs. The server reloads the checkpoint whenever it changes:

```bash
cd sim && go run ./cmd/simserver -policy checkpoints/walk.json -world worlds/flat.json
```

Let an LLM drive the robot instead of the scripted stand-in. Read `GET http://localhost:8080/perception` for the prompt, and post its reply, such as `{"heading_deg": -20, "speed": 0.3}`, to `POST /command`:

```bash
cd sim && go run ./cmd/simserver -policy checkpoints/walk.json -commander external
```

`-task stand` trains standing only.

Continue a run with `-resume`. The trainer has no real-time clock: every environment runs on its own goroutine, flat out across all cores. On a 6-core i5 that is about 18× real time during rollouts and about 11× including the network updates.

The sim server runs at real time so the view is watchable. Add `-speed 4` to fast-forward, or `-speed 0` to run as fast as one core allows. Add `-push 2` to push the robot sideways every 2 s on average, as in training.

## Status

- Done: 3D body spec with two-anchor cylinders and ground contact points; joint spans and pose-dependent moment arms (checked against finite differences); floating-base dynamics with stick–slip floor contact, coupled to the hydraulics (tests for energy, momentum, sliding distance and standing); fixed-pelvis test stand; spec validation and reach report; pelvis IMU and yaw-free heading frame in the observations; network that handles any cylinder count; hot reload; observer with floor, contact forces and fall count.
- Training: PPO written from scratch in Go (gradients checked against finite differences) trains the standing task across random bodies, friction and pushes. The checkpoints are not in git; retrain with `go run ./cmd/train`.
- Not modelled: leg self-collision.
- Next: walk to commanded targets, the Phase 1 exit criterion.
