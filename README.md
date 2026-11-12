# Mars: hydraulic biped

Simulation-first stack for a hydraulic humanoid: a full-size body (5 ft 10 in, 225 lb) with legs, a two-stage spine and arms. The full plan is in [docs/architecture.md](docs/architecture.md).

**Phase 1 goal:** one movement network that walks in the simulator on a body with any number of cylinders, any mounting geometry and any centre of gravity, and keeps working as the body spec is corrected to match real hardware.

## Layout

| Path | What it is |
| --- | --- |
| `sim/geom` | 3D vectors and rotations (X forward, Y up, Z right) |
| `sim/body` | Body spec: link tree with hinge joints, cylinders pinned between two anchors, supply; validation; the human body generator (adult anthropometry, cylinders sized to cover each joint's range) and random humanoids |
| `sim/cmd/bodygen` | Writes a human body spec for a height and weight and reports each cylinder's bore, stroke and torque across its range |
| `sim/model` | Kinematics, cylinder length and moment arms per spanned joint, inverse dynamics, the coupled world step, reach report |
| `sim/hydraulics` | Double-acting cylinder and servo-valve pressure model (also a standalone bench model) |
| `sim/obs` | Per-cylinder tokens, per-sensor tokens (sole pads: place, pressure, shear, slip; tendon tension), per-body-part tokens (the body's sense of its own shape: part length, extent, mass, pose, load on the floor) and a global vector, all scaled to body size; the left–right mirror of an observation |
| `sim/policy` | Morphology-agnostic set network (body parts → shape context; shared φ per cylinder → mean pool → shared ρ), input normalisation saved with the weights, critic with privileged inputs, backprop, Adam, JSON checkpoints |
| `sim/command` | The LLM loop interface: perception text in, JSON command out; the scripted LLM stand-in with latency and misjudgement |
| `sim/world` | World files (friction, objectives), random worlds, the tracker that produces the perception |
| `sim/worlds` | Hand-made worlds |
| `sim/env` | Training environment: control timing, randomisation (body, CoG, friction, pushes), command-following reward |
| `sim/ppo` | PPO with parallel rollouts and gradients, a left–right symmetry loss and a learning rate that follows the KL divergence |
| `sim/cmd/train` | Trains a scenario and writes `checkpoints/policy.json` every iteration |
| `sim/cmd/mocapref` | Turns motion capture (CMU ASF/AMC) into human motion references |
| `training` | Snapshot of the training data (checkpoints and run logs) that git carries between machines |
| `sim/cmd/simserver` | Runs the sim, hot-reloads the body spec and the policy checkpoint, streams frames on `ws://localhost:8080/ws` |
| `sim/bodies` | Body specs; edit these as hardware measurements come in |
| `tools/blender` | Blender add-on: import a body spec as an armature with pins, contacts and collision capsules, edit it, export it back |
| `observer` | Blazor Server app: side and front views, cylinder pressures and valve commands, CoG; a 3D orbit view at `/3d` with every link, joint axis, anchor, cylinder (by chamber pressure), cable, tendon (by tension), sole pad (by load) and collision capsule |

## Set up

Requires Go 1.22+ and the .NET 10 SDK. The setup script checks for them, builds everything, and restores the carried training data on a new machine. It's safe to run again.

```powershell
.\setup.ps1            # Windows; add -Install to install Go/.NET with winget, -Test to run the tests
```

```bash
./setup.sh             # macOS/Linux; --install uses Homebrew, --test runs the tests
```

To carry training to another machine, snapshot it into `training/` and commit:

```powershell
.\setup.ps1 -Save; git add training; git commit -m "training snapshot"; git push
```

On the other machine, clone and run setup. It copies the snapshot into `sim/checkpoints` and `sim/runs` when there is no local checkpoint, and `-Restore` (`--restore`) overwrites local data. Then resume with `-resume`. `-Mocap` (`--mocap`) downloads the CMU recordings, which are only needed to re-extract references.

## The body

`sim/bodies/biped.json` is generated from adult male proportions (Drillis & Contini segment lengths, Dempster/Winter masses and radii of gyration) for 1.778 m and 102.06 kg:

- **Legs:** hip yaw, roll and pitch, knee, ankle, with the small biped's cylinder layout scaled up.
- **Spine:** lumbar pitch and side bend driven by two erector cylinders, plus thoracic twist and pitch. The head is part of the chest.
- **Arms:** shoulder flexion (40° back to 90° forward), shoulder abduction (10° in to 80° out), and an elbow driven by a "triceps" cylinder (0–115°).
- **Feet:** each foot is three segments with tendons:
  - **Joints:** heel on a subtalar joint (tilts the sole 30° in, 15° out), midfoot on a sprung arch hinge, toes on a sprung toe joint.
  - **Plantar fascia:** a passive tendon from heel to toes over the ball of the foot. Bending the toes up winds it and stiffens the arch (the windlass).
  - **Stirrup:** a force-controlled cylinder in the shin pulls a cable loop behind both ankle bones. One valve tilts the sole either way and sets how firmly it's held.
  - **Pads:** nine pressure pads on each sole.

The supply is 210 bar. Each cylinder's stroke covers every combination of its joints' limits, and its bore is sized for a torque at the neutral pose. Regenerate the spec, or try another build, with:

```bash
cd sim && go run ./cmd/bodygen -out bodies/biped.json
```

```bash
cd sim && go run ./cmd/bodygen -height 1.65 -mass 70 -arms=false
```

**Collision.** Contact points (`contacts`) touch the floor. Each link can also have `colliders`, capsules given as two link-frame points and a radius in metres:

```json
"colliders": [{ "a": [0, 0, 0], "b": [0, -0.4356, 0], "radius": 0.05 }]
```

Capsules rest on the floor, using the same spring-damper and stick–slip model as the contact points. They also collide with each other, so legs and arms can't pass through each other. The exception is a link and its *collision parent* (its nearest ancestor that has capsules, so a thigh's is the pelvis, skipping the hip brackets): those two meet at their joint and never collide. Validation rejects any other pair that already overlaps in the neutral pose. The human template has capsules on the pelvis, belly, rib cage, head, thighs, shins, heels, arches, upper arms and forearms. The heel and arch capsules sit 1 cm above the sole, so the robot stands on its pads. The left and right sides are exact mirror images, which training's symmetry loss relies on. Press C in the 3D view to show or hide the capsules (tinted by the force on them) and self-contacts.

The old 0.6 m, 6 kg robot is kept as `bodies/biped-small.json`.

### Editing the body in Blender

`tools/blender/mars_body.py` is a Blender 4.x add-on that turns the spec into something you can see and drag around. It's good for catching misplaced cylinder pins (like hip-yaw bases outside the pelvis) and for shaping collision capsules.

1. Install it: Edit › Preferences › Add-ons › Install…, choose `tools/blender/mars_body.py`, then tick "Mars body spec".
2. Import with File › Import › Mars body spec (.json), and pick `sim/bodies/biped.json`. You get the body in its neutral pose (all joints at 0), standing on Blender's floor:
   - an armature with one bone per link (its head is the joint centre);
   - an empty for every cylinder pin (`cyl.knee_l.rod`), cable point, tendon point and contact point (`contact.heel_l.0`), each parented to its link's bone;
   - a wire capsule for every collider (`collider.thigh_r.0`);
   - a line through each cylinder's pins that follows the armature when you pose it.

   Each bone carries its joint axis, limits, damping, stiffness, mass, CoM and inertia as custom properties (`mars_*`, in sim units and the sim frame). They're also shown in the 3D view's sidebar under the Mars tab.
3. Edit:
   - Move pins and contacts.
   - Move, rotate or scale capsules: Z scale changes the length, X/Y scale the radius. Mars › Apply capsule scale rebuilds the mesh at its new size.
   - Duplicate a contact or capsule to add one, or delete it to remove it.
   - Re-parent a point to another bone, in the rest pose, to move it to that link.
   - Add a capsule along a bone with Mars › Add capsule to bone.
   - Move bones in Edit Mode to move joints. Whatever is parented to a bone moves with it.

   Pose the armature to check clearances through a joint's range. Export always reads the rest pose.
4. Export with File › Export › Mars body spec (.json), which defaults to the imported file. Every point is written in its link's own frame. Everything the add-on doesn't edit (hydraulics, tendons, supply, cylinder sizes) is copied unchanged. A value that moved by less than 0.01 mm keeps its exact original number, so exporting with no edits reproduces the file byte for byte, and an edit changes only what you touched. A running `simserver` reloads the file within a second. If it rejects the spec (an overlapping capsule, or a cylinder that no longer fits its stroke), it logs why and keeps the previous body. Run `go run ./cmd/simserver -check` for the full reach report.

Coordinates: the sim has X forward, Y up and Z to the robot's right. A Blender character faces −Y with Z up and its right side at −X, so sim (x, y, z) is Blender (−z, −x, y), and Blender (X, Y, Z) is sim (−Y, Z, −X). That's a rotation, not a mirror. Structural changes (adding links or cylinders, cylinder sizing) still go through `bodygen` or the JSON. Note that `bodygen` regenerates `biped.json` from the template and overwrites Blender edits.

Tests: `sim/body/blender_test.go` checks the mapping and the float32 round-trip arithmetic in Go. The add-on's own round trip runs in Blender (import, export, posed export, one edited pin, one scaled capsule):

```bash
blender -b --factory-startup -P tools/blender/test_roundtrip.py -- sim/bodies/biped.json
```

Plain Python (`python tools/blender/test_roundtrip.py`) checks the Go-style JSON writer and the mapping only. Random training bodies are humanoids of 1.55–1.95 m and a BMI of 20–33. Some have no arms, a shorter or rigid spine, no hip yaw or roll, or a second knee cylinder.

## Run

```bash
cd sim && go test ./... && go run ./cmd/simserver
```

```bash
dotnet run --project observer
```

Open http://localhost:5217, or http://localhost:5217/3d for the 3D view (drag to orbit, right-drag to pan, wheel to zoom, F to follow, hover for numbers). Edit `sim/bodies/biped.json` and the server reloads it within a second, still running the same weights. An invalid edit is rejected with a reason, and the previous body keeps running.

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

Training runs in scenarios. Each one practises something different, and all of them train the same network (`checkpoints/policy.json`), so skills carry over from one to the next:

| Scenario | What it practises |
|---|---|
| `getup` | start lying on the back, front or a side; get up and stay up (the kicking and pushing a baby learns on the floor) |
| `stand` | stand still through random pushes |
| `walk` | walk where the LLM stand-in says, held by a fading harness at first |
| `life` | all of it: start on the floor or standing, get up, walk, get pushed, fall and get back up |
| `scenarios/*.json` | your own: name a built-in in `"base"` and override fields |

```bash
cd sim && go run ./cmd/train -scenario getup -iters 400
```

```bash
cd sim && go run ./cmd/train -scenario life -resume
```

Statistics go to `runs/<scenario>.csv`, and the checkpoint is rewritten after every iteration. Watch the latest weights while training runs; the server reloads the checkpoint whenever it changes:

```bash
cd sim && go run ./cmd/simserver -policy checkpoints/policy.json -scenario getup
```

Add `-world worlds/flat.json` to walk a hand-made route instead of random objectives.

Watch the training course from the observer. The scenario bar picks any scenario and start level to watch (Watch). Follow training mirrors whatever the trainer is practising right now, including the start level the curriculum has reached, and shows its latest numbers. Start the sim server with `-follow` to begin in that mode.

Let an LLM drive the robot instead of the scripted stand-in. Read `GET http://localhost:8080/perception` for the prompt, and post its reply, such as `{"heading_deg": -20, "speed": 0.3}`, to `POST /command`:

```bash
cd sim && go run ./cmd/simserver -policy checkpoints/walk.json -commander external
```

In `walk` and `life`, standing starts may be held by a harness that carries the pelvis along the command while the legs kick and step; it fades as the robot stops falling. Unheld standing starts are logged as "solo", and floor starts as "got up n/m". `simserver -assist 1` shows the held stage.

Continue a run with `-resume`. Resuming the same scenario picks up its iteration count, start level, harness strength and learning rate from `checkpoints/policy.status.json`; `-start-level` and `-assist` on the command line still override. A checkpoint trained on an older observation layout is refused, and you start a new network instead.

Floors and feet: each episode of a built-in scenario draws a floor with bumps up to 1.5–2.5 cm, a softness from gym mat to concrete, and patches of better and worse grip; a quarter stay plain. The sole pads and tendon gauges behave like cheap sensors: some are dead for the episode, each has its own gain and offset error, and they lag and jitter. `-rough-floor` changes the bump height (0 = flat), and `-no-foot-sensors` trains without them for comparison.

How it learns: inputs are normalised by running statistics stored in the checkpoint. The critic also sees what only the simulator knows: true velocity and height, friction, harness and time left. A symmetry loss (`-symmetry`, default 0.5) asks the policy to command the mirror image of every situation the mirror way, so a skill learned on one side carries to the other. The learning rate rises and falls to keep each update's KL divergence near 0.01. The training log adds `lr` and `sym` columns. `-embed` and `-hidden` size a new network (default 64 and 128). The trainer has no real-time clock: every environment runs on its own goroutine, flat out across all cores. On a 6-core i5 that is about 18× real time during rollouts and about 11× including the network updates.

The sim server runs at real time so the view is watchable. Add `-speed 4` to fast-forward, or `-speed 0` to run as fast as one core allows. Add `-push 2` to push the robot sideways every 2 s on average, as in training.

## Status

- Done: 3D body spec with two-anchor cylinders and ground contact points; joint spans and pose-dependent moment arms (checked against finite differences); floating-base dynamics with stick–slip floor contact, coupled to the hydraulics (tests for energy, momentum, sliding distance and standing); fixed-pelvis test stand; spec validation and reach report; pelvis IMU and yaw-free heading frame in the observations; network that handles any cylinder count; hot reload; observer with floor, contact forces and fall count.
- Training: PPO written from scratch in Go (gradients checked against finite differences) trains across random humanoids, friction and pushes. The human body needs a new network: the small biped's checkpoints in `training/` use the old observation layout.
- Physics: the mass matrix uses the composite rigid body algorithm (checked against Newton–Euler). The human body costs about 0.35 s of CPU per simulated second on one core. Floor, joint-stop and cylinder end-stop stiffness grow with body mass.
- Not modelled: leg self-collision.
- Next: walk to commanded targets, the Phase 1 exit criterion.
