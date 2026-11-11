# Hydraulic Biped Robot: Architecture & Build Plan

Nov 10, 2026

## Vision

Build a small, 3D-printed biped with hydraulic legs that can walk to a target, pick up an object and hand it to a person. It is trained first in a simulator written in Go and watched through a .NET Blazor observer.

**Core requirement: a stabilised, gravity-aligned sensor head at about human height.** Legs are desirable, not essential. They carry the head through stairs and clutter, but a wheeled or wheel-legged base can carry the same head. Only the movement handler and solver depend on the body type, so locomotion is swappable.

The project is also a way to learn hardware: each stage on the roadmap produces a working device on the way to the full robot.

Design principles:

- **Layered control.** Fast loops own movement; slower layers set policy and can interrupt downward.
- **Authority by speed and familiarity.** Known dangers are handled by reflexes; only novel situations escalate to reasoning.
- **Immutable baseline.** Learning adds bounded layers on top of frozen baseline abilities and never replaces them.
- **Simulation first.** Policies, skills and danger models are trained and tested in simulation before they touch hardware.

## Phase 1 core requirement: a morphology-agnostic movement network

**Phase 1 delivers one neural network that controls a body with any number of actuators and any centre of gravity.** The same weights must run a 3-actuator test rig, a 6-cylinder biped and a body that carries an off-centre payload, with no retraining and no change to the network's shape.

Consequences for the design:

- **Actuators are tokens, not fixed input slots.** Each actuator becomes a feature vector: its normalised stroke position, velocity, chamber pressures, piston areas, moment arm, and its mount position relative to the centre of gravity. One shared encoder processes every token, so the parameter count does not depend on the actuator count.
- **Pooling gives each actuator a whole-body context.** Token embeddings are pooled (mean pooling at first; attention over actuator tokens later) into a body context. A shared decoder then reads each token together with that context and outputs that actuator's valve command. The network is permutation-equivariant: reordering the actuators reorders the outputs and changes nothing else.
- **Centre of gravity is an input, not a constant.** It enters globally (mass, CoG position, IMU tilt and rate) and per actuator (mount position relative to the CoG). Training randomises it on every episode and changes it mid-episode as payloads are picked up and set down.
- **Morphology randomisation is part of domain randomisation.** Training episodes draw the actuator count, cylinder sizes, mount geometry, masses and CoG from ranges, so the network learns control in general instead of memorising one body.
- **The body description is data.** The simulator, the observation builder and the Blazor observer all read the same `Body` description, so adding or removing an actuator is a data change only.

### Body spec (3D)

The sim is 3D from the start, because much of what makes the pelvis matter is 3D: hip roll for shifting weight sideways, hip spacing, splayed cylinders.

- **Links and joints.** The body is a tree of rigid links rooted at the pelvis, each with a mass, a centre of mass and principal inertia. Links connect through hinge joints with an origin, an axis, limits and damping. A multi-axis joint, such as a hip with roll and pitch, is a chain of hinges through a light intermediate link.
- **Cylinders are pinned between two anchors.** Each anchor is a point on a link. Which joints a cylinder spans follows from the tree (a pelvis-to-thigh cylinder spans both hip joints; a pelvis-to-shin cylinder would span hip and knee). Its moment arm about every spanned joint is computed from the current pose each step, and its force acts along the line between the pins. Strength comes from the bore and rod areas times its pressure limit (supply or its own relief setting).
- **Pelvis configuration is data.** Hip spacing, hip drop and every anchor point on the pelvis plate live in the spec file.
- **Geometry validation.** A spec is rejected if any cylinder cannot be assembled in the neutral pose. The reach report (`simserver -check`) lists, for each cylinder and each joint it spans, how much of the joint's range the stroke actually allows. Use it to adjust anchor points before printing parts.
- **Joint ranges follow a human range-of-motion chart** where one linear cylinder can reach, checked with the anchor search and `-check`:

  | Joint | Biped | Human chart | Walking uses |
  |---|---|---|---|
  | Hip yaw | 40° internal .. 45° external rotation | about 35–45° in, 45–50° out | turning, and toeing in or out |
  | Hip roll | 30° adduction .. 45° abduction | same | small |
  | Hip pitch | −20° .. 90° | −20° .. 120° | about −10° .. 30° |
  | Knee | 3° hyperextension lock .. 90° flexion | 0 .. 135° | about 0 .. 60° |
  | Ankle | 45° plantar .. 20° dorsi | same | about 20° plantar .. 10° dorsi |

  - **Hip yaw comes first in the chain** (pelvis → yaw → roll → pitch). The yaw link is a bracket that carries the roll and pitch cylinders' bases, so turning a leg never stretches them. A horizontal cylinder from the back of the pelvis to a lever on the bracket gives the full yaw range with a 25 mm minimum lever arm (about 11 N·m).
  - **Muscle placement is anatomical.** The knee cylinder is a "quad" down the front of the thigh to the front of the shin. The ankle cylinder is a "calf" down the back of the shin to the heel.
  - **Knee depth limit.** A straight quad loses its lever beyond about 90° of knee flexion. Humans keep it with the kneecap, which works as a pulley; a kneecap-style linkage would be the way to reach 110–135°.
  - **Knees lock.** Joint limits are stiff mechanical stops, and the knee's stop sits 3° past straight. On a straight leg the weight pushes the knee into the stop, so standing needs no knee cylinder force, like the human screw-home lock.
- **Contact points are data.** Each link can list points that touch the ground: sole corners on the feet, plus kneecaps and pelvis corners so falls land properly.
- **Floor or test stand.** By default the pelvis floats and the robot stands on a flat floor (recursive Newton–Euler with a free base, 50 µs steps coupled to the hydraulics). `simserver -stand` bolts the pelvis to the world so the legs swing freely.

**Phase 1 exit criterion: the network walks in the simulator** to commanded targets across the randomised body range, not just holds a pose.

### Iteration loop: spec, sim, train, measure

The real robot will not match the spec the trainer started from. Cylinder bores, valve response, hose compliance, masses and CoG will all differ, and they change as the robot is rebuilt. The pipeline is designed to iterate rather than restart:

1. **Spec as data.** Each body is a JSON spec (`sim/bodies/*.json`). The sim server watches the file and hot-reloads it, so an edit shows up in the observer within a second.
2. **Weights independent of the spec.** Checkpoints store only network weights. Because the network is morphology-agnostic, an old checkpoint runs on an edited spec, and training continues from it (fine-tuning) instead of starting again.
3. **Measure and write back.** Bench tests (hardware ladder step 5 onwards) measure real cylinder and valve behaviour (step response, pressure rise, friction) and the robot's real mass and CoG. The measured values replace guesses in the spec.
4. **Randomise around the measurement.** Domain randomisation is centred on the latest measured spec, with ranges wide enough to cover what is still unknown.
5. **Regression gate.** After each spec change and fine-tune, the frozen evaluation suite (stand, walk to target, push recovery) must still pass on both the old and the new spec before a checkpoint is promoted.

The **movement handler** in the cognitive stack below is this network. It stays body-specific in the sense that it is conditioned on the body description, but it no longer has to be rebuilt per body.

## Cognitive stack

Layers run at different speeds: lower layers have the final say on movement, higher layers set overarching policy. Commands, goals and tightened limits flow down; events and interrupts flow up.

| Layer (slowest at top) | Rate | Responsibility |
| --- | --- | --- |
| Big LLM controller | seconds | Reasoning, planning and conversation; turns requests into goals. Writes skills and reward functions; resolves escalations. |
| General lifecycle (fast LLM) | sub-second, event-driven | Watches events, keeps the task going, picks the next skill. Handles routine failures; escalates when stuck or ambiguous. |
| Skill coordinator (scaffolding) | ~10 Hz | Runs library skills (walk_to, grasp, hand_over). Consolidation gradually distils it into networks. |
| Movement handler (learned) | 100–500 Hz | Goal-conditioned, morphology-agnostic network: balance, gait, stumble recovery. Reads IMU, joint angles, pressures, the body description and the current command. |
| Movement solver | ~1 kHz, hard real time | Valves, pressures, joint targets. Hard limits and the human-safety floor, backed by a safety MCU. Body-specific. |
| Head stabiliser (parallel to solver) | ~1 kHz | Gimbal/neck keeps IMU and camera level, giving all layers a calm horizon. |

Each layer sends commands down and events up. An event rises only as far as needed: a slip is handled by the movement handler, while a failed grasp reaches the fast LLM or the big LLM. LLMs never touch joints; they call skills, and the layers below can veto anything unsafe.

**The skill coordinator is scaffolding.** It starts as thin code because three things need a deterministic home: running LLM-written skills, enforcing the safety envelope, and emitting clear events. The skills it chains are mostly trained policies. Like evolution and childhood for a human, it stands in for experience the robot does not have yet. As experience accumulates, consolidation distils its decisions into networks, shadow-tests them and retires the code piece by piece. Only the hard safety floor is never replaced, just as human withdrawal reflexes are innate.

## Safety and authority

Authority goes to whichever layer is fast enough and knows the situation, not automatically to the highest layer. Avoiding a known danger is a reflex; only a novel danger needs reasoning.

- **Safety floor at the bottom.** Human-proximity limits on speed and force live in the solver and handler and run in milliseconds. The approach follows industrial cobot practice in ISO/TS 15066: speed-and-separation monitoring, power and force limiting.
- **Tighten, never loosen.** Higher layers may make human-safety limits stricter ("careful mode near the toddler") but can never relax them. Non-safety envelope settings, such as stance width, may be relaxed within bounds.
- **Novelty: freeze, then escalate.** When a lower layer meets something it cannot classify, it goes to a safe state (stop, go compliant, step back) and then reports upward. LLM layers decide what happens next; they never act during the dangerous moment.
- **Solved novelty becomes reflex.** Once the big LLM resolves a new hazard, it becomes a rule or training scenario for the lower layers.
- **Independent safety controller.** A separate, simple microcontroller (for example an ESP32) enforces the human-safety floor and pressure limits, so a bug in learned layers cannot disable it.

## Learning without retraining the LLMs

The robot gets more capable by growing a skill library and retraining small, bounded models. The LLMs themselves are never retrained.

**Skill library (written by the big LLM).** Following the Voyager pattern, the big LLM writes composite skills as scripts that call the skill API (`find`, `walk_to`, `grasp`, `place`, `hand_over`). Skills are hot-loaded, tested in simulation, then promoted. Following the Eureka pattern, it can also write reward functions so the simulator retrains the movement handler for new physical skills, such as stepping over an obstacle.

**Danger vision (distillation with active learning).**

1. A small, fast detector scores every camera frame for hazards.
2. Uncertain frames go to the big vision-language model, which labels them.
3. The labels build a dataset that periodically retrains the small detector.
4. Starter open vision models (detection, segmentation, depth) seed the first labels.

**Environmental danger map.** A floor-plan risk map fuses vision with pressure spikes and IMU jolts. Risk fades unless reinforced, and the navigator treats high-risk areas as costly.

**Retraining without forgetting.** The baseline is immutable; learning only adds bounded layers on top.

- **Frozen base + adapters:** small LoRA-style layers are trained on a locked base and can be removed or switched ("carrying mode", "stairs").
- **Residual policies:** the new movement network outputs a clipped correction added to the frozen baseline's output.
- **Replay:** every retraining batch mixes in original training data.
- **Old model as teacher:** the new model is penalised for disagreeing with the old one on old scenarios.
- **Regression gate:** a frozen suite (walking, push recovery, known hazards) must be passed at least as well as before.
- **Shadow mode and rollback:** a new model runs alongside the live one before promotion; every version is kept.

**Modulation signals (the "chemical" layer).** Like dopamine, noradrenaline and acetylcholine, a few learned broadcast signals carry no skill detail. They report reward, raise attention or caution, and gate when weights may change. "Careful mode" becomes a modulation signal that changes the gain of every layer at once, raised by the danger models, rather than a coordinator rule.

**Day/night consolidation ("baked as neurons").**

1. **Day:** only small adapters change online, driven by the reward and danger signals.
2. **Night:** the robot replays logged experience in the simulator and distils the adapters into the base networks, as sleep consolidates memories from the hippocampus into the cortex.
3. **Gate:** the regression suite must pass before the consolidated network goes live; otherwise the adapters stay separate.

**Guardrails on self-training.** Learned danger models can only add caution. Missed dangers are weighted above false alarms. New hazard categories need human approval.

## Simulation and training

Most world-model basics are learned in simulation; real hardware only receives policies that already pass there.

**Physics.** 3D, pure Go, no C toolchain needed. The pelvis is a free 6-DOF base. Ground contact is a spring-damper per contact point, with stick–slip Coulomb friction: a point sticks where it touched down until the tangential force leaves the friction cone, then slides at μ·N. The hydraulics already force 50 µs steps, so this simple contact model is stable. Tests check momentum conservation in free flight, the Coulomb sliding distance, and the biped standing with its valves closed. The factored mass matrix is reused for 5 steps; one simulated second costs about 0.22 s on one core. Floor stiffness, damping and friction are `model.Ground` parameters, ready for randomisation. Cross-checking against MuJoCo by exporting the spec remains an option if the contact model is ever in doubt.

**Hydraulic actuator model**, stepped alongside the physics for each cylinder:

- Chamber pressures driven by valve flow and fluid compressibility.
- Force = P_A·A_A − P_B·A_B, minus friction, mapped to joint torque through the linkage geometry.
- Servo-valve lag and saturation.
- Small timesteps for pressure dynamics, offset by running many walkers in parallel on goroutines.

**Sensing.** Observations are joint angles, chamber pressures (load and foot contact) and a fused IMU at the pelvis: gyro plus accelerometer for "up" and angular rate.

**Goal-conditioned policy.** One network learns to reach any commanded target given in the robot's own frame ("1.2 m ahead, 0.4 m left"). Training details:

- Random targets per episode, changed mid-walk.
- Random bodies per episode: actuator count, cylinder sizes, mount geometry, mass and centre of gravity (see Phase 1 core requirement).
- Reward for progress, staying upright and a mocap imitation term, minus hydraulic effort.
- Random pushes and payloads, so recovery and carrying are part of the same skill.

**Worlds and navigation: walk to the horizon, no map.** Humans head for an objective they can see or know the direction of, and deal with obstacles as they meet them. They remember what they have seen (object permanence) but do not carry a global map. The walking stack is built the same way:

- **Loadable worlds.** A world is a data file, like a body spec. It holds the ground and, later, obstacles and regions with their own friction, plus one objective: a far-off point. The trainer generates random worlds; hand-made ones are for tests and demos. The sim server and the trainer both take a world file.
- **The objective is perceived by the LLM loop, not known to the legs.** The movement network has no innate sense of objectives or paths. It only follows the current command: a heading relative to where it faces, and a speed, where 0 means stand. Seeing the objective and deciding where to go is the fast LLM layer's job, and it runs as a slow input/output loop:
  - **Input:** a perception message, e.g. "The objective is about 4.8 m away, 22° to your left." In the sim it is computed from the world; on hardware it comes from the camera.
  - **Output:** a JSON command, e.g. `{"heading_deg": -22, "speed": 0.3}`.
  - **Repeat:** the robot moves, the view changes, the LLM looks again.

  In training, a scripted commander (`command.Scripted` inside `command.Loop`) stands in for the LLM. It looks every 0.5–1.5 s, answers after 0.2–0.8 s, misjudges angles (±8°) and distances (±15%), and slows down as the objective grows near. The movement network is therefore rewarded only for tracking the commanded velocity while upright; standing is the stop command. The sim server exposes the same loop (`GET /perception`, `POST /command`, `-commander external`), so a real LLM can drive it.
- **First-person perception, added later.** Depth rays from a head-mounted viewpoint, matching the real depth camera, enter as another set of tokens, so the ray count can change too. No map is ever passed in.
- **Object permanence, added later.** A recurrent memory (GRU) in the policy carries what was seen. An obstacle that has dropped out of view, such as a box now underfoot or behind, is still accounted for.
- **Stages:**
  1. Flat world, walk where the LLM loop says until the objective grows near; random objective directions and distances (in training now).
  2. Obstacles and terrain, with collision against the contact points.
  3. Depth rays.
  4. Memory.
  5. An obstacle curriculum.

**Learning to kick by getting up off the floor.** Babies learn to kick and push hard by lying on the floor and struggling to roll, sit and get up, long before they walk. Training does the same:

- **Floor starts.** Episodes can start lying on the back, front or a side, with the legs at random angles. Contact points on the thighs, shins and pelvis corners let the body rest on the floor.
- **A fall is not the end.** Being down pays little. Getting higher and more upright pays as it happens (potential-based shaping, so it cannot be farmed). Being up pays far more and switches to the command-following reward.
- **Getting up is part of the same skill.** While the robot is down, the LLM loop is told it has fallen and its command is to stand still.
- **Reverse curriculum for getting up,** the way babies come to it. Floor starts go down a ladder, and the trainer moves the start level deeper as get-ups succeed:
  - a squat balanced over flat feet
  - kneeling
  - sitting
  - lying on the back, front or side
- **What the tests show so far:**
  - **Strength:** the quads straighten the knees under the full weight and lift the pelvis to 91% of standing height.
  - **Balance:** this is the hard part. The squat already uses the hip's 90° flexion and the ankle's 20° dorsiflexion limits, so there is little range left to keep the weight over the feet on the way up. Humans have about 120° and 35–40°. A hand-written controller rises and then sits back down.
  - **If training cannot find a way,** the hardware answer is more hip flexion and ankle dorsiflexion range.
- **Following a human: motion references.** A reference is a human movement written as keyframes of trunk lean and hip, knee and ankle angles, with sources. Each one is a JSON file in `sim/env/references`.
  - **Squat-to-stand** comes from the sit-to-stand literature: flexion-momentum, momentum transfer (seat-off), then extension.
  - **Retargeting:** each body gets its own version, clipped to its joint limits, with joint roles found from link names.
  - **Tracking is state-based:** each step finds the nearest point on the reference path. While down, being close to the human pose and moving along the path towards standing both pay, capped below what being up pays.
  - **No new network inputs,** so references can be added without retraining from scratch.
  - **The balanced squat start matches the human at about seat-off** (64% along, 12° rms off), so the reference mainly teaches the extension phase.
  - **Motion capture fills the floor rungs.** Two CMU takes of subject 114 getting up off the floor (114_02, 114_11; free for research and commercial use per CMU) are turned into three references by `cmd/mocapref`:
    - kneel-to-stand
    - floor-to-stand (from lying on the back)
    - side-to-stand
  - **Calibration:** angles are measured relative to her standing pose. Both takes agree on it (about 31° lean, 45° hips, 31° knees, from marker placement and posture).
  - **Ankles are left out,** because the foot data disagree between legs.
  - **The hand-supported moments are imitated by the legs and trunk only.** The subject was pregnant and can use her hands; the robot has no arms.
  - **Each episode tracks the reference that best fits its start:**
    - squat → squat-to-stand
    - kneel → side-to-stand
    - sit or lying → floor-to-stand
  - **More recordings:** subjects 139 and 140 add nine dedicated get-up takes. CMU's subject 77 takes are byte-identical copies of 139's, so they were dropped.
    - face down: 2
    - on the side: 2
    - on the back: 3
    - from the ground: 2
  - **Automatic extraction:** `cmd/mocapref -auto` finds each rise, from the last moment near the floor to the straightest standing moment. It calibrates to that straightest pose rather than the end of the clip, because subjects shift about afterwards.
  - **Coverage:** 13 references in all; scenarios use `"reference": "all"`. Each floor start picks a close match:
    - squat: 10°
    - kneel: 8°
    - sit: 20°
    - lying: 13°
  - **The raw recordings stay out of git** (`sim/mocap/cmu-*`). Only the extracted keyframes, with attribution, are committed.
- **Longer practice:**
  - getup episodes last 20 s, so slow rises are not cut off;
  - each iteration collects 256 steps per environment;
  - exploration noise never drops below σ = 0.2 (it had collapsed to about 0.13 before get-ups were found).
- **Scenarios** (`-scenario` on the trainer and the sim server) choose the mix:
  - `getup`: always start on the floor.
  - `stand`: stand through pushes.
  - `walk`: walk where told.
  - `life`: everything at once.
  - Scenario files in `sim/scenarios/` combine these. All of them train the same network.

**Learning to step like a baby: held, carried and kicking.** Babies first learn to step by being held under the arms and carried forward while they kick. Training does the same:

- **The harness** (`model.Harness`) takes up to half the weight, keeps the pelvis upright, and carries it along at the commanded velocity.
- **A stepping reward** (the kicking hint) pays for feet that land after a 0.2–0.5 s swing while the command says to move, and charges for shuffles. The feet are whichever links carry the lowest contact points, so this works for any body.
- **The trainer fades the harness** when held episodes rarely fall and brings it back when they do. One episode in five is never held, so real ability is always being practised and logged ("solo" in the training log).
- **Watching:** `simserver -assist 1` shows the held stage.

**Human-like motion.** Use Adversarial Motion Priors (AMP) or DeepMimic-style imitation of human motion-capture clips (CMU, AMASS), retargeted to the robot's leg proportions.

**Domain randomisation.** Randomise fluid properties, valve response, friction, masses, centre of gravity, actuator layout and sensor noise so policies transfer to hardware.

**Vision starter models.** Use open pretrained detection, segmentation and depth models as teachers to label simulated and real camera data. The sim's own renders provide perfectly labelled synthetic images.

## Software stack

Go runs the simulator, trainer and robot runtime; Blazor is the window into all of it.

| Component | Technology | Role |
| --- | --- | --- |
| Simulator | Go (pure 2D first; MuJoCo via cgo for 3D) | Physics, hydraulic model, parallel environments |
| Trainer | Go, hand-written MLP + PPO | Goal-conditioned, morphology-agnostic movement policy, residual retraining |
| Layer runtime | Go goroutines + channels | Each cognitive layer is a goroutine: command channel down, event channel up |
| Shared state | Go structs | Task, location, held object, battery; LLMs read compact summaries, so a restart loses nothing |
| Skills | Starlark (or Lua via gopher-lua), embedded | Sandboxed, hot-loadable scripts that can only call the skill API |
| Skill store | Git repository | Versioning and rollback of LLM-written skills |
| LLM access | HTTP with tool calling | Fast LLM (small local or fast hosted model) and big LLM (reasoning) |
| Transport | WebSocket, JSON or MessagePack frames | Streams joints, pressures, IMU, rewards and layer state |
| Observer | .NET Blazor + canvas | Live biped view, pressure and reward plots, layer states, danger heat map, skill library, flagged-frame review queue, click-to-set targets |

## Hardware

The mini robot uses hobby RC-excavator hydraulics for the legs, electric servos for the arms and hands, and costs roughly $500–1,200 in parts.

| Part | Choice | Approx. cost (USD) |
| --- | --- | --- |
| Leg cylinders (×6: hip, knee, ankle per leg) | Double-acting RC cylinders, ~$33–51 each | 200–300 |
| Pump | Brushless gear pump, ~$50 (premium Leimbach units ~€250) | 50–270 |
| Valves (6 channels) | Servo-driven spool valve blocks | 100–200 (estimate) |
| Pressure sensors | Thin-film, 0.5–4.5 V output, up to 60 bar; ~$12–50 each | 75–600 |
| IMU | BNO085 (on-chip fusion) or MPU-6050 | < 20 |
| Arms and gripper | Hobby servos, 3D-printed parallel gripper | not yet costed |
| Camera | Depth camera on the head | not yet costed |
| Safety controller | Separate ESP32 | < 10 |

- **Pressure range:** hobby systems run at about 3–4 MPa (30–40 bar), so 60-bar sensors cover it.
- **Tether first:** pump, battery and oil are heavy for a small walker, so keep them off-board with hoses at first. The sim models the same tethered setup.
- **What to print:** frame and linkages can be 3D-printed. Anything that holds pressure (fittings, manifolds) stays metal.
- **Electric hands, hydraulic legs:** at mini scale, fingers are too small for cylinders and leaks are hard to manage.

## Roadmap

Two tracks run side by side: a hardware ladder where every step is a working device, and a software track that builds the full stack in simulation.

**Hardware ladder**

1. ESP32 + IMU on a breadboard, streaming fused "up" to a Blazor page.
2. Self-balancing two-wheeled robot: feedback loops, motor drivers, tuning. With a stabilised head and an arm added, it is already a fetch-and-hand-over platform.
3. 3D-printed servo arm with gripper: kinematics and grasping.
4. Small servo biped: first transfer of a sim-trained policy to hardware.
5. Single hydraulic cylinder with a pressure sensor on the bench, validating the actuator model.
6. Tethered hydraulic legs, then arms, camera and safety controller.

**Software track**

1. Go 3D sim with the hydraulic actuator model, a data-driven body description (links, joints, two-anchor cylinders), a reach check, and the Blazor observer. Test stand (fixed pelvis) and a floating base standing on the floor with contact friction (done).
2. **Phase 1 core:** the morphology-agnostic movement network, trained across random actuator counts and centres of gravity. First a balance/hold-pose task, then goal-conditioned walking ("reach this x-position"), then 2D targets, pushes and payloads.
3. Arm reach-and-grasp policy in the sim.
4. Combined walk + grasp + hand-over through the skill API.
5. Fast LLM lifecycle loop, prototyped against a fake event stream.
6. Big LLM planner, Starlark skill library and sim promotion pipeline.
7. Danger vision distillation and danger map.
8. Retraining pipeline: adapters or residuals, replay, regression gate, shadow mode.

## Open questions

- [ ] Leg dimensions and target height, which set cylinder sizes and stroke lengths.
- [ ] One pressure sensor per cylinder or one per chamber?
- [ ] Blazor observer: watch only, or also control training (pause, tweak parameters, load checkpoints)?
- [ ] Fast LLM: small local model on a Jetson-class board, or a fast hosted model?
- [ ] Which depth camera and which starter vision models?
- [ ] Workstation spec for training: CPU cores, GPU availability.
- [ ] Movement network pooling: is mean pooling enough, or is attention over actuator tokens needed from the start? What range of actuator counts and CoG offsets must phase 1 cover?
- [ ] Movement network structure: cylinders that span two joints need more than a fixed-size token. Move to a graph network over links, joints and cylinders?
- [x] Contact physics for walking: pure Go (floating base plus point contacts with stick–slip friction). MuJoCo stays an optional cross-check.
- [ ] Leg self-collision is not modelled yet; legs can pass through each other.
- [ ] Floor variety: slopes, steps and soft ground are not modelled; the floor is flat at y = 0.
