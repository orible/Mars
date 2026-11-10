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
- **Test stand first.** The current dynamics bolt the pelvis to the world and let the legs swing (recursive Newton–Euler, 50 µs steps coupled to the hydraulics). A floating base with foot contacts comes next and is required for walking.

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

**Physics.** 3D from the start. A pure-Go articulated-body sim covers the test stand (fixed pelvis) today. For walking it needs a floating base and foot–ground contact with friction: either extend the pure-Go sim, or call MuJoCo's C API from Go through cgo and keep the hydraulic model in Go.

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

1. Go 3D sim with the hydraulic actuator model, a data-driven body description (links, joints, two-anchor cylinders), a reach check, and the Blazor observer. Test stand (fixed pelvis) first, then a floating base with foot contacts.
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
- [ ] Contact physics for walking: extend the pure-Go sim with a floating base and foot contacts, or use MuJoCo through cgo (needs a C toolchain on Windows)?
- [ ] Leg self-collision is not modelled yet; legs can pass through each other on the test stand.
