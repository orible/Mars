// 3D orbit view of the robot: every link, joint axis, anchor, cylinder
// (barrel coloured by its cap-side pressure, rod by its rod-side
// pressure), cable run and tendon (coloured by tension), sole pad
// (coloured by pressure), collision capsule (tinted by the force on it),
// self-contact and ground force, drawn with a small perspective
// renderer on a 2D canvas (no libraries, works offline).
//
// Drag to orbit, right-drag or shift-drag to pan, wheel to zoom, F to
// follow the robot again, hover anything for its numbers.
//
// World frame: X forward, Y up, Z to the robot's right.
//
// Test-world terrain (stairs, slopes, noise ground, objects) arrives as a
// separate {"terrain": …} message; the floor grid follows its height
// field and the solids are drawn as shaded boxes and cylinders.

import { fieldHeight, heightAt, faces, colour } from "./terrain.js";

let socket = null;
let latest = null;
let terrain = null; // latest terrain message (null: plain floor)
let rafId = 0;
let statusEl = null;
const cam = { yaw: -2.3, pitch: 0.3, dist: 3.4, target: [0, 0.9, 0], follow: true };
let mouse = null; // last pointer position on the canvas
let drag = null;
let showColliders = true;

export function connect(url, canvas, status) {
  disconnect();
  statusEl = status;
  setStatus("connecting");
  socket = new WebSocket(url);
  socket.onopen = () => setStatus("connected");
  socket.onclose = () => setStatus("disconnected");
  socket.onerror = () => setStatus("error");
  socket.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.terrain !== undefined) { terrain = m.terrain && m.terrain.name ? m.terrain : null; return; }
    latest = m;
  };
  bindInput(canvas);
  const loop = () => {
    if (latest) draw(canvas, latest);
    rafId = requestAnimationFrame(loop);
  };
  rafId = requestAnimationFrame(loop);
}

export function disconnect() {
  cancelAnimationFrame(rafId);
  if (socket) {
    socket.onclose = null;
    socket.close();
    socket = null;
  }
}

function setStatus(s) {
  if (statusEl) { statusEl.textContent = s; statusEl.className = "status " + s; }
}

function bindInput(canvas) {
  if (canvas.dataset.bound) return;
  canvas.dataset.bound = "1";
  const pos = (e) => {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * canvas.width / r.width, (e.clientY - r.top) * canvas.height / r.height];
  };
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    drag = { p: pos(e), pan: e.button === 2 || e.shiftKey };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("pointerleave", () => { mouse = null; });
  canvas.addEventListener("pointermove", (e) => {
    const p = pos(e);
    mouse = p;
    if (!drag) return;
    const dx = p[0] - drag.p[0], dy = p[1] - drag.p[1];
    drag.p = p;
    if (drag.pan) {
      const { right, up } = basis();
      const s = cam.dist / 900;
      for (let k = 0; k < 3; k++) cam.target[k] += (-dx * right[k] + dy * up[k]) * s;
      cam.follow = false;
    } else {
      cam.yaw -= dx * 0.008;
      cam.pitch = Math.max(-0.2, Math.min(1.45, cam.pitch + dy * 0.008));
    }
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    cam.dist = Math.max(0.4, Math.min(12, cam.dist * Math.exp(e.deltaY * 0.001)));
  }, { passive: false });
  window.addEventListener("keydown", (e) => {
    if (e.key === "f" || e.key === "F") cam.follow = true;
    if (e.key === "c" || e.key === "C") showColliders = !showColliders;
  });
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);
const unit = (a) => { const n = norm(a) || 1; return scale(a, 1 / n); };

function basis() {
  const eye = add(cam.target, [
    cam.dist * Math.cos(cam.pitch) * Math.cos(cam.yaw),
    cam.dist * Math.sin(cam.pitch),
    cam.dist * Math.cos(cam.pitch) * Math.sin(cam.yaw),
  ]);
  const fwd = unit(sub(cam.target, eye));
  const right = unit(cross(fwd, [0, 1, 0]));
  const up = cross(right, fwd);
  return { eye, fwd, right, up };
}

// Pressure (0..1 of supply) or load to a colour: blue (low) to red (high).
function heat(t) {
  t = Math.max(0, Math.min(1, t));
  return `hsl(${Math.round(220 - 220 * t)}, 85%, ${Math.round(48 + 6 * Math.sin(Math.PI * t))}%)`;
}

function floorHeight(f, x, z) {
  return fieldHeight(terrain, f, x, z);
}

function draw(canvas, f) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = css("--panel");
  ctx.fillRect(0, 0, W, H);

  if (cam.follow && f.pelvis) {
    const a = 0.1;
    cam.target[0] += (f.pelvis[0] - cam.target[0]) * a;
    cam.target[2] += (f.pelvis[2] - cam.target[2]) * a;
    cam.target[1] += (Math.max(0.3, f.cog[1]) - cam.target[1]) * a;
  }
  const { eye, fwd, right, up } = basis();
  const focal = 0.9 * H;
  const proj = (p) => {
    const d = sub(p, eye);
    const z = dot3(d, fwd);
    if (z < 0.05) return null;
    return { x: W / 2 + focal * dot3(d, right) / z, y: H / 2 - focal * dot3(d, up) / z, z };
  };

  // Primitives: collected, sorted far to near, then drawn.
  const prims = [];
  const line = (a, b, color, width, info, dash) => {
    const pa = proj(a), pb = proj(b);
    if (!pa || !pb) return;
    prims.push({ kind: "line", pa, pb, z: (pa.z + pb.z) / 2, color, width, info, dash });
  };
  const ball = (p, r, color, info, outline) => {
    const pp = proj(p);
    if (!pp) return;
    prims.push({ kind: "dot", pa: pp, z: pp.z, r, color, info, outline });
  };
  const poly = (pts, color, width, info, dash) => {
    for (let i = 1; i < pts.length; i++) line(pts[i - 1], pts[i], color, width, info, dash);
  };
  // A capsule is a translucent round-capped stroke as wide as it looks at
  // its depth; it sorts slightly behind what sits on its axis.
  const capsule = (a, b, r, color, info) => {
    const pa = proj(a), pb = proj(b);
    if (!pa || !pb) return;
    const px = focal * r * (1 / pa.z + 1 / pb.z);
    prims.push({ kind: "capsule", pa, pb, z: (pa.z + pb.z) / 2 + r, px, color, info });
  };

  // A filled polygon (a face of a terrain solid), culled when it faces
  // away from the camera.
  const face = (pts, n, color, info) => {
    if (dot3(n, sub(eye, pts[0])) <= 0) return;
    const pp = pts.map(proj);
    if (pp.some((q) => !q)) return;
    prims.push({ kind: "face", pp, z: pp.reduce((a, q) => a + q.z, 0) / pp.length, color, info });
  };

  // Terrain solids: stair steps, platforms and objects, lit from above
  // and the front left; and the course's goal as a post.
  if (terrain) {
    const light = unit([0.4, 1, -0.5]);
    for (const p of terrain.prims || []) {
      const info = {
        title: p.role === "object" ? `${p.kind} (object)` : p.role,
        rows: [`${(100 * p.size[0]).toFixed(0)} × ${(100 * (p.center[1] + p.size[1] / 2)).toFixed(0)} cm top × ${(100 * p.size[2]).toFixed(0)} cm`, p.role === "object" ? `appearance seed ${p.seed}` : `top at ${(100 * (p.center[1] + p.size[1] / 2)).toFixed(1)} cm`],
      };
      for (const fc of faces(p)) face(fc.pts, fc.n, colour(p, 0.5 + 0.5 * Math.max(0, dot3(unit(fc.n), light))), info);
    }
    const g = terrain.goal;
    if (g && (g[0] || g[1])) {
      const base = [g[0], heightAt(terrain, f, g[0], g[1]), g[1]];
      const info = { title: "course goal", rows: [`x ${g[0].toFixed(2)} m, z ${g[1].toFixed(2)} m`] };
      line(base, add(base, [0, 1.2, 0]), css("--warn"), 3, info);
      ball(add(base, [0, 1.2, 0]), 5, css("--warn"), info, true);
    }
  }

  // Floor grid (follows the floor's bumps and the terrain's height field).
  const grid = css("--grid"), g0 = 0.25, span = terrain && terrain.grid ? 4 : 3;
  const cx = Math.round(cam.target[0] / g0) * g0, cz = Math.round(cam.target[2] / g0) * g0;
  for (let i = -span / g0; i <= span / g0; i++) {
    for (const along of ["x", "z"]) {
      let prev = null;
      for (let j = -span / g0; j <= span / g0; j += 0.5) {
        const x = along === "x" ? cx + j * g0 : cx + i * g0;
        const z = along === "x" ? cz + i * g0 : cz + j * g0;
        const p = [x, floorHeight(f, x, z), z];
        if (prev) line(prev, p, grid, 1);
        prev = p;
      }
    }
  }

  const weight = Math.max(1, (f.mass || 1) * 9.81);
  const supply = f.supply || 1;
  const text = css("--text"), muted = css("--muted-text");

  // Bones, joint axes, centres of mass.
  const hasChild = new Set(f.links.map((l) => l.parent));
  f.links.forEach((l, i) => {
    const info = { title: l.name, rows: [`mass ${l.mass.toFixed(2)} kg`] };
    if (l.parent >= 0) line(f.links[l.parent].origin, l.origin, text, 5, info);
    if (!hasChild.has(i) && i > 0) line(l.origin, add(l.origin, scale(sub(l.com, l.origin), 2)), text, 5, info);
    if (i > 0 && l.axis) {
      line(sub(l.origin, scale(l.axis, 0.04)), add(l.origin, scale(l.axis, 0.04)), css("--accent"), 2, { title: `${l.name} joint axis`, rows: [] });
    }
    ball(l.origin, 3.5, text, info);
    ball(l.com, 2 + Math.sqrt(l.mass) * 0.8, muted, info, true);
  });

  // Collision capsules, tinted by the force on them (floor and each
  // other), and where two of them touch. C hides them.
  if (showColliders) {
    for (const c of f.colliders || []) {
      const info = { title: `${c.link} collider`, rows: [`radius ${(1000 * c.r).toFixed(0)} mm · length ${(1000 * norm(sub(c.b, c.a))).toFixed(0)} mm`, `pressed ${c.f.toFixed(0)} N`] };
      capsule(c.a, c.b, c.r, c.f > 1 ? heat(c.f / (0.5 * weight)) : muted, info);
    }
    for (const h of f.hits || []) {
      const info = { title: "self-contact", rows: [`${norm(h.f).toFixed(0)} N`] };
      ball(h.p, 4, css("--warn"), info, true);
      line(h.p, add(h.p, scale(h.f, 0.6 / weight)), css("--warn"), 1.5, info);
    }
  }

  // Cylinders: barrel (cap side) from the base pin, rod from the rod pin.
  for (const c of f.cylinders) {
    const force = c.pA * c.boreArea - c.pB * c.rodArea;
    const info = {
      title: `${c.name}${c.forceControl ? " (force-controlled)" : ""}`,
      rows: [
        `stroke ${(1000 * c.x).toFixed(0)} / ${(1000 * c.stroke).toFixed(0)} mm, ${(c.v * 1000).toFixed(0)} mm/s`,
        `P_A ${(c.pA / 1e5).toFixed(0)} bar · P_B ${(c.pB / 1e5).toFixed(0)} bar (supply ${(supply / 1e5).toFixed(0)})`,
        `force ${force.toFixed(0)} N · valve ${c.cmd >= 0 ? "+" : ""}${c.cmd.toFixed(2)}`,
      ],
    };
    const d = sub(c.rod, c.base), L = norm(d), u = scale(d, 1 / (L || 1));
    if (c.cables && c.cables.length) {
      // Tendon drive: the body sits in the limb; its runs are coloured
      // by the tension the piston puts in them.
      line(c.base, c.rod, heat(c.pA / supply), 9, info);
      const runTension = (i) => Math.max(0, (i === 0 ? -1 : 1) * force);
      c.cables.forEach((run, i) => {
        const tn = runTension(i);
        poly(run, heat(tn / (supply * c.rodArea)), 2.5, { title: `${c.name} cable ${i === 0 ? "A" : "B"}`, rows: [`tension ≈ ${tn.toFixed(0)} N`] }, tn < 1 ? [4, 4] : null);
        run.forEach((p, k) => { if (k > 0) ball(p, 3, css("--panel"), info, true); });
      });
    } else {
      const barrelEnd = add(c.base, scale(u, Math.max(0.02, L - c.x - 0.015)));
      line(c.base, barrelEnd, heat(c.pA / supply), 9, info);
      line(barrelEnd, c.rod, heat(c.pB / supply), 4, info);
    }
    ball(c.base, 3.5, css("--panel"), info, true);
    ball(c.rod, 3.5, css("--panel"), info, true);
  }

  // Passive tendons.
  for (const t of f.tendons || []) {
    const info = { title: t.name, rows: [`tension ${t.tension.toFixed(0)} N`] };
    poly(t.path, t.tension > 1 ? heat(t.tension / (0.5 * weight)) : muted, 2.5, info, t.tension > 1 ? null : [3, 3]);
    t.path.forEach((p) => ball(p, 2.5, css("--panel"), info, true));
  }

  // Contacts: sole pads coloured by pressure, ground forces as arrows.
  for (const c of f.contacts || []) {
    const fn = Math.max(0, c.f[1]);
    const info = { title: c.pad ? "sole pad" : "contact point", rows: [`load ${fn.toFixed(0)} N · shear ${Math.hypot(c.f[0], c.f[2]).toFixed(0)} N`] };
    if (c.pad) ball(c.p, 4, fn > 0 ? heat(fn / (0.15 * weight)) : muted, info, true);
    else ball(c.p, 2, muted, info);
    if (fn > 0) line(c.p, add(c.p, scale(c.f, 0.6 / weight)), css("--press-a"), 1.5, info);
  }

  // Centre of gravity and its spot on the floor.
  const cogInfo = { title: "centre of gravity", rows: [`${f.mass.toFixed(1)} kg`] };
  ball(f.cog, 6, css("--warn"), cogInfo, true);
  line(f.cog, [f.cog[0], heightAt(terrain, f, f.cog[0], f.cog[2]), f.cog[2]], css("--warn"), 1, cogInfo, [3, 4]);

  prims.sort((a, b) => b.z - a.z);
  let hit = null, best = 9;
  for (const p of prims) {
    const w = (p.width || 1) * Math.min(2.5, 3 / p.z);
    ctx.setLineDash(p.dash || []);
    if (p.kind === "face") {
      ctx.fillStyle = p.color;
      ctx.globalAlpha = 0.92;
      ctx.beginPath();
      p.pp.forEach((q, i) => (i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y)));
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 0.5;
      ctx.strokeStyle = text;
      ctx.lineWidth = 0.6;
      ctx.stroke();
      ctx.globalAlpha = 1;
      if (mouse && p.info && ctx.isPointInPath(mouse[0], mouse[1]) && best >= 7) { best = 7; hit = p.info; }
    } else if (p.kind === "capsule") {
      ctx.strokeStyle = p.color;
      ctx.globalAlpha = 0.22;
      ctx.lineWidth = Math.max(2, p.px);
      ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(p.pa.x, p.pa.y); ctx.lineTo(p.pb.x, p.pb.y); ctx.stroke();
      ctx.globalAlpha = 1;
      if (mouse && p.info) {
        const d = segDist(mouse, p.pa, p.pb) - p.px / 2 + 8; // anything on top of it wins
        if (d < best) { best = d; hit = p.info; }
      }
    } else if (p.kind === "line") {
      ctx.strokeStyle = p.color;
      ctx.lineWidth = Math.max(1, w);
      ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(p.pa.x, p.pa.y); ctx.lineTo(p.pb.x, p.pb.y); ctx.stroke();
      if (mouse && p.info) {
        const d = segDist(mouse, p.pa, p.pb);
        if (d < best) { best = d; hit = p.info; }
      }
    } else {
      const r = Math.max(1.5, p.r * Math.min(2.5, 3 / p.z));
      ctx.fillStyle = p.color;
      ctx.beginPath(); ctx.arc(p.pa.x, p.pa.y, r, 0, Math.PI * 2); ctx.fill();
      if (p.outline) { ctx.strokeStyle = text; ctx.lineWidth = 1; ctx.stroke(); }
      if (mouse && p.info) {
        const d = Math.hypot(mouse[0] - p.pa.x, mouse[1] - p.pa.y) - r;
        if (d < best) { best = d; hit = p.info; }
      }
    }
  }
  ctx.setLineDash([]);

  legend(ctx, f, W, H, text, muted);
  if (hit && mouse) tooltip(ctx, hit, mouse, W, H, text);
}

function segDist(m, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y, l2 = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((m[0] - a.x) * dx + (m[1] - a.y) * dy) / l2));
  return Math.hypot(m[0] - (a.x + t * dx), m[1] - (a.y + t * dy));
}

function legend(ctx, f, W, H, text, muted) {
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillStyle = muted;
  const lines = [
    ...(terrain ? [`terrain "${terrain.name}" (seed ${terrain.seed}): ${(terrain.prims || []).length} solids${terrain.grid ? `, ${terrain.grid.nx}×${terrain.grid.nz} height grid` : ""}`] : []),
    `${f.body} · ${f.links.length} links · ${f.cylinders.length} cylinders · ${(f.tendons || []).length} tendons · ${(f.colliders || []).length} colliders · t ${f.t.toFixed(1)} s`,
    "drag: orbit · right/shift-drag: pan · wheel: zoom · F: follow · C: colliders · hover: details",
  ];
  lines.forEach((s, i) => ctx.fillText(s, 14, 20 + 16 * i));
  const x0 = 14, y0 = H - 34, w = 200;
  for (let i = 0; i < w; i++) {
    ctx.fillStyle = heat(i / w);
    ctx.fillRect(x0 + i, y0, 1, 10);
  }
  ctx.fillStyle = muted;
  ctx.fillText("0", x0, y0 + 24);
  ctx.fillText("pressure (of supply) · tension · pad load", x0 + 30, y0 + 24);
  ctx.fillText("high", x0 + w - 24, y0 + 24);
  ctx.fillStyle = text;
  ctx.fillText("barrel = cap-side pressure · rod = rod-side pressure · dashed = slack", x0 + w + 20, y0 + 9);
}

function tooltip(ctx, info, m, W, H, text) {
  ctx.font = "12px system-ui, sans-serif";
  const rows = [info.title, ...info.rows];
  const w = Math.max(...rows.map((r) => ctx.measureText(r).width)) + 16, h = 16 * rows.length + 10;
  const x = Math.min(m[0] + 14, W - w - 4), y = Math.min(m[1] + 14, H - h - 4);
  ctx.fillStyle = css("--panel");
  ctx.globalAlpha = 0.94;
  ctx.fillRect(x, y, w, h);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = css("--grid");
  ctx.strokeRect(x, y, w, h);
  rows.forEach((r, i) => {
    ctx.fillStyle = i ? css("--muted-text") : text;
    ctx.fillText(r, x + 8, y + 18 + 16 * i);
  });
}
