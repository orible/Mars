// Test-world terrain shared by the 2D and 3D views. The sim server sends
// it as a separate {"terrain": …} message whenever it changes: a height
// grid of the smooth floor (noise, slopes, ramps; absent when flat) and
// the solids (stair steps, platforms, objects) as boxes and cylinders.
//
// World frame: X forward, Y up, Z to the robot's right. A primitive's yaw
// turns its length axis from +x towards −z (like the robot's heading).

// fieldHeight is the smooth floor under (x, z): the grid, bilinear, and
// the frame's sine-wave bumps on top.
export function fieldHeight(t, f, x, z) {
  let h = 0;
  for (const w of (f && f.floor) || []) h += w.A * Math.cos(w.Kx * x + w.Kz * z + w.Phase);
  const g = t && t.grid;
  if (!g) return h;
  const u = Math.max(0, Math.min(g.nx - 1.001, (x - g.x0) / g.step));
  const v = Math.max(0, Math.min(g.nz - 1.001, (z - g.z0) / g.step));
  const i = Math.floor(u), j = Math.floor(v), a = u - i, b = v - j;
  const at = (ii, jj) => g.h[jj * g.nx + ii] / 1000;
  return h + (1 - b) * ((1 - a) * at(i, j) + a * at(i + 1, j)) + b * ((1 - a) * at(i, j + 1) + a * at(i + 1, j + 1));
}

// local turns (x, z) into a primitive's own (u along its length, w across).
function local(p, x, z) {
  const a = (p.yawDeg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  const dx = x - p.center[0], dz = z - p.center[2];
  return [dx * c - dz * s, dx * s + dz * c];
}

// topAt is the top of the solid standing at (x, z), or null.
export function topAt(p, x, z) {
  const top = p.center[1] + p.size[1] / 2;
  if (p.kind === "cylinder") return Math.hypot(x - p.center[0], z - p.center[2]) <= p.size[0] / 2 ? top : null;
  const [u, w] = local(p, x, z);
  return Math.abs(u) <= p.size[0] / 2 && Math.abs(w) <= p.size[2] / 2 ? top : null;
}

// heightAt is the top surface under (x, z): floor or solid.
export function heightAt(t, f, x, z) {
  let h = fieldHeight(t, f, x, z);
  for (const p of (t && t.prims) || []) {
    const top = topAt(p, x, z);
    if (top !== null && top > h) h = top;
  }
  return h;
}

// toWorld maps a primitive's local (u, y, w) to the world.
function toWorld(p, u, y, w) {
  const a = (p.yawDeg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  return [p.center[0] + u * c + w * s, p.center[1] + y, p.center[2] - u * s + w * c];
}

// faces returns a primitive's surface as polygons, each with its outward
// normal: a box's six faces, a cylinder as a 14-sided prism.
export function faces(p) {
  const hy = p.size[1] / 2;
  if (p.kind === "cylinder") {
    const r = p.size[0] / 2, n = 14, out = [], top = [];
    for (let k = 0; k < n; k++) {
      const a0 = (2 * Math.PI * k) / n, a1 = (2 * Math.PI * (k + 1)) / n;
      const P = (a, y) => [p.center[0] + r * Math.cos(a), p.center[1] + y, p.center[2] + r * Math.sin(a)];
      const am = (a0 + a1) / 2;
      out.push({ pts: [P(a0, -hy), P(a1, -hy), P(a1, hy), P(a0, hy)], n: [Math.cos(am), 0, Math.sin(am)] });
      top.push(P(a0, hy));
    }
    out.push({ pts: top, n: [0, 1, 0] });
    return out;
  }
  const hu = p.size[0] / 2, hw = p.size[2] / 2;
  const V = (su, sy, sw) => toWorld(p, su * hu, sy * hy, sw * hw);
  const N = (u, y, w) => { const o = toWorld(p, u, y, w), c = p.center; return [o[0] - c[0], o[1] - c[1], o[2] - c[2]]; };
  return [
    { pts: [V(-1, 1, -1), V(1, 1, -1), V(1, 1, 1), V(-1, 1, 1)], n: [0, 1, 0] },
    { pts: [V(1, -1, -1), V(1, -1, 1), V(1, 1, 1), V(1, 1, -1)], n: N(1, 0, 0) },
    { pts: [V(-1, -1, -1), V(-1, 1, -1), V(-1, 1, 1), V(-1, -1, 1)], n: N(-1, 0, 0) },
    { pts: [V(-1, -1, 1), V(1, -1, 1), V(1, 1, 1), V(-1, 1, 1)], n: N(0, 0, 1) },
    { pts: [V(-1, -1, -1), V(-1, 1, -1), V(1, 1, -1), V(1, -1, -1)], n: N(0, 0, -1) },
  ];
}

// footprint is a primitive's outline on the floor (x, z pairs).
export function footprint(p) {
  if (p.kind === "cylinder") {
    const r = p.size[0] / 2;
    return Array.from({ length: 16 }, (_, k) => [p.center[0] + r * Math.cos((k * Math.PI) / 8), p.center[2] + r * Math.sin((k * Math.PI) / 8)]);
  }
  const hu = p.size[0] / 2, hw = p.size[2] / 2;
  return [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => { const q = toWorld(p, a * hu, 0, b * hw); return [q[0], q[2]]; });
}

// colour gives each object a stable muted colour from its appearance
// seed; stairs and platforms are stone grey.
export function colour(p, shade) {
  const l = Math.round(38 + 30 * shade);
  if (p.role !== "object") return `hsl(30, 6%, ${l}%)`;
  return `hsl(${(p.seed * 137.5) % 360}, 30%, ${l}%)`;
}
