// Connects the browser straight to the Go simulator's WebSocket and draws
// frames on a canvas. Blazor only receives a throttled status summary, so
// the 30 Hz frame stream never crosses the SignalR circuit.
//
// World frame: X forward, Y up, Z to the robot's right. The canvas shows a
// side view (looking at the robot's left side) and a front view.

let socket = null;
let latest = null;
let rafId = 0;
let lastStatus = 0;
let dotnetRef = null;

export function connect(url, canvas, dotnet) {
  disconnect();
  dotnetRef = dotnet;
  report("connecting");
  socket = new WebSocket(url);
  socket.onopen = () => report("connected");
  socket.onclose = () => report("disconnected");
  socket.onerror = () => report("error");
  socket.onmessage = (e) => {
    latest = JSON.parse(e.data);
    const now = performance.now();
    if (now - lastStatus > 500) {
      lastStatus = now;
      report("connected");
    }
  };
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

function report(state) {
  if (!dotnetRef) return;
  const f = latest;
  dotnetRef.invokeMethodAsync(
    "OnStatus",
    state,
    f ? f.body : "–",
    f ? f.bodyRev : 0,
    f ? f.cylinders.length : 0,
    f ? f.mass : 0,
    f ? f.t : 0
  );
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// Orthographic projections: [screen-right, up] for a world point.
const views = [
  { title: "side (left)", cx: 150, project: (p) => [p[0], p[1]] },
  { title: "front", cx: 410, project: (p) => [-p[2], p[1]] },
];
const viewTop = 90, scale = 520;

function draw(canvas, f) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = css("--panel");
  ctx.fillRect(0, 0, W, H);
  ctx.font = "12px system-ui, sans-serif";

  for (const v of views) {
    const at = (p) => {
      const [x, y] = v.project(p);
      return [v.cx + x * scale, viewTop - y * scale];
    };
    ctx.fillStyle = css("--muted-text");
    ctx.fillText(v.title, v.cx - 110, 24);

    // Floor-free test stand: draw the fixed pelvis mount.
    ctx.strokeStyle = css("--grid");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(v.cx - 110, viewTop); ctx.lineTo(v.cx + 110, viewTop);
    ctx.stroke();

    // Bones: parent joint -> child joint; leaf links extend through their CoM.
    const hasChild = new Set(f.links.map((l) => l.parent));
    ctx.strokeStyle = css("--text");
    ctx.lineWidth = 3;
    f.links.forEach((l, i) => {
      if (l.parent >= 0) line(ctx, at(f.links[l.parent].origin), at(l.origin));
      if (!hasChild.has(i) && i > 0) {
        const tip = [0, 1, 2].map((k) => l.origin[k] + 2 * (l.com[k] - l.origin[k]));
        line(ctx, at(l.origin), at(tip));
      }
    });
    ctx.fillStyle = css("--text");
    for (const l of f.links) dot(ctx, at(l.origin), 3);

    // Cylinders: barrel-to-rod line coloured by valve command.
    ctx.lineWidth = 4;
    for (const c of f.cylinders) {
      ctx.strokeStyle = c.cmd >= 0 ? css("--accent") : css("--warn");
      ctx.globalAlpha = 0.35 + 0.65 * Math.min(1, Math.abs(c.cmd));
      line(ctx, at(c.base), at(c.rod));
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = css("--muted-text");
    for (const c of f.cylinders) { dot(ctx, at(c.base), 2.5); dot(ctx, at(c.rod), 2.5); }

    // Centre of gravity.
    const [gx, gy] = at(f.cog);
    ctx.strokeStyle = css("--warn");
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(gx, gy, 8, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(gx - 12, gy); ctx.lineTo(gx + 12, gy); ctx.moveTo(gx, gy - 12); ctx.lineTo(gx, gy + 12); ctx.stroke();
  }

  // Right: one row per cylinder — stroke, chamber pressures, valve command.
  const left = 590, rowH = Math.min(44, (H - 60) / Math.max(1, f.cylinders.length));
  ctx.fillStyle = css("--muted-text");
  ctx.fillText("stroke", left + 90, 24);
  ctx.fillText("P_A / P_B (of supply)", left + 230, 24);
  ctx.fillText("valve cmd", left + 400, 24);
  f.cylinders.forEach((c, i) => {
    const y = 40 + i * rowH;
    ctx.fillStyle = css("--text");
    ctx.fillText(c.name, left, y + 12);
    bar(ctx, left + 90, y, 120, 14, c.x / c.stroke, css("--accent"));
    bar(ctx, left + 230, y, 150, 6, c.pA / f.supply, css("--press-a"));
    bar(ctx, left + 230, y + 8, 150, 6, c.pB / f.supply, css("--press-b"));
    centred(ctx, left + 400, y, 100, 14, c.cmd);
  });
}

function line(ctx, a, b) {
  ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
}

function dot(ctx, p, r) {
  ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, Math.PI * 2); ctx.fill();
}

function bar(ctx, x, y, w, h, frac, color) {
  ctx.fillStyle = css("--muted");
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = color;
  ctx.fillRect(x, y, w * Math.max(0, Math.min(1.2, frac)), h);
}

function centred(ctx, x, y, w, h, v) {
  ctx.fillStyle = css("--muted");
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = v >= 0 ? css("--accent") : css("--warn");
  const mid = x + w / 2, len = (w / 2) * Math.max(-1, Math.min(1, v));
  ctx.fillRect(Math.min(mid, mid + len), y, Math.abs(len), h);
}
