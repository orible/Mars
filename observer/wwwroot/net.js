// Network view: the set policy's activations, live, drawn on a 2D canvas
// (no libraries, works offline).
//
// The policy is a Deep Sets network, so the view is laid out as one row
// per token rather than as a dense-net diagram: every row in a lane goes
// through the same shared MLP, and the lane's bottom row is the mean they
// pool into. Lanes, top to bottom:
//   body parts → Part net → mean = body-shape context
//   sensors    → Sensor net → mean = what the body feels
//   cylinders  → φ → mean = body context, then ρ per cylinder → command
// Columns: normalised input · tanh embedding · share of the pooled mean ·
// ρ's hidden layer · command μ with its ±σ exploration band.
//
// "Share" is how much of the pooled embedding a token supplies: its
// embedding's projection on the mean, h_i·m / (n |m|²). Shares sum to 1;
// the dominant tokens are listed in each lane's header.
//
// Hover any cell for its numbers; hovering an input feature or an
// embedding or hidden unit outlines that same unit in every token of the
// lane. The page asks the sim server for activations with ?net=1, so
// other viewers' streams stay lean.

let socket = null;
let latest = null;
let rafId = 0;
let statusEl = null;
let mouse = null;
let hits = [];

const W = 1360; // logical width; the canvas scales to the page
const ROW = 13, CELL = 11;
const X = { name: 8, input: 156, emb: 340, share: 672, rho: 760, cmd: 1214 };
const WIDTH = { input: 168, emb: 320, share: 72, rho: 438, cmd: 136 };

export function connect(url, canvas, status) {
  disconnect();
  statusEl = status;
  setStatus("connecting");
  if (!/[?&]net=/.test(url)) url += (url.includes("?") ? "&" : "?") + "net=1";
  socket = new WebSocket(url);
  socket.onopen = () => setStatus("connected");
  socket.onclose = () => setStatus("disconnected");
  socket.onerror = () => setStatus("error");
  socket.onmessage = (e) => { latest = JSON.parse(e.data); };
  bindInput(canvas);
  const loop = () => {
    draw(canvas, latest);
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
  canvas.addEventListener("pointermove", (e) => {
    const r = canvas.getBoundingClientRect();
    mouse = [(e.clientX - r.left) * W / r.width, (e.clientY - r.top) * W / r.width];
  });
  canvas.addEventListener("pointerleave", () => { mouse = null; });
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const dark = () => matchMedia("(prefers-color-scheme: dark)").matches;
const fmt = (v, d = 3) => (v == null ? "–" : (v >= 0 ? " " : "") + v.toFixed(d));
const pct = (v) => (v * 100).toFixed(1) + "%";

// Signed value to colour: blue (negative) through the panel colour to
// orange (positive), |v| = scale at full strength.
function diverge(v, scale) {
  const t = Math.max(-1, Math.min(1, v / scale));
  const a = Math.abs(t);
  const hue = t < 0 ? 212 : 22;
  return dark()
    ? `hsl(${hue}, ${Math.round(10 + 70 * a)}%, ${Math.round(17 + 40 * a)}%)`
    : `hsl(${hue}, ${Math.round(10 + 75 * a)}%, ${Math.round(97 - 46 * a)}%)`;
}

// share[i] = h_i·m / (n |m|²): each token's part of the pooled mean.
function shares(emb, pool) {
  const n = emb.length;
  let mm = 0;
  for (const v of pool) mm += v * v;
  return emb.map((h) => {
    let d = 0;
    for (let j = 0; j < h.length; j++) d += h[j] * pool[j];
    return mm > 0 ? d / (n * mm) : 0;
  });
}

function lanes(net) {
  const out = [];
  out.push({ key: "parts", title: "Body parts", net: "Part MLP", pooled: "body shape",
    names: net.labels.parts, feats: net.features.parts, input: net.parts, emb: net.partEmb || [], pool: net.partPool });
  out.push({ key: "sensors", title: "Sensors", net: "Sensor MLP", pooled: "feel",
    names: net.labels.sensors, feats: net.features.sensors, input: net.sensors, emb: net.sensorEmb || [], pool: net.sensorPool });
  out.push({ key: "tokens", title: "Cylinders", net: "φ", pooled: "body context",
    names: net.labels.tokens, feats: net.features.tokens, input: net.tokens, emb: net.tokenEmb || [], pool: net.tokenPool, rho: true });
  for (const l of out) l.share = shares(l.emb, l.pool || []);
  return out;
}

function draw(canvas, f) {
  const net = f && f.net;
  const ls = net ? lanes(net) : [];
  let H = 84;
  for (const l of ls) H += 40 + ROW * (l.input.length + 1) + 18;
  const attn = net && net.attn && net.attn[0];
  if (attn) H += 60 + ROW * attn.length;
  H = Math.max(H, 220);
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || W;
  const pxW = Math.round(cssW * dpr), pxH = Math.round(H * cssW / W * dpr);
  if (canvas.width !== pxW || canvas.height !== pxH) {
    canvas.width = pxW; canvas.height = pxH;
    canvas.style.height = (H * cssW / W) + "px";
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(pxW / W, 0, 0, pxW / W, 0, 0);
  const text = css("--text"), muted = css("--muted-text"), accent = css("--accent"), grid = css("--grid");
  ctx.fillStyle = css("--panel");
  ctx.fillRect(0, 0, W, H);
  ctx.font = "12px system-ui, sans-serif";
  ctx.textBaseline = "middle";
  const hov = hovered(); // against the previous frame's layout
  hits = [];

  if (!f) { label(ctx, "waiting for the sim server…", 16, 30, muted); return; }
  if (!net) {
    label(ctx, "No activations yet: the policy has not run (the sim server may be running with -hold, or is an older build without ?net=1).", 16, 30, muted);
    return;
  }

  // Header and the global vector every set network also reads.
  const std = Math.exp(net.logStd);
  ctx.font = "600 13px system-ui, sans-serif";
  label(ctx, `Policy: ${net.policy}`, 8, 14, text);
  ctx.font = "12px system-ui, sans-serif";
  label(ctx, `t = ${f.t.toFixed(2)} s · ${f.body} · exploration σ = ${std.toFixed(3)} (log σ ${net.logStd.toFixed(2)}) · ` +
    `blue negative, orange positive · inputs normalised (±3 full colour), embeddings and hidden units tanh (±1)`, 8, 32, muted);
  label(ctx, "global (normalised)", X.name, 56, text);
  const gw = 16;
  cells(ctx, net.global, X.input, 50, gw, CELL, 3);
  hits.push({ x: X.input, y: 50, w: gw * net.global.length, h: CELL, cw: gw, tip: (j) => [
    "global input, normalised", `${net.features.global[j] || "#" + j}: ${fmt(net.global[j])}`,
    "read by every set network and by ρ"] });
  label(ctx, "→ read by every network below, with the three pooled contexts", X.input + gw * net.global.length + 12, 56, muted);

  let y = 84;
  for (const l of ls) y = drawLane(ctx, l, y, hov, { text, muted, accent, grid, std, net });
  if (attn) drawAttn(ctx, net, attn, y, hov, { text, muted, accent, grid });
  drawTooltip(ctx, H);
}

function drawLane(ctx, l, y, hov, c) {
  const n = l.input.length;
  // Lane header: what the rows are, the network they share, the pool, and
  // the tokens that dominate it.
  ctx.font = "600 13px system-ui, sans-serif";
  label(ctx, `${l.title} · ${n} tokens`, X.name, y + 8, c.text);
  ctx.font = "12px system-ui, sans-serif";
  const order = l.share.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]);
  const top = order.slice(0, 3).filter((p) => p[0] > 0);
  const dom = top.map(([s, i]) => `${l.names[i] || "#" + i} ${pct(s)}`).join(", ");
  label(ctx, `each row → shared ${l.net} → mean = ${l.pooled}${dom ? " · dominated by " + dom : ""}`, X.input, y + 8, c.muted);
  y += 22;
  ctx.fillStyle = c.muted;
  ctx.font = "11px system-ui, sans-serif";
  label(ctx, "input", X.input, y + 4, c.muted);
  label(ctx, `${l.net} embedding`, X.emb, y + 4, c.muted);
  label(ctx, "share of mean", X.share, y + 4, c.muted);
  if (l.rho) {
    label(ctx, "ρ hidden layer", X.rho, y + 4, c.muted);
    label(ctx, "command μ ± σ", X.cmd, y + 4, c.muted);
  }
  ctx.font = "12px system-ui, sans-serif";
  y += 14;
  if (n === 0) { label(ctx, "none on this body", X.name, y + 6, c.muted); return y + 32; }
  const dims = l.input[0].length, iw = WIDTH.input / dims;
  const E = (l.pool || []).length, ew = WIDTH.emb / Math.max(1, E);
  const topSet = new Set(top.map((p) => p[1]));
  const y0 = y;
  for (let i = 0; i < n; i++, y += ROW) {
    const name = l.names[i] || "#" + i;
    ctx.font = topSet.has(i) ? "600 12px system-ui, sans-serif" : "12px system-ui, sans-serif";
    label(ctx, name.length > 22 ? name.slice(0, 21) + "…" : name, X.name, y + CELL / 2, topSet.has(i) ? c.text : c.muted);
    cells(ctx, l.input[i], X.input, y, iw, CELL, 3);
    hits.push({ lane: l.key, stage: "input", x: X.input, y, w: WIDTH.input, h: CELL, cw: iw, tip: (j) => [
      `${name} · input`, `${l.feats[j] || "#" + j}: ${fmt(l.input[i][j])} (normalised)`] });
    if (l.emb[i]) {
      cells(ctx, l.emb[i], X.emb, y, ew, CELL, 1);
      hits.push({ lane: l.key, stage: "emb", x: X.emb, y, w: WIDTH.emb, h: CELL, cw: ew, tip: (j) => [
        `${name} · ${l.net} embedding unit ${j}`, `value ${fmt(l.emb[i][j])}`,
        `adds ${fmt(l.emb[i][j] / n)} to the mean (${fmt(l.pool[j])})`] });
    }
    shareBar(ctx, l.share[i], n, X.share, y, c);
    hits.push({ x: X.share, y, w: WIDTH.share, h: CELL, tip: () => [
      `${name} · share of the pooled ${l.pooled}`,
      `${pct(l.share[i])} = ${(l.share[i] * n).toFixed(2)}× an even share (rank ${order.findIndex((p) => p[1] === i) + 1} of ${n})`,
      "projection of its embedding on the mean, h·m / (n|m|²)"] });
    if (l.rho) drawRho(ctx, c.net, i, name, y, c);
  }
  // The pooled mean: what the rest of the network sees of this set.
  ctx.strokeStyle = c.grid;
  ctx.beginPath(); ctx.moveTo(X.name, y + 1.5); ctx.lineTo(X.emb + WIDTH.emb, y + 1.5); ctx.stroke();
  y += 4;
  ctx.font = "600 12px system-ui, sans-serif";
  label(ctx, `mean = ${l.pooled}`, X.name, y + CELL / 2, c.accent);
  ctx.font = "12px system-ui, sans-serif";
  if (E) {
    cells(ctx, l.pool, X.emb, y, ew, CELL, 1);
    ctx.strokeStyle = c.accent; ctx.lineWidth = 1.5;
    ctx.strokeRect(X.emb - 1, y - 1, WIDTH.emb + 2, CELL + 2);
    ctx.lineWidth = 1;
    hits.push({ lane: l.key, stage: "emb", x: X.emb, y, w: WIDTH.emb, h: CELL, cw: ew, tip: (j) => [
      `mean ${l.net} embedding (${l.pooled}) unit ${j}`, `value ${fmt(l.pool[j])}`,
      l.rho ? "fed to ρ for every cylinder" : "appended to the global vector for φ and ρ"] });
  }
  // Outline the hovered unit across every row of this lane.
  if (hov && hov.lane === l.key) {
    const x = (hov.stage === "input" ? X.input : hov.stage === "emb" ? X.emb : X.rho) + hov.j * hov.cw;
    ctx.strokeStyle = c.text; ctx.lineWidth = 1;
    ctx.strokeRect(x - 0.5, y0 - 1.5, hov.cw + 1, (hov.stage === "rho" ? n * ROW : y - y0 + CELL) + 2);
  }
  return y + CELL + 22;
}

// Attention between cylinders (policies trained with -attn): row i is
// cylinder i listening, column j the cylinder it hears, averaged over
// heads; each row sums to 1, so an even spread is 1/n. The bar on the
// right is how much attention changes that cylinder's embedding.
function drawAttn(ctx, net, A, y, hov, c) {
  const n = A.length, names = net.labels.tokens, cw = 13;
  let top = 0;
  for (const row of A) for (const v of row) top = Math.max(top, v);
  ctx.font = "600 13px system-ui, sans-serif";
  label(ctx, `Attention · ${n} cylinders`, X.name, y + 8, c.text);
  ctx.font = "12px system-ui, sans-serif";
  label(ctx, `row listens to column (mean over heads) · orange above an even ${pct(1 / n)}, blue below; strongest here ${pct(top)} · bar: how much it changes the row's embedding`, X.input, y + 8, c.muted);
  y += 30;
  const x0 = X.input, gx = x0 + n * cw + 16;
  for (let i = 0; i < n; i++, y += ROW) {
    const name = names[i] || "#" + i;
    label(ctx, name.length > 22 ? name.slice(0, 21) + "…" : name, X.name, y + CELL / 2, c.muted);
    for (let j = 0; j < n; j++) {
      ctx.fillStyle = diverge(A[i][j] * n - 1, 1); // orange: above an even share, blue: below; 2× even is full
      ctx.fillRect(x0 + j * cw, y, cw - 1, CELL);
    }
    hits.push({ lane: "attn", x: x0, y, w: n * cw, h: CELL, cw, tip: (j) => [
      `${name} listens to ${names[j] || "#" + j}`, `weight ${pct(A[i][j])} (${(A[i][j] * n).toFixed(2)}× even)`] });
    const g = (net.attnGain || [])[i] || 0;
    ctx.fillStyle = c.accent;
    ctx.fillRect(gx, y + 2, Math.min(1, g) * 120, CELL - 4);
    ctx.fillStyle = c.grid;
    ctx.fillRect(gx, y + CELL / 2 - 0.5, 120, 1);
    hits.push({ x: gx, y, w: 120, h: CELL, tip: () => [
      `${name} · attention's change to its embedding`, `${pct(g)} of the embedding's size`, "0% = attention not used yet (it starts switched off)"] });
  }
  if (hov && hov.lane === "attn") {
    ctx.strokeStyle = c.text;
    ctx.strokeRect(x0 + hov.j * cw - 0.5, y - n * ROW - 1.5, cw + 1, n * ROW + 2);
  }
}

function drawRho(ctx, net, i, name, y, c) {
  const layers = net.rhoHid[i] || [];
  const h = layers[layers.length - 1] || [];
  const rw = WIDTH.rho / Math.max(1, h.length);
  cells(ctx, h, X.rho, y, rw, CELL, 1);
  hits.push({ lane: "tokens", stage: "rho", x: X.rho, y, w: WIDTH.rho, h: CELL, cw: rw, tip: (j) => [
    `${name} · ρ hidden unit ${j}`, `value ${fmt(h[j])}`] });
  // Command: μ as a bar from 0 on a −1…1 axis, with the ±σ exploration band.
  const x0 = X.cmd, w = WIDTH.cmd, mid = x0 + w / 2, sx = w / 2;
  const mu = net.mu[i], std = c.std;
  ctx.fillStyle = c.grid;
  ctx.fillRect(x0, y + CELL / 2 - 0.5, w, 1);
  ctx.globalAlpha = 0.25;
  ctx.fillStyle = c.accent;
  const lo = Math.max(-1, mu - std), hi = Math.min(1, mu + std);
  ctx.fillRect(mid + lo * sx, y + 1, (hi - lo) * sx, CELL - 2);
  ctx.globalAlpha = 1;
  ctx.fillStyle = diverge(mu, 1);
  ctx.fillRect(Math.min(mid, mid + mu * sx), y + 2, Math.abs(mu * sx), CELL - 4);
  ctx.fillStyle = c.text;
  ctx.fillRect(mid + mu * sx - 1, y, 2, CELL);
  ctx.fillStyle = c.muted;
  ctx.fillRect(mid - 0.5, y, 1, CELL);
  hits.push({ x: x0, y, w, h: CELL, tip: () => [
    `${name} · valve command`, `μ = ${fmt(mu)} (tanh of ${fmt(net.pre[i])})`,
    `σ = ${std.toFixed(3)} exploration (training only)`, `range μ±σ: ${fmt(lo, 2)} … ${fmt(hi, 2)}`] });
}

// Share bar: three times the even share (1/n) fills it; the tick marks the
// even share, so tokens left of it supply less than their part.
function shareBar(ctx, s, n, x, y, c) {
  const w = WIDTH.share, mid = x + w * 0.2, full = w * 0.8;
  ctx.fillStyle = c.grid;
  ctx.fillRect(x, y + CELL / 2 - 0.5, w, 1);
  const len = Math.max(-0.25, Math.min(1, s * n / 3)) * full;
  ctx.fillStyle = s >= 0 ? c.accent : css("--warn");
  ctx.fillRect(Math.min(mid, mid + len), y + 2, Math.abs(len), CELL - 4);
  ctx.fillStyle = c.muted;
  ctx.fillRect(mid + full / 3 - 0.5, y, 1, CELL);
}

function cells(ctx, v, x, y, w, h, scale) {
  for (let j = 0; j < v.length; j++) {
    ctx.fillStyle = diverge(v[j], scale);
    ctx.fillRect(x + j * w, y, Math.max(1, w - (w > 3 ? 1 : 0)), h);
  }
}

function label(ctx, s, x, y, color) {
  ctx.fillStyle = color;
  ctx.fillText(s, x, y);
}

function hovered() {
  if (!mouse) return null;
  for (const r of hits) {
    if (mouse[0] >= r.x && mouse[0] < r.x + r.w && mouse[1] >= r.y && mouse[1] < r.y + r.h) {
      const j = r.cw ? Math.floor((mouse[0] - r.x) / r.cw) : 0;
      return { ...r, j };
    }
  }
  return null;
}

function drawTooltip(ctx, H) {
  const h = hovered();
  if (!h) return;
  const lines = h.tip(h.j);
  ctx.font = "12px ui-monospace, Consolas, monospace";
  const w = Math.max(...lines.map((s) => ctx.measureText(s).width)) + 16;
  const th = lines.length * 16 + 8;
  let x = mouse[0] + 14, y = mouse[1] + 14;
  if (x + w > W - 4) x = mouse[0] - w - 10;
  if (y + th > H - 4) y = mouse[1] - th - 10;
  ctx.fillStyle = css("--bg");
  ctx.strokeStyle = css("--grid");
  ctx.fillRect(x, y, w, th);
  ctx.strokeRect(x + 0.5, y + 0.5, w - 1, th - 1);
  lines.forEach((s, k) => label(ctx, s, x + 8, y + 12 + k * 16, k ? css("--muted-text") : css("--text")));
}
