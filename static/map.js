// 3D map of ResNet18: one sheet per unit (stem + 8 residual blocks), one point per channel/kernel.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = s => document.querySelector(s);
const view = $("#mapView"), tip = $("#tip");
const SHEET = 24, GAP = 34; // every sheet spans SHEET world units; sheets are GAP apart along x
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

if (!window.appReady) await new Promise(r => window.addEventListener("app-ready", r, { once: true }));
const MAP = await (await fetch("/api/map")).json();

// ---------- scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1115);
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 2000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const material = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false,
  vertexShader: `attribute float size; attribute vec3 color; varying vec3 vColor;
    void main() { vColor = color; vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = size * (760.0 / -mv.z); gl_Position = projectionMatrix * mv; }`,
  fragmentShader: `varying vec3 vColor;
    void main() { vec2 d = gl_PointCoord - 0.5; float r = length(d); if (r > 0.5) discard;
      gl_FragColor = vec4(vColor, smoothstep(0.5, 0.38, r)); }`,
});

const sheets = MAP.layers.map((L, k) => {
  const sp = SHEET / L.side, n = L.C, x0 = k * GAP;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * n), 3));
  geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(3 * n), 3));
  geo.setAttribute("size", new THREE.BufferAttribute(new Float32Array(n).fill(sp * 0.8), 1));
  const pts = new THREE.Points(geo, material);
  pts.userData = { layer: L.name, k };
  scene.add(pts);
  // faint frame so empty sheets still read as planes
  const frame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(SHEET + 2, SHEET + 2)),
    new THREE.LineBasicMaterial({ color: 0x2a3040 }));
  frame.rotation.y = Math.PI / 2; frame.position.x = x0; scene.add(frame);
  pts.userData.frame = frame;
  const label = document.createElement("div");
  label.className = "mlabel"; label.textContent = `${L.name} · ${L.C}`;
  label.onclick = () => faceOn(k);
  view.appendChild(label);
  return { L, pts, geo, sp, x0, label, target: null };
});
const center = new THREE.Vector3((sheets.length - 1) * GAP / 2, 0, 0);

function cellPos(s, layout) {
  const out = new Float32Array(3 * s.L.C), half = (s.L.side - 1) / 2;
  s.L[layout].forEach(([gx, gy], c) => out.set([s.x0, (half - gy) * s.sp, (gx - half) * s.sp], 3 * c));
  return out;
}
function setLayout(layout, animate) {
  for (const s of sheets) {
    s.target = cellPos(s, layout);
    if (!animate) s.geo.attributes.position.array.set(s.target);
    s.geo.attributes.position.needsUpdate = true;
  }
  raycastDirty = true;
}

// ---------- camera moves ----------
let fly = null;
function flyTo(pos, look) {
  fly = { p0: camera.position.clone(), l0: controls.target.clone(), p1: pos, l1: look, t: 0 };
}
function solo(k) { // face-on view hides the other sheets, which would otherwise sit in front of the camera
  sheets.forEach((s, i) => { const v = k === null || i === k; s.pts.visible = s.pts.userData.frame.visible = v; s.hidden = !v; });
}
const OVERVIEW_POS = new THREE.Vector3(-215, 75, 250), OVERVIEW_LOOK = new THREE.Vector3(-26, -6, 0);
function overview() {
  sheets.forEach(s => s.label.classList.remove("on")); solo(null);
  flyTo(center.clone().add(OVERVIEW_POS), center.clone().add(OVERVIEW_LOOK));
}
function faceOn(k) {
  sheets.forEach((s, i) => s.label.classList.toggle("on", i === k)); solo(k);
  const x = sheets[k].x0;
  flyTo(new THREE.Vector3(x - 46, 0, 0.01), new THREE.Vector3(x, 0, 0));
}
$("#overview").onclick = overview;

// ---------- coloring ----------
const hsl = (h, s, l) => new THREE.Color().setHSL(h, s, l);
const heat = t => { t = Math.max(0, Math.min(1, t)); return new THREE.Color().setRGB(0.12 + 0.88 * t, 0.14 + 0.7 * t * t, 0.2 + 0.25 * (1 - t) * t); };
const coarseColor = i => hsl((i * 0.618034) % 1, 0.62, 0.58);
let conceptScores = {}, conceptKey = null, acts = null, actsIdx = null;

async function loadConcept() {
  if (!S.concept || S.concept === conceptKey) return;
  conceptKey = S.concept;
  const res = await Promise.all(MAP.layers.map(L => fetch(`/api/neurons?layer=${L.name}&concept=${S.concept}`).then(r => r.json())));
  conceptScores = {};
  res.forEach((list, i) => { const a = new Float32Array(MAP.layers[i].C); list.forEach(n => a[n.channel] = n.score); conceptScores[MAP.layers[i].name] = a; });
}
async function loadActs() {
  if (actsIdx === S.img) return;
  actsIdx = S.img;
  acts = await (await fetch(`/api/acts/${S.img}`)).json();
}

async function recolor() {
  const mode = $("#colorMode").value;
  if (mode === "concept") await loadConcept();
  if (mode === "fires") await loadActs();
  const abl = new Set(S.abl.map(a => `${a.layer}:${a.channel}`));
  for (const s of sheets) {
    const col = s.geo.attributes.color.array, size = s.geo.attributes.size.array, name = s.L.name;
    s.L.units.forEach((u, c) => {
      let color;
      if (mode === "coarse") color = coarseColor(u.coarse).multiplyScalar(0.35 + 0.65 * Math.min(1, u.csel / 1.5));
      else if (mode === "concept") color = conceptScores[name] ? heat(conceptScores[name][c] / 2.5) : heat(0);
      else if (mode === "fires") color = acts ? heat(acts[name][c]) : heat(0);
      else color = abl.has(`${name}:${c}`) ? new THREE.Color(0xff7a6a) : new THREE.Color(0x343b4c);
      const sel = S.sel && S.sel.layer === name && S.sel.channel === c;
      if (sel) color = new THREE.Color(0xffffff);
      if (mode !== "ablation" && abl.has(`${name}:${c}`)) color = new THREE.Color(0xff7a6a);
      col.set([color.r, color.g, color.b], 3 * c);
      size[c] = s.sp * (sel ? 1.6 : 0.8);
    });
    s.geo.attributes.color.needsUpdate = s.geo.attributes.size.needsUpdate = true;
  }
  legend(mode);
}
function legend(mode) {
  const L = $("#mapLegend");
  if (mode === "coarse") L.innerHTML = MAP.coarse.map((n, i) => `<span style="white-space:nowrap;margin-right:8px"><span class="sw" style="background:#${coarseColor(i).getHexString()}"></span>${esc(n.replaceAll("_", " "))}</span>`).join(" ") + "<br>brightness = selectivity (" + tex("d'") + ")";
  else if (mode === "concept") L.innerHTML = S.concept ? `brightness = ${tex("d'")} for “${esc($("#concept").selectedOptions[0].textContent)}” (pick concept on the left)` : "pick a concept on the left";
  else if (mode === "fires") L.textContent = `brightness = channel max on lab image #${S.img}, relative to its 99th-pct dataset activation`;
  else L.textContent = `${S.abl.length} channel(s) in ablation set (red)`;
}
let pending = false;
window.mapRefresh = () => { if (!pending) { pending = true; requestAnimationFrame(async () => { pending = false; await recolor(); }); } };
$("#colorMode").onchange = window.mapRefresh;
$("#layoutMode").onchange = e => setLayout(e.target.value, true);

// ---------- hover + click ----------
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
let hover = null, raycastDirty = false, downAt = null;
function pick(ev) {
  const r = renderer.domElement.getBoundingClientRect();
  mouse.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  let best = null;
  for (const s of sheets) {
    if (s.hidden) continue;
    ray.params.Points.threshold = s.sp * 0.5;
    for (const h of ray.intersectObject(s.pts)) if (!best || h.distanceToRay / s.sp < best.d) best = { d: h.distanceToRay / s.sp, s, c: h.index, dist: h.distance };
  }
  return best;
}
function showTip(ev, s, c) {
  const u = s.L.units[c], r = view.getBoundingClientRect();
  tip.innerHTML = `<div class="fv" style="${fvStyle(s.L.name, c, 96)}"></div>
    <b>${esc(s.L.name)} · #${c}</b><div class="hint">${esc(MAP.coarse[u.coarse].replaceAll("_", " "))} (${tex(`d' = ${u.csel.toFixed(2)}`)})</div>
    <div style="margin-top:4px">${u.fine.map(esc).join(", ")}</div><div class="hint">best class ${tex(`d' = ${u.fsel.toFixed(2)}`)}</div>
    <div class="tops" style="clear:both">${u.top.map(i => `<img src="/api/img/${i}?s=2">`).join("")}</div>`;
  tip.style.display = "block";
  const x = ev.clientX - r.left + 14, y = ev.clientY - r.top + 14;
  tip.style.left = Math.min(x, r.width - 250) + "px"; tip.style.top = Math.min(y, r.height - 200) + "px";
}
renderer.domElement.addEventListener("pointermove", ev => {
  const p = pick(ev);
  if (p) { showTip(ev, p.s, p.c); renderer.domElement.style.cursor = "pointer"; hover = p; }
  else { tip.style.display = "none"; renderer.domElement.style.cursor = ""; hover = null; }
});
renderer.domElement.addEventListener("pointerleave", () => { tip.style.display = "none"; });
renderer.domElement.addEventListener("pointerdown", ev => { downAt = [ev.clientX, ev.clientY]; });
renderer.domElement.addEventListener("pointerup", ev => {
  if (!downAt || Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) > 4) return; // was a drag
  const p = pick(ev);
  if (p) { tip.style.display = "none"; openNeuron(p.s.L.name, p.c); }
});

// ---------- loop ----------
function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
}
window.mapResize = resize;
// screen position of a channel's point (page coords) - used by scripted demos/tests
window.mapScreenPos = (layer, c) => {
  const s = sheets.find(s => s.L.name === layer), a = s.geo.attributes.position.array, r = renderer.domElement.getBoundingClientRect();
  const p = new THREE.Vector3(a[3 * c], a[3 * c + 1], a[3 * c + 2]).project(camera);
  return { x: r.left + (p.x + 1) / 2 * r.width, y: r.top + (1 - p.y) / 2 * r.height };
};
window.mapFaceOn = name => faceOn(sheets.findIndex(s => s.L.name === name));
new ResizeObserver(resize).observe(view);
const tmp = new THREE.Vector3();
function frame() {
  requestAnimationFrame(frame);
  if (view.hidden) return;
  if (fly) {
    fly.t = Math.min(1, fly.t + 0.04);
    const e = 1 - Math.pow(1 - fly.t, 3);
    camera.position.lerpVectors(fly.p0, fly.p1, e); controls.target.lerpVectors(fly.l0, fly.l1, e);
    if (fly.t >= 1) fly = null;
  }
  for (const s of sheets) {
    if (!s.target) continue;
    const a = s.geo.attributes.position.array; let moving = false;
    for (let i = 0; i < a.length; i++) { const d = s.target[i] - a[i]; if (Math.abs(d) > 1e-3) { a[i] += d * 0.15; moving = true; } else a[i] = s.target[i]; }
    s.geo.attributes.position.needsUpdate = true;
    if (!moving) { s.target = null; s.geo.computeBoundingSphere(); }
  }
  controls.update();
  renderer.render(scene, camera);
  const w = view.clientWidth, h = view.clientHeight;
  for (const s of sheets) {
    tmp.set(s.x0, SHEET / 2 + 2.5, 0).project(camera);
    const vis = tmp.z < 1 && !s.hidden;
    s.label.style.display = vis ? "" : "none";
    s.label.style.left = ((tmp.x + 1) / 2 * w) + "px"; s.label.style.top = ((1 - tmp.y) / 2 * h) + "px";
  }
}

setLayout("grid", false);
camera.position.copy(center.clone().add(OVERVIEW_POS));
controls.target.copy(center.clone().add(OVERVIEW_LOOK));
if (location.hash.startsWith("#face=")) { faceOn(+location.hash.slice(6)); fly.t = 1; } // deep link / screenshot hook
resize();
await recolor();
frame();
