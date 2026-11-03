'use strict';
/* BRIGHT viewer front-end.
 * One <canvas> draws two layouts of the same tiles:
 *   mosaic - tiles placed back into their scenes using the GeoTIFF coordinates (build.py: cluster_scenes);
 *            tiles that fail the filter are dimmed, so you keep the geography
 *   grid   - only the tiles that pass the filter, ordered by the chosen key
 * Image levels: per-scene atlases (32 px/tile) when far out, then 64 px and 256 px proxies, then
 * full-resolution tiles rendered by the server once a tile is larger than ~560 device pixels. */

const $ = (s) => document.querySelector(s);
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const f0 = (x) => (Number.isFinite(x) ? Math.round(x).toString() : '–');
const f1 = (x) => (Number.isFinite(x) ? x.toFixed(1) : '–');
const f2 = (x) => (Number.isFinite(x) ? x.toFixed(2) : '–');
const pct = (a, b) => (b > 0 ? (100 * a) / b : NaN);
const mean = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : NaN);
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const SPLIT_COLORS = { train: '#4c8dff', val: '#ffd23f', test: '#ff4fd8', holdout: '#ff8c42', train_extra: '#2dd4bf', none: '#6b7785' };
const TYPE_COLORS = { earthquake: '#d29a5c', wildfire: '#ff6b3d', volcano: '#e05252', explosion: '#b48cff', flood: '#3fa7ff', conflict: '#9aa5b1', hurricane: '#2ec4b6' };
const LIMIT = { atlas: 400, 64: 10000, 256: 900, full: 40 };
const GRID_GAP = 0.12, GRID_ROW = 1.46;
const CLS = ['bg', 'intact', 'damaged', 'destroyed'];

const S = {
  meta: null, T: null, N: 0, events: [], sceneTiles: [], nameIdx: new Map(), runs: [], runData: {}, run: '',
  view: 'mosaic', scope: 'all', base: 'opt', bright: 1, contrast: 1,
  ov: { gt: { on: false, a: 0.7 }, pred: { on: false, a: 0.7 }, err: { on: false, a: 0.9 }, split: { on: false, a: 0.4 } },
  scheme: 'standard_ML', filt: null, sort: 'event',
  cam: { x: 0, y: 0, z: 20 }, rx: null, ry: null, labels: [], order: [], hash: new Map(),
  matches: null, scopeMask: null, hover: -1, dirty: false, cw: 1, ch: 1, dpr: 1, ctx: null,
  D: { i: -1, base: 'opt', alpha: 0.65, s: 1, tx: 0, ty: 0 },
  pending: {},
};
const IMG = { atlas: new Map(), 64: new Map(), 256: new Map(), full: new Map() };

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

// ------------------------------------------------------------------ tile helpers
const tileEvent = (i) => S.events[S.T.event[i]];
const tileType = (i) => S.meta.events[tileEvent(i)].type;
const schemeLabels = () => (S.meta.schemes[S.scheme] ? S.meta.schemes[S.scheme].labels : []);
const splitOf = (i) => (S.meta.schemes[S.scheme] && S.meta.schemes[S.scheme].tile[i]) || '';
const primarySplit = (i) => splitOf(i).split('+')[0] || 'none';
function bldFrac(i) { const c = S.T.counts[i]; const b = c[1] + c[2] + c[3]; return b / (b + c[0] || 1); }
function dmgFrac(i) { const c = S.T.counts[i]; const b = c[1] + c[2] + c[3]; return b ? (c[2] + c[3]) / b : 0; }
function dstFrac(i) { const c = S.T.counts[i]; const b = c[1] + c[2] + c[3]; return b ? c[3] / b : 0; }
const cmOf = (i) => (S.run && S.runData[S.run] ? S.runData[S.run].cm[i] : null);
const hasPred = (i) => !!cmOf(i);

function cmStats(cm) {   // cm: 16 numbers, rows = ground truth, cols = prediction
  const row = [0, 0, 0, 0], col = [0, 0, 0, 0];
  let tot = 0, diag = 0;
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) { const v = cm[r * 4 + c]; row[r] += v; col[c] += v; tot += v; if (r === c) diag += v; }
  const iou = [], present = [];
  for (let k = 0; k < 4; k++) { const tp = cm[k * 5], u = row[k] + col[k] - tp; iou.push(u > 0 ? tp / u : NaN); present.push(row[k] + col[k] > 0); }
  return { row, col, tot, diag, iou, present };
}
function setMetrics(cm) {   // same definitions as the repo's train/infer scripts
  const s = cmStats(cm);
  let tp = 0, fp = 0, fn = 0;
  for (let r = 1; r < 4; r++) for (let c = 1; c < 4; c++) tp += cm[r * 4 + c];
  for (let c = 1; c < 4; c++) fp += cm[c];
  for (let r = 1; r < 4; r++) fn += cm[r * 4];
  const f1s = [1, 2, 3].map((k) => {
    const t = cm[k * 5]; let rs = 0, cs = 0;
    for (let j = 0; j < 4; j++) rs += cm[k * 4 + j];
    for (let r = 1; r < 4; r++) cs += cm[r * 4 + k];
    return (2 * t) / (2 * t + (cs - t) + (rs - t)) || 0;
  });
  return {
    oa: s.diag / s.tot, iou: s.iou, present: s.present, allPresent: s.present.every(Boolean),
    miou: mean(s.iou.filter((v, k) => s.present[k])),
    f1loc: (2 * tp) / (2 * tp + fp + fn), f1clf: f1s.some((v) => !(v > 0)) ? 0 : 3 / f1s.reduce((a, v) => a + 1 / v, 0),
  };
}
function tileMIoU(i) { const cm = cmOf(i); if (!cm) return NaN; const s = cmStats(cm); return mean(s.iou.filter((v, k) => s.present[k])); }
function tileBldErr(i) {
  const cm = cmOf(i); if (!cm) return NaN;
  let b = 0, ok = 0;
  for (let k = 1; k < 4; k++) { for (let j = 0; j < 4; j++) b += cm[k * 4 + j]; ok += cm[k * 5]; }
  return b ? 1 - ok / b : NaN;
}

// ------------------------------------------------------------------ init
async function init() {
  const [meta, runs] = await Promise.all([getJSON('/api/meta'), getJSON('/api/runs')]);
  S.meta = meta; S.T = meta.tiles; S.N = S.T.name.length; S.events = meta.event_list; S.runs = runs;
  S.T.name.forEach((n, i) => S.nameIdx.set(n, i));
  S.sceneTiles = meta.scenes.map(() => []);
  for (let i = 0; i < S.N; i++) S.sceneTiles[S.T.scene[i]].push(i);
  for (const a of S.sceneTiles) a.sort((p, q) => S.T.row[p] - S.T.row[q] || S.T.col[p] - S.T.col[q]);
  readHash();
  buildControls();
  if (S.run && !S.runs.some((r) => r.name === S.run)) S.run = '';
  if (S.run) await loadRun(S.run);
  syncControls();
  setupCanvas();
  resize();
  computeMatches();
  doLayout();
  if (S.pending.cam) Object.assign(S.cam, S.pending.cam); else fitView();
  updateAll();
  if (S.pending.tile !== undefined && S.nameIdx.has(S.pending.tile)) openDetail(S.nameIdx.get(S.pending.tile));
}

function defaultFilter() {
  return { splits: new Set([...schemeLabels(), 'none']), types: new Set(S.meta.types), minBld: 0, minDmg: 0, onlyPred: false, search: '' };
}

async function loadRun(name) {
  if (!name || S.runData[name]) return;
  const r = await getJSON(`/api/run/${encodeURIComponent(name)}`);
  const cm = new Array(S.N).fill(null);
  for (const [n, v] of Object.entries(r.cm)) { const i = S.nameIdx.get(n); if (i !== undefined) cm[i] = v; }
  S.runData[name] = { cm, built: r.built, summary: r.summary, pred_dir: r.pred_dir, n: r.n, v: encodeURIComponent(r.built) };
}

// ------------------------------------------------------------------ controls
function buildControls() {
  const scope = $('#scope');
  scope.innerHTML = '';
  scope.append(new Option(`All events · ${S.N} tiles`, 'all'));
  for (const t of S.meta.types) {
    const g = document.createElement('optgroup');
    g.label = t;
    for (const e of S.events) {
      if (S.meta.events[e].type !== t) continue;
      g.append(new Option(`${e} · ${S.meta.events[e].n} tiles`, `event:${e}`));
      const scs = S.meta.scenes.filter((s) => s.event === e);
      if (scs.length > 1) for (const sc of scs) g.append(new Option(`   └ area ${sc.area} · ${sc.n} tiles`, `scene:${sc.id}`));
    }
    scope.append(g);
  }
  const scheme = $('#scheme');
  scheme.innerHTML = '';
  for (const k of Object.keys(S.meta.schemes)) scheme.append(new Option(k, k));
  refreshRunSelect();

  document.querySelectorAll('#viewSeg button').forEach((b) => b.addEventListener('click', () => setView(b.dataset.v)));
  document.querySelectorAll('#baseSeg button').forEach((b) => b.addEventListener('click', () => setBase(b.dataset.b)));
  scope.addEventListener('change', () => { S.scope = scope.value; refresh({ layout: true, fit: true }); });
  let searchT = 0;
  $('#search').addEventListener('input', (e) => { clearTimeout(searchT); searchT = setTimeout(() => { S.filt.search = e.target.value.trim().toLowerCase(); refresh(); }, 200); });
  $('#bright').addEventListener('input', (e) => { S.bright = +e.target.value; requestDraw(); applyD(); });
  $('#contrast').addEventListener('input', (e) => { S.contrast = +e.target.value; requestDraw(); applyD(); });
  for (const k of Object.keys(S.ov)) {
    $(`#ov_${k}`).addEventListener('change', (e) => { S.ov[k].on = e.target.checked; requestDraw(); writeHash(); });
    $(`#a_${k}`).addEventListener('input', (e) => { S.ov[k].a = +e.target.value; requestDraw(); });
  }
  scheme.addEventListener('change', () => {
    S.scheme = scheme.value; S.filt.splits = new Set([...schemeLabels(), 'none']);
    refresh({ layout: S.view === 'grid' });
  });
  $('#run').addEventListener('change', async (e) => {
    S.run = e.target.value;
    if (S.run) {
      await loadRun(S.run);
      if (!S.ov.pred.on && !S.ov.err.on) { S.ov.pred.on = true; $('#ov_pred').checked = true; }
    }
    refresh({ layout: S.view === 'grid' });
    if (S.D.i >= 0) openDetail(S.D.i);
  });
  $('#f_bld').addEventListener('input', (e) => { S.filt.minBld = +e.target.value; $('#f_bld_v').textContent = e.target.value; refresh(); });
  $('#f_dmg').addEventListener('input', (e) => { S.filt.minDmg = +e.target.value; $('#f_dmg_v').textContent = e.target.value; refresh(); });
  $('#f_pred').addEventListener('change', (e) => { S.filt.onlyPred = e.target.checked; refresh(); });
  $('#sort').addEventListener('change', (e) => { S.sort = e.target.value; if (S.view === 'grid') refresh({ layout: true, fit: true }); else writeHash(); });
  $('#resetFilt').addEventListener('click', () => { S.filt = defaultFilter(); syncControls(); refresh(); });
  for (const [id, key] of [['#f_splits', 'splits'], ['#f_types', 'types']]) {
    const box = $(id);
    box.addEventListener('click', (e) => {
      const c = e.target.closest('.chip'); if (!c) return;
      const set = S.filt[key], v = c.dataset.v;
      if (e.altKey || e.shiftKey) { set.clear(); set.add(v); } else if (set.has(v)) set.delete(v); else set.add(v);
      refresh();
    });
  }
  $('#rescan').addEventListener('click', rescan);
  $('#helpBtn').addEventListener('click', () => { $('#helpBox').hidden = false; });
  $('#helpClose').addEventListener('click', () => { $('#helpBox').hidden = true; });
  $('#d_close').addEventListener('click', closeDetail);
  $('#d_prev').addEventListener('click', () => stepDetail(-1));
  $('#d_next').addEventListener('click', () => stepDetail(1));
  $('#detail').addEventListener('click', (e) => { if (e.target.id === 'detail') closeDetail(); });
  document.querySelectorAll('#d_baseSeg button').forEach((b) => b.addEventListener('click', () => {
    S.D.base = b.dataset.b; document.querySelectorAll('#d_baseSeg button').forEach((x) => x.classList.toggle('on', x === b)); renderPanels(false);
  }));
  $('#d_alpha').addEventListener('input', (e) => { S.D.alpha = +e.target.value; document.querySelectorAll('#d_panels img.ov').forEach((im) => { im.style.opacity = S.D.alpha; }); });
  document.addEventListener('keydown', onKey);
}

function refreshRunSelect() {
  const sel = $('#run');
  sel.innerHTML = '';
  sel.append(new Option(S.runs.length ? 'none' : 'none (no runs yet)', ''));
  for (const r of S.runs) sel.append(new Option(`${r.name} · ${r.n} tiles`, r.name));
  sel.value = S.run;
}

function syncControls() {
  $('#scope').value = S.scope;
  if ($('#scope').value !== S.scope) { S.scope = 'all'; $('#scope').value = 'all'; }
  document.querySelectorAll('#viewSeg button').forEach((b) => b.classList.toggle('on', b.dataset.v === S.view));
  document.querySelectorAll('#baseSeg button').forEach((b) => b.classList.toggle('on', b.dataset.b === S.base));
  $('#bright').value = S.bright; $('#contrast').value = S.contrast;
  for (const k of Object.keys(S.ov)) { $(`#ov_${k}`).checked = S.ov[k].on; $(`#a_${k}`).value = S.ov[k].a; }
  $('#scheme').value = S.scheme;
  $('#run').value = S.run;
  $('#f_bld').value = S.filt.minBld; $('#f_bld_v').textContent = S.filt.minBld;
  $('#f_dmg').value = S.filt.minDmg; $('#f_dmg_v').textContent = S.filt.minDmg;
  $('#f_pred').checked = S.filt.onlyPred;
  $('#search').value = S.filt.search;
  $('#sort').value = S.sort;
}

function setView(v) {
  if (v === S.view) return;
  S.view = v;
  document.querySelectorAll('#viewSeg button').forEach((b) => b.classList.toggle('on', b.dataset.v === v));
  refresh({ layout: true, fit: true });
}
function setBase(b) {
  S.base = b;
  document.querySelectorAll('#baseSeg button').forEach((x) => x.classList.toggle('on', x.dataset.b === b));
  requestDraw(); writeHash();
}
function toggleOv(k) { S.ov[k].on = !S.ov[k].on; $(`#ov_${k}`).checked = S.ov[k].on; requestDraw(); writeHash(); }

function refresh({ layout = false, fit = false } = {}) {
  computeMatches();
  if (layout || S.view === 'grid') { doLayout(); if (fit || S.view === 'grid') fitView(); }
  updateAll();
}
function updateAll() { updateStats(); updateChips(); updateLegend(); updateRunNote(); updateCount(); writeHash(); requestDraw(); }

// ------------------------------------------------------------------ filtering + layout
function sceneInScope(sc) {
  if (S.scope === 'all') return true;
  if (S.scope.startsWith('event:')) return sc.event === S.scope.slice(6);
  if (S.scope.startsWith('scene:')) return sc.id === Number(S.scope.slice(6));
  return true;
}
function computeMatches() {
  const f = S.filt, N = S.N, sceneOK = S.meta.scenes.map(sceneInScope);
  S.matches = new Uint8Array(N); S.scopeMask = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    if (!sceneOK[S.T.scene[i]]) continue;
    S.scopeMask[i] = 1;
    if (!(splitOf(i) || 'none').split('+').some((l) => f.splits.has(l))) continue;
    if (!f.types.has(tileType(i))) continue;
    if (f.minBld && 100 * bldFrac(i) < f.minBld) continue;
    if (f.minDmg && 100 * dmgFrac(i) < f.minDmg) continue;
    if (f.onlyPred && !hasPred(i)) continue;
    if (f.search && !S.T.name[i].includes(f.search)) continue;
    S.matches[i] = 1;
  }
}
function doLayout() { if (S.view === 'mosaic') layoutMosaic(); else layoutGrid(); buildHash(); }

function layoutMosaic() {
  const N = S.N;
  S.rx = new Float32Array(N).fill(NaN); S.ry = new Float32Array(N).fill(NaN); S.labels = []; S.order = [];
  const scenes = S.meta.scenes.filter(sceneInScope);
  let area = 0, maxW = 0;
  for (const sc of scenes) { area += (sc.cols + 2) * (sc.rows + 3); maxW = Math.max(maxW, sc.cols); }
  const target = Math.max(Math.sqrt(area) * 1.6, maxW);
  let x = 0, y = 0, rowH = 0;
  for (const sc of scenes) {
    if (x > 0 && x + sc.cols > target) { x = 0; y += rowH + 3; rowH = 0; }
    for (const i of S.sceneTiles[sc.id]) { S.rx[i] = x + S.T.col[i]; S.ry[i] = y + S.T.row[i]; }
    S.labels.push({ x, y, sc, maxW: Infinity });
    x += sc.cols + 2; rowH = Math.max(rowH, sc.rows);
  }
  for (let k = 0; k + 1 < S.labels.length; k++) {   // a label may run into the gap, up to the next scene in its row
    const a = S.labels[k], b = S.labels[k + 1];
    if (b.y === a.y) a.maxW = b.x - a.x - 0.4;
  }
  for (const sc of scenes) for (const i of S.sceneTiles[sc.id]) if (S.matches[i]) S.order.push(i);
}

function layoutGrid() {
  const N = S.N;
  S.rx = new Float32Array(N).fill(NaN); S.ry = new Float32Array(N).fill(NaN); S.labels = [];
  const idx = [];
  for (let i = 0; i < N; i++) if (S.matches[i]) idx.push(i);
  sortTiles(idx);
  const n = Math.max(1, idx.length);
  const cols = Math.max(1, Math.min(n, Math.round(Math.sqrt((n * (S.cw / S.ch) * GRID_ROW) / (1 + GRID_GAP)))));
  idx.forEach((i, k) => { S.rx[i] = (k % cols) * (1 + GRID_GAP); S.ry[i] = Math.floor(k / cols) * GRID_ROW; });
  S.order = idx;
}

function sortTiles(idx) {
  const T = S.T;
  const by = (f, dir) => idx.sort((a, b) => dir * (f(a) - f(b)) || a - b);
  switch (S.sort) {
    case 'bld': return by(bldFrac, -1);
    case 'dmg': return by(dmgFrac, -1);
    case 'destroyed': return by(dstFrac, -1);
    case 'nodata': return by((i) => T.nodata[i], -1);
    case 'miou': return by((i) => { const v = tileMIoU(i); return Number.isFinite(v) ? v : 9; }, 1);
    case 'err': return by((i) => { const v = tileBldErr(i); return Number.isFinite(v) ? v : -1; }, -1);
    case 'name': return idx.sort((a, b) => (T.name[a] < T.name[b] ? -1 : 1));
    default: return idx;   // tiles are stored ordered by disaster type, event, name
  }
}

function buildHash() {
  const H = new Map();
  for (let i = 0; i < S.N; i++) {
    const x = S.rx[i]; if (x !== x) continue;
    const y = S.ry[i];
    for (let gx = Math.floor(x); gx <= Math.floor(x + 0.999); gx++) {
      for (let gy = Math.floor(y); gy <= Math.floor(y + 0.999); gy++) {
        const k = gx * 65536 + gy; let a = H.get(k);
        if (!a) H.set(k, (a = []));
        a.push(i);
      }
    }
  }
  S.hash = H;
}
function hit(mx, my) {
  const wx = S.cam.x + mx / S.cam.z, wy = S.cam.y + my / S.cam.z;
  const a = S.hash.get(Math.floor(wx) * 65536 + Math.floor(wy));
  if (!a) return -1;
  for (let k = a.length - 1; k >= 0; k--) {
    const i = a[k];
    if (wx >= S.rx[i] && wx < S.rx[i] + 1 && wy >= S.ry[i] && wy < S.ry[i] + 1) return i;
  }
  return -1;
}

// ------------------------------------------------------------------ camera + canvas
function fitView() {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < S.N; i++) {
    const x = S.rx[i]; if (x !== x) continue;
    const y = S.ry[i];
    x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x + 1); y1 = Math.max(y1, y + 1);
  }
  if (x0 === Infinity) return;
  if (S.view === 'mosaic') y0 -= 1.3;
  const pad = 28;
  const z = clamp(Math.min((S.cw - 2 * pad) / (x1 - x0), (S.ch - 2 * pad) / (y1 - y0)), 1, 1500);
  S.cam = { z, x: x0 - (S.cw / z - (x1 - x0)) / 2, y: y0 - (S.ch / z - (y1 - y0)) / 2 };
  requestDraw(); writeHash();
}

function setupCanvas() {
  const cv = $('#cv');
  S.ctx = cv.getContext('2d');
  new ResizeObserver(resize).observe(cv);
  let drag = null;
  const pos = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  cv.addEventListener('pointerdown', (e) => {
    drag = { x: e.clientX, y: e.clientY, cx: S.cam.x, cy: S.cam.y, moved: false };
    cv.setPointerCapture(e.pointerId); cv.classList.add('drag');
  });
  cv.addEventListener('pointermove', (e) => {
    const [mx, my] = pos(e);
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      S.cam.x = drag.cx - dx / S.cam.z; S.cam.y = drag.cy - dy / S.cam.z;
      hideTip(); requestDraw();
      return;
    }
    const i = hit(mx, my);
    if (i !== S.hover) { S.hover = i; requestDraw(); }
    if (i >= 0) showTip(i, mx, my); else hideTip();
  });
  cv.addEventListener('pointerup', (e) => {
    cv.classList.remove('drag');
    const d = drag; drag = null;
    if (d && !d.moved) { const [mx, my] = pos(e); const i = hit(mx, my); if (i >= 0) openDetail(i); }
    writeHash();
  });
  cv.addEventListener('pointerleave', () => { if (!drag) { S.hover = -1; hideTip(); requestDraw(); } });
  cv.addEventListener('wheel', (e) => {
    e.preventDefault();
    const [mx, my] = pos(e);
    const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
    const z2 = clamp(S.cam.z * Math.exp(-dy * 0.0015), 0.5, 4000);
    const wx = S.cam.x + mx / S.cam.z, wy = S.cam.y + my / S.cam.z;
    S.cam.z = z2; S.cam.x = wx - mx / z2; S.cam.y = wy - my / z2;
    hideTip(); requestDraw(); writeHash();
  }, { passive: false });
}
function resize() {
  const cv = $('#cv'), r = cv.getBoundingClientRect();
  S.dpr = window.devicePixelRatio || 1; S.cw = Math.max(1, r.width); S.ch = Math.max(1, r.height);
  cv.width = Math.round(S.cw * S.dpr); cv.height = Math.round(S.ch * S.dpr);
  requestDraw();
}

// ------------------------------------------------------------------ images
function imgURL(layer, i, level, run) {
  const n = S.T.name[i];
  if (level === 'full') return run ? `/full/${layer}/${run}/${n}?v=${S.runData[run].v}` : `/full/${layer}/${n}`;
  if (run) return `/runs/${run}/p${level}/${layer}/${n}.png?v=${S.runData[run].v}`;
  return `/p/${level}/${layer}/${n}.${layer === 'gt' ? 'png' : 'jpg'}`;
}
function fetchImg(map, key, url, limit) {
  let e = map.get(key);
  if (e) { if (e.ok) { map.delete(key); map.set(key, e); return e.img; } return null; }
  e = { img: new Image(), ok: false };
  e.img.decoding = 'async';
  e.img.onload = () => { e.ok = true; requestDraw(); };
  e.img.src = url;
  map.set(key, e);
  if (map.size > limit) {
    const it = map.keys();
    for (let d = map.size - limit; d > 0; d--) map.delete(it.next().value);
  }
  return null;
}
const getImg = (layer, i, level, run) => fetchImg(IMG[level], `${run}|${layer}|${i}`, imgURL(layer, i, level, run), LIMIT[level]);
function peekImg(layer, i, level, run) { const e = IMG[level].get(`${run}|${layer}|${i}`); return e && e.ok ? e.img : null; }
function pickImg(layer, i, level, run) {
  if (level === 'full') return getImg(layer, i, 'full', run) || getImg(layer, i, 256, run) || peekImg(layer, i, 64, run);
  if (level === 256) return getImg(layer, i, 256, run) || getImg(layer, i, 64, run);
  return getImg(layer, i, 64, run);
}
function getAtlas(layer, sid, run) {
  const url = run ? `/runs/${run}/atlas/${layer}/scene_${sid}.png?v=${S.runData[run].v}`
    : `/atlas/${layer}/scene_${sid}.png?v=${S.meta.built}`;
  return fetchImg(IMG.atlas, `${run}|${layer}|${sid}`, url, LIMIT.atlas);
}

// ------------------------------------------------------------------ drawing
function requestDraw() { if (!S.dirty) { S.dirty = true; requestAnimationFrame(draw); } }
function fitText(ctx, s, w) {
  if (w <= 8) return '';
  if (ctx.measureText(s).width <= w) return s;
  let lo = 0, hi = s.length;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (ctx.measureText(s.slice(0, m) + '…').width <= w) lo = m; else hi = m - 1; }
  return s.slice(0, lo) + '…';
}

function draw() {
  S.dirty = false;
  const ctx = S.ctx, z = S.cam.z, W = S.cw, H = S.ch, cx = S.cam.x, cy = S.cam.y;
  ctx.setTransform(S.dpr, 0, 0, S.dpr, 0, 0);
  ctx.fillStyle = '#0b0f14'; ctx.fillRect(0, 0, W, H);
  if (!S.rx) return;
  const vis = [], x1 = cx + W / z, y1 = cy + H / z;
  for (let i = 0; i < S.N; i++) {
    const x = S.rx[i]; if (x !== x || x + 1 < cx || x > x1) continue;
    const y = S.ry[i]; if (y + 1 < cy || y > y1) continue;
    vis.push(i);
  }
  const dev = z * S.dpr;
  let lvl = dev < 40 ? 'atlas' : dev < 110 ? 64 : dev < 560 ? 256 : 'full';
  if (lvl === 'atlas' && S.view === 'grid') lvl = 64;
  if (lvl === 'full' && vis.length > 24) lvl = 256;
  const run = S.run && S.runData[S.run] ? S.run : '';
  const X = (i) => (S.rx[i] - cx) * z, Y = (i) => (S.ry[i] - cy) * z;

  ctx.fillStyle = '#17202b';
  for (const i of vis) ctx.fillRect(X(i), Y(i), z, z);
  const layers = [];
  if (S.base !== 'none') layers.push({ layer: S.base, a: 1, run: '', base: true });
  if (S.ov.gt.on) layers.push({ layer: 'gt', a: S.ov.gt.a, run: '' });
  if (run && S.ov.pred.on) layers.push({ layer: 'pred', a: S.ov.pred.a, run });
  if (run && S.ov.err.on) layers.push({ layer: 'err', a: S.ov.err.a, run });
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'medium';
  for (const L of layers) {
    ctx.globalAlpha = L.a;
    ctx.filter = L.base && (S.bright !== 1 || S.contrast !== 1) ? `brightness(${S.bright}) contrast(${S.contrast})` : 'none';
    if (lvl === 'atlas') {
      for (const lab of S.labels) {
        const sc = lab.sc, sx = (lab.x - cx) * z, sy = (lab.y - cy) * z, sw = sc.cols * z, sh = sc.rows * z;
        if (sx > W || sy > H || sx + sw < 0 || sy + sh < 0) continue;
        const im = getAtlas(L.layer, sc.id, L.run);
        if (im) ctx.drawImage(im, sx, sy, sw, sh);
      }
    } else {
      for (const i of vis) {
        if (L.run && !hasPred(i)) continue;
        const im = pickImg(L.layer, i, lvl, L.run);
        if (im) ctx.drawImage(im, X(i), Y(i), z, z);
      }
    }
  }
  ctx.globalAlpha = 1; ctx.filter = 'none';

  if (S.ov.split.on) {
    ctx.globalAlpha = S.ov.split.a;
    for (const i of vis) { const l = primarySplit(i); if (l === 'none') continue; ctx.fillStyle = SPLIT_COLORS[l] || '#888'; ctx.fillRect(X(i), Y(i), z, z); }
    ctx.globalAlpha = 1;
  }
  if (S.view === 'mosaic') {
    ctx.fillStyle = 'rgba(8,11,15,0.82)';
    for (const i of vis) if (!S.matches[i]) ctx.fillRect(X(i) - 0.5, Y(i) - 0.5, z + 1, z + 1);
  }
  if (z >= 26) {
    ctx.strokeStyle = 'rgba(255,255,255,0.09)'; ctx.lineWidth = 1;
    for (const i of vis) ctx.strokeRect(X(i) + 0.5, Y(i) + 0.5, z - 1, z - 1);
  }
  if (S.view === 'grid' && z >= 90) {
    ctx.textBaseline = 'top';
    for (const i of vis) {
      const x = X(i), y = Y(i) + z + 4, l = primarySplit(i), n = S.T.name[i];
      ctx.fillStyle = SPLIT_COLORS[l] || '#888'; ctx.fillRect(x, y + 2, 8, 8);
      ctx.font = '600 11px system-ui, sans-serif'; ctx.fillStyle = '#d9e1ea';
      const short = `${n.split('-')[0]} #${parseInt(n.slice(n.lastIndexOf('_') + 1), 10)}`;
      ctx.fillText(fitText(ctx, z >= 260 ? n : short, z - 12), x + 12, y);
      ctx.font = '11px system-ui, sans-serif'; ctx.fillStyle = '#8796a8';
      let s = `bldg ${f0(100 * bldFrac(i))}% · dmg ${f0(100 * dmgFrac(i))}%`;
      if (cmOf(i)) s += ` · mIoU ${f0(100 * tileMIoU(i))}`;
      ctx.fillText(fitText(ctx, s, z), x, y + 15);
    }
  }
  if (S.view === 'mosaic') {
    ctx.font = '600 12px system-ui, sans-serif'; ctx.textBaseline = 'bottom';
    for (const lab of S.labels) {
      const sc = lab.sc, sx = (lab.x - cx) * z, sy = (lab.y - cy) * z - 5;
      if (sx > W || sy < -4 || sy > H + 30) continue;
      const w = Math.min(lab.maxW * z, Math.max(sc.cols * z, 420), W - sx - 4) - 13;
      const short = sc.event.split('-')[0] + (sc.label !== sc.event ? ` ${sc.area}` : '');
      const txt = [`${sc.label} · ${sc.n} tiles`, sc.label, `${short} · ${sc.n}`, short].find((s) => ctx.measureText(s).width <= w);
      ctx.fillStyle = TYPE_COLORS[S.meta.events[sc.event].type] || '#aaa'; ctx.fillRect(sx, sy - 11, 9, 9);
      ctx.fillStyle = '#d9e1ea';
      ctx.fillText(txt || fitText(ctx, short, w), sx + 13, sy);
    }
  }
  const h = S.hover;
  if (h >= 0 && S.rx[h] === S.rx[h]) { ctx.strokeStyle = '#ffd166'; ctx.lineWidth = 2; ctx.strokeRect(X(h) + 1, Y(h) + 1, z - 2, z - 2); }
}

// ------------------------------------------------------------------ tooltip, stats, legend
function showTip(i, mx, my) {
  const tip = $('#tip'), T = S.T, ev = S.meta.events[tileEvent(i)], sc = S.meta.scenes[T.scene[i]];
  const c = T.counts[i], b = c[1] + c[2] + c[3];
  let h = `<div class="t">${esc(T.name[i])}</div>`;
  h += `<div><span class="badge" style="background:${TYPE_COLORS[ev.type]}">${esc(ev.type)}</span>${esc(ev.place)} · ${esc(ev.date)}</div>`;
  h += `<div class="muted">${esc(sc.label)}</div>`;
  h += `<div>split (${esc(S.scheme)}): ${splitBadges(splitOf(i))}</div>`;
  h += `<div>buildings ${f1(100 * bldFrac(i))}% of tile${b ? ` · intact ${f1(pct(c[1], b))}% · damaged ${f1(pct(c[2], b))}% · destroyed ${f1(pct(c[3], b))}%` : ''}</div>`;
  if (T.nodata[i] > 0.005) h += `<div class="muted">no-data (black) in optical: ${f1(100 * T.nodata[i])}%</div>`;
  if (cmOf(i)) h += `<div>run: tile mIoU <b>${f1(100 * tileMIoU(i))}</b> · building px wrong <b>${f1(100 * tileBldErr(i))}%</b></div>`;
  else if (S.run) h += '<div class="muted">not predicted in this run</div>';
  tip.innerHTML = h; tip.hidden = false;
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  tip.style.left = `${mx + 16 + tw > S.cw ? Math.max(4, mx - tw - 12) : mx + 16}px`;
  tip.style.top = `${my + 16 + th > S.ch ? Math.max(4, my - th - 12) : my + 16}px`;
}
function hideTip() { $('#tip').hidden = true; }
function splitBadges(lab) {
  if (!lab) return '<span class="badge" style="background:#6b7785">none</span>';
  return lab.split('+').map((l) => `<span class="badge" style="background:${SPLIT_COLORS[l] || '#888'}">${esc(l)}</span>`).join('');
}

function classBar(c) {
  const b = c[1] + c[2] + c[3];
  if (!b) return '<div class="note">no building pixels</div>';
  const rgb = S.meta.class_rgb;
  const seg = [1, 2, 3].map((k) => `<span style="width:${(100 * c[k]) / b}%;background:rgb(${rgb[k]})" title="${S.meta.classes[k]} ${f1(pct(c[k], b))}%"></span>`).join('');
  return `<div class="bar">${seg}</div><div class="muted">intact ${f1(pct(c[1], b))}% · damaged ${f1(pct(c[2], b))}% · destroyed ${f1(pct(c[3], b))}%</div>`;
}

function updateStats() {
  const T = S.T, c = [0, 0, 0, 0], sp = {}, cm = new Float64Array(16), ev = new Map();
  let n = 0, np = 0;
  for (let i = 0; i < S.N; i++) {
    if (!S.matches[i]) continue;
    n++;
    const t = T.counts[i];
    for (let k = 0; k < 4; k++) c[k] += t[k];
    const l = primarySplit(i); sp[l] = (sp[l] || 0) + 1;
    const m = cmOf(i);
    if (m) {
      np++;
      let a = ev.get(T.event[i]);
      if (!a) ev.set(T.event[i], (a = { n: 0, cm: new Float64Array(16) }));
      a.n++;
      for (let k = 0; k < 16; k++) { cm[k] += m[k]; a.cm[k] += m[k]; }
    }
  }
  const b = c[1] + c[2] + c[3];
  let h = `<div><span class="big">${n.toLocaleString()}</span> tiles <span class="muted">in view after filters</span></div>`;
  h += `<div class="chips" style="margin-top:6px">${Object.entries(sp).map(([l, k]) => `<span class="chip"><span class="dot" style="background:${SPLIT_COLORS[l] || '#888'}"></span>${esc(l)} <span class="n">${k}</span></span>`).join('')}</div>`;
  h += `<div class="sub">Buildings are ${f1(pct(b, b + c[0]))}% of pixels. Of building pixels:</div>${classBar(c)}`;
  if (S.run && S.runData[S.run]) {
    if (!np) {
      h += `<div class="note">No tile in this selection has a prediction in run “${esc(S.run)}”.</div>`;
    } else {
      const M = setMetrics(cm);
      h += `<div class="sub">Run “${esc(S.run)}” on the ${np} predicted tiles (set level)</div>`;
      h += `<div class="kv"><span>mIoU</span><b>${f2(100 * M.miou)}${M.allPresent ? '' : '*'}</b><span>OA</span><b>${f2(100 * M.oa)}</b>`;
      h += `<span>F1 localisation</span><b>${f2(100 * M.f1loc)}</b><span>F1 damage (harm.)</span><b>${f2(100 * M.f1clf)}</b>`;
      if (ev.size > 1) h += `<span>mIoU, event average</span><b>${f2(100 * mean([...ev.values()].map((a) => setMetrics(a.cm).miou)))}</b>`;
      h += '</div>';
      h += `<table><tr><th>IoU</th>${CLS.map((x) => `<th>${x}</th>`).join('')}</tr><tr><td></td>${M.iou.map((v, k) => `<td>${M.present[k] ? f1(100 * v) : '–'}</td>`).join('')}</tr></table>`;
      if (!M.allPresent) h += '<div class="note">* mIoU over the classes present in this selection.</div>';
      if (ev.size > 1) {
        const rows = [...ev.entries()].map(([e, a]) => ({ e: S.events[e], n: a.n, m: setMetrics(a.cm) })).sort((p, q) => p.m.miou - q.m.miou);
        h += `<div class="sub">Per event (worst first) · click to zoom to it</div><div class="evtable"><table><tr><th>event</th><th>tiles</th><th>mIoU</th><th>I</th><th>D</th><th>X</th></tr>`;
        for (const r of rows) h += `<tr data-ev="${esc(r.e)}"><td>${esc(r.e)}</td><td>${r.n}</td><td><b>${f1(100 * r.m.miou)}</b></td>${[1, 2, 3].map((k) => `<td>${r.m.present[k] ? f1(100 * r.m.iou[k]) : '–'}</td>`).join('')}</tr>`;
        h += '</table></div>';
      }
    }
  }
  const box = $('#stats');
  box.innerHTML = h;
  box.querySelectorAll('tr[data-ev]').forEach((tr) => tr.addEventListener('click', () => {
    S.scope = `event:${tr.dataset.ev}`; $('#scope').value = S.scope; refresh({ layout: true, fit: true });
  }));
}

function updateChips() {
  const spC = {}, tyC = {};
  for (let i = 0; i < S.N; i++) {
    if (!S.scopeMask[i]) continue;
    for (const l of (splitOf(i) || 'none').split('+')) spC[l] = (spC[l] || 0) + 1;
    const t = tileType(i); tyC[t] = (tyC[t] || 0) + 1;
  }
  const chip = (v, color, label, n, on) => `<span class="chip${on ? '' : ' off'}" data-v="${esc(v)}" title="click: toggle · shift-click: only this"><span class="dot" style="background:${color}"></span>${esc(label)} <span class="n">${n || 0}</span></span>`;
  $('#f_splits').innerHTML = [...schemeLabels(), 'none'].filter((l) => spC[l] || l !== 'none')
    .map((l) => chip(l, SPLIT_COLORS[l] || '#888', l, spC[l], S.filt.splits.has(l))).join('');
  $('#f_types').innerHTML = S.meta.types.map((t) => chip(t, TYPE_COLORS[t], t, tyC[t], S.filt.types.has(t))).join('');
}

function updateLegend() {
  const rgb = S.meta.class_rgb;
  let h = '<div class="grp">Building classes (truth / prediction)</div>';
  for (const k of [1, 2, 3]) h += `<div class="row"><span class="sw" style="background:rgb(${rgb[k]})"></span>${S.meta.classes[k]}</div>`;
  h += '<div class="grp">Errors (prediction vs truth)</div>';
  for (const e of S.meta.err_cats) h += `<div class="row"><span class="sw" style="background:rgb(${e.rgb})${e.key === 'correct' ? ';opacity:.45' : ''}"></span>${esc(e.label)}</div>`;
  h += `<div class="grp">Splits (${esc(S.scheme)})</div>`;
  for (const l of schemeLabels()) h += `<div class="row"><span class="sw" style="background:${SPLIT_COLORS[l]}"></span>${l}</div>`;
  h += '<div class="grp">Disaster types</div>';
  for (const t of S.meta.types) h += `<div class="row"><span class="sw" style="background:${TYPE_COLORS[t]}"></span>${t}</div>`;
  $('#legend').innerHTML = h;
}

function updateRunNote() {
  const r = S.run && S.runData[S.run];
  $('#runNote').innerHTML = r
    ? `Run <b>${esc(S.run)}</b>: ${r.n} predicted tiles from <code>${esc(r.pred_dir)}</code> (built ${esc(r.built)})`
    : S.runs.length ? 'Pick a run (top right) to overlay predictions and errors.'
      : 'No evaluation runs yet: write predictions with infer_UNet.py into results/&lt;name&gt;/ and press “Rescan runs”.';
}
function updateCount() {
  let a = 0, b = 0;
  for (let i = 0; i < S.N; i++) { a += S.matches[i]; b += S.scopeMask[i]; }
  $('#count').textContent = a === b ? `${b.toLocaleString()} tiles` : `${a.toLocaleString()} of ${b.toLocaleString()} tiles match`;
}

// ------------------------------------------------------------------ tile detail view
function openDetail(i) {
  hideTip();
  const T = S.T, n = T.name[i], ev = S.meta.events[tileEvent(i)], sc = S.meta.scenes[T.scene[i]];
  const same = S.D.i === i;
  S.D.i = i;
  if (!same) { S.D.s = 1; S.D.tx = 0; S.D.ty = 0; }
  $('#d_title').innerHTML = `${esc(n)} <span class="badge" style="background:${TYPE_COLORS[ev.type]}">${esc(ev.type)}</span>`;
  const splits = Object.entries(S.meta.schemes).map(([k, s]) => `${esc(k)} ${splitBadges(s.tile[i])}`).join(' &nbsp; ');
  const ll = T.lat[i] != null
    ? `<a href="https://www.openstreetmap.org/?mlat=${T.lat[i]}&mlon=${T.lon[i]}#map=17/${T.lat[i]}/${T.lon[i]}" target="_blank" rel="noopener noreferrer">${T.lat[i].toFixed(5)}, ${T.lon[i].toFixed(5)}</a>`
    : '?';
  $('#d_info').innerHTML = `<b>${esc(ev.place)}</b> · ${esc(ev.paper_type)} · ${esc(ev.date)} · imagery: ${esc(ev.sources)} (paper GSD ${esc(ev.gsd)} m)<br>`
    + `scene <b>${esc(sc.label)}</b> (${sc.n} tiles), column ${T.col[i].toFixed(0)}, row ${T.row[i].toFixed(0)} · EPSG:${T.crs[i] ?? '?'} · `
    + `pixel ${T.res[i] ? T.res[i].toFixed(3) + ' m' : '?'} · centre ${ll}`
    + (T.nodata[i] > 0.005 ? ` · <b>${f1(100 * T.nodata[i])}%</b> of the optical tile is no-data` : '')
    + `<br>${splits}`;
  renderPanels(!same);
  $('#detail').hidden = false;
  writeHash();
}

function renderPanels() {
  const i = S.D.i, n = S.T.name[i], run = S.run && cmOf(i) ? S.run : '', v = run ? S.runData[run].v : '';
  const under = S.D.base === 'sar' ? `/full/sar/${n}` : `/full/opt/${n}`;
  const none = S.run ? 'This tile has no prediction in the selected run.' : 'Select a run (top right) to see predictions.';
  const P = [
    { t: 'Optical · pre-event', base: `/full/opt/${n}` },
    { t: 'SAR · post-event', base: `/full/sar/${n}` },
    { t: 'Ground truth', base: under, ov: `/full/gt/${n}` },
    { t: run ? `Prediction · ${run}` : 'Prediction', base: under, ov: run ? `/full/pred/${run}/${n}?v=${v}` : null },
    { t: 'Errors', base: under, ov: run ? `/full/err/${run}/${n}?v=${v}` : null },
  ];
  const wrap = $('#d_panels');
  wrap.innerHTML = '';
  for (const p of P) {
    const d = document.createElement('div');
    d.className = 'panel';
    d.innerHTML = p.ov === null
      ? `<span class="ptitle">${esc(p.t)}</span><div class="empty">${esc(none)}</div>`
      : `<span class="ptitle">${esc(p.t)}</span><div class="stage"><img class="base" src="${p.base}" alt="">${p.ov ? `<img class="ov" src="${p.ov}" alt="" style="opacity:${S.D.alpha}">` : ''}</div>`;
    wrap.append(d);
  }
  const st = document.createElement('div');
  st.className = 'panel statcell';
  st.innerHTML = tileStatsHTML(i);
  wrap.append(st);
  applyD();
  bindPanels();
}

function tileStatsHTML(i) {
  const c = S.T.counts[i], tot = c[0] + c[1] + c[2] + c[3], b = tot - c[0];
  let h = '<h4>Ground-truth pixels</h4><table><tr><th></th><th>pixels</th><th>% tile</th><th>% bldg</th></tr>';
  for (let k = 0; k < 4; k++) h += `<tr><td>${CLS[k]}</td><td>${c[k].toLocaleString()}</td><td>${f1(pct(c[k], tot))}</td><td>${k ? f1(pct(c[k], b)) : ''}</td></tr>`;
  h += '</table>';
  const cm = cmOf(i);
  if (!cm) return h + `<div class="muted">${S.run ? 'No prediction for this tile in the selected run.' : 'Select a run to see prediction statistics.'}</div>`;
  const s = cmStats(cm);
  h += '<h4>Confusion (rows truth, columns prediction, % of row)</h4><table><tr><th></th>' + CLS.map((x) => `<th>${x}</th>`).join('') + '</tr>';
  for (let r = 0; r < 4; r++) {
    if (!s.row[r]) continue;
    h += `<tr><td>${CLS[r]}</td>${[0, 1, 2, 3].map((k) => `<td${r === k ? ' style="color:#fff;font-weight:650"' : ''}>${f1(pct(cm[r * 4 + k], s.row[r]))}</td>`).join('')}</tr>`;
  }
  h += '</table><h4>IoU (classes present in this tile)</h4><table><tr>' + CLS.map((x) => `<th>${x}</th>`).join('') + '<th>mIoU</th></tr><tr>';
  h += s.iou.map((v, k) => `<td>${s.present[k] ? f1(100 * v) : '–'}</td>`).join('') + `<td><b>${f1(100 * tileMIoU(i))}</b></td></tr></table>`;
  return h + `<div class="muted">building pixels with the wrong class: <b>${f1(100 * tileBldErr(i))}%</b></div>`;
}

function applyD() {
  const t = `translate(${S.D.tx}px, ${S.D.ty}px) scale(${S.D.s})`;
  document.querySelectorAll('#d_panels .stage').forEach((st) => { st.style.transform = t; });
  const f = S.bright !== 1 || S.contrast !== 1 ? `brightness(${S.bright}) contrast(${S.contrast})` : '';
  document.querySelectorAll('#d_panels img.base').forEach((im) => { im.style.filter = f; });
}
function clampD(w) { const lim = w - w * S.D.s; S.D.tx = clamp(S.D.tx, lim, 0); S.D.ty = clamp(S.D.ty, lim, 0); }
function bindPanels() {
  document.querySelectorAll('#d_panels .panel:not(.statcell)').forEach((p) => {
    p.addEventListener('wheel', (e) => {
      e.preventDefault();
      const r = p.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
      const s2 = clamp(S.D.s * Math.exp(-dy * 0.0018), 1, 24);
      S.D.tx = mx - (mx - S.D.tx) * (s2 / S.D.s); S.D.ty = my - (my - S.D.ty) * (s2 / S.D.s); S.D.s = s2;
      clampD(r.width); applyD();
    }, { passive: false });
    p.addEventListener('pointerdown', (e) => {
      const r = p.getBoundingClientRect(), st = { x: e.clientX, y: e.clientY, tx: S.D.tx, ty: S.D.ty };
      p.setPointerCapture(e.pointerId);
      const mv = (ev) => { S.D.tx = st.tx + ev.clientX - st.x; S.D.ty = st.ty + ev.clientY - st.y; clampD(r.width); applyD(); };
      const up = () => { p.removeEventListener('pointermove', mv); p.removeEventListener('pointerup', up); };
      p.addEventListener('pointermove', mv); p.addEventListener('pointerup', up);
    });
    p.addEventListener('dblclick', () => { S.D.s = 1; S.D.tx = 0; S.D.ty = 0; applyD(); });
  });
}
function stepDetail(d) {
  const o = S.order;
  if (!o.length) return;
  const k = o.indexOf(S.D.i);
  openDetail(o[k < 0 ? 0 : (k + d + o.length) % o.length]);
}
function closeDetail() { $('#detail').hidden = true; S.D.i = -1; writeHash(); }

// ------------------------------------------------------------------ keyboard, rescan, URL state
function onKey(e) {
  if (e.target.matches('input, select, textarea')) return;
  if (!$('#detail').hidden) {
    if (e.key === 'Escape') closeDetail(); else if (e.key === 'ArrowLeft') stepDetail(-1); else if (e.key === 'ArrowRight') stepDetail(1);
    return;
  }
  if (!$('#helpBox').hidden) { if (e.key === 'Escape' || e.key === '?') $('#helpBox').hidden = true; return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === 'm') setView(S.view === 'mosaic' ? 'grid' : 'mosaic');
  else if (k === 'o' || k === 's' || k === 'n') setBase({ o: 'opt', s: 'sar', n: 'none' }[k]);
  else if (k === 'g' || k === 'p' || k === 'e' || k === 't') toggleOv({ g: 'gt', p: 'pred', e: 'err', t: 'split' }[k]);
  else if (k === 'f') fitView();
  else if (e.key === '?') $('#helpBox').hidden = false;
}

function toast(msg, ms = 0) {
  const t = $('#toast');
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast.t);
  if (ms) toast.t = setTimeout(() => { t.hidden = true; }, ms);
}
async function rescan() {
  const btn = $('#rescan');
  btn.disabled = true;
  toast('Scanning results/ for new evaluation folders …');
  try {
    await getJSON('/api/rescan', { method: 'POST' });
    let st;
    do { await wait(1000); st = await getJSON('/api/rescan'); } while (st.running);
    S.runs = await getJSON('/api/runs');
    for (const r of S.runs) if (S.runData[r.name] && S.runData[r.name].built !== r.built) delete S.runData[r.name];
    refreshRunSelect();
    if (S.run) await loadRun(S.run);
    refresh({ layout: S.view === 'grid' });
    toast(st.msg, 6000);
  } catch (err) {
    toast(`Rescan failed: ${err.message}`, 8000);
  } finally {
    btn.disabled = false;
  }
}

function writeHash() {
  clearTimeout(writeHash.t);
  writeHash.t = setTimeout(() => {
    if (!S.meta) return;
    const p = new URLSearchParams(), f = S.filt;
    p.set('view', S.view); p.set('scope', S.scope); p.set('base', S.base);
    const ov = Object.keys(S.ov).filter((k) => S.ov[k].on);
    if (ov.length) p.set('ov', ov.join(','));
    if (S.run) p.set('run', S.run);
    if (S.scheme !== 'standard_ML') p.set('scheme', S.scheme);
    if (S.sort !== 'event') p.set('sort', S.sort);
    if ([...schemeLabels(), 'none'].some((l) => !f.splits.has(l))) p.set('splits', [...f.splits].join(','));
    if (S.meta.types.some((t) => !f.types.has(t))) p.set('types', [...f.types].join(','));
    if (f.minBld) p.set('bld', f.minBld);
    if (f.minDmg) p.set('dmg', f.minDmg);
    if (f.onlyPred) p.set('onlypred', '1');
    if (f.search) p.set('q', f.search);
    if (S.D.i >= 0) p.set('tile', S.T.name[S.D.i]);
    p.set('cam', [S.cam.x, S.cam.y, S.cam.z].map((v) => v.toFixed(2)).join(','));
    history.replaceState(null, '', `#${p.toString()}`);
  }, 300);
}
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  if (['mosaic', 'grid'].includes(p.get('view'))) S.view = p.get('view');
  if (p.get('scope')) S.scope = p.get('scope');
  if (['opt', 'sar', 'none'].includes(p.get('base'))) S.base = p.get('base');
  for (const k of (p.get('ov') || '').split(',')) if (S.ov[k]) S.ov[k].on = true;
  if (p.get('scheme') && S.meta.schemes[p.get('scheme')]) S.scheme = p.get('scheme');
  if (p.get('sort')) S.sort = p.get('sort');
  if (p.get('run')) S.run = p.get('run');
  S.filt = defaultFilter();
  if (p.get('splits') !== null) S.filt.splits = new Set(p.get('splits').split(',').filter(Boolean));
  if (p.get('types') !== null) S.filt.types = new Set(p.get('types').split(',').filter(Boolean));
  S.filt.minBld = +p.get('bld') || 0;
  S.filt.minDmg = +p.get('dmg') || 0;
  S.filt.onlyPred = p.get('onlypred') === '1';
  S.filt.search = (p.get('q') || '').toLowerCase();
  if (p.get('tile')) S.pending.tile = p.get('tile');
  const cam = (p.get('cam') || '').split(',').map(Number);
  if (cam.length === 3 && cam.every(Number.isFinite) && cam[2] > 0) S.pending.cam = { x: cam[0], y: cam[1], z: cam[2] };
}

init().catch((err) => {
  document.body.innerHTML = `<pre style="color:#f88;padding:20px">Viewer failed to start: ${esc(err.stack || err)}</pre>`;
});
