/* SLoG SNR^2 explorer -- user interface only.
   Every number on this page comes from the slogpet package running in Pyodide;
   nothing here computes physics. */
"use strict";

const PAL = ["--s1", "--s2", "--s3", "--s4"];        // validated 4-slot categorical
const DASH = ["", "7 3", "2 2.5", "9 2.5 2 2.5"];    // configuration within a family
const MAXFAM = PAL.length, MAXCFG = DASH.length;

const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const SUP = { "-": "\u207b", "0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3",
  "4": "\u2074", "5": "\u2075", "6": "\u2076", "7": "\u2077", "8": "\u2078", "9": "\u2079" };
const sup = (n) => String(n).split("").map(c => SUP[c] || c).join("");
function sci(v, d = 2) {                     // 5e-4 -> 5·10⁻⁴
  if (v === 0) return "0";
  const e = Math.floor(Math.log10(Math.abs(v)));
  const m = v / Math.pow(10, e);
  const ms = (+m.toFixed(d)).toString();
  return (ms === "1" ? "" : ms + "\u00b7") + "10" + sup(e);
}
const fmt = (v, d = 3) => (v === null || v === undefined || !isFinite(v)) ? "—"
  : (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0)) ? sci(v, d - 1)
    : (+v.toPrecision(d)).toString();

let PY = null, CAT = null, SERIES = [], STALE = true, BUSY = false;
const CUSTOM = [];
let LOGY = false;

/* ---------------------------------------------------------------- charts */
function niceTicks(lo, hi, n) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || 10 * mag;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi * (1 + 1e-9); t += step) out.push(t);
  return out;
}
function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, e);
      if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
    }
  }
  return out;
}
function svgEl(n, a) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", n);
  for (const k in a) e.setAttribute(k, a[k]);
  return e;
}

function lineChart(host, opt) {
  const S = opt.series.filter(s => s.pts.length);
  host.innerHTML = "";
  if (!S.length) { host.innerHTML = '<div class="hint">Nothing selected.</div>'; return; }
  const label = S.length <= 4 && opt.endLabels !== false;
  const W = Math.max(340, host.clientWidth || 460), H = opt.height || 286;
  const CH = 5.9;                                     // approx px per character
  const rlab = label ? Math.max(...S.map(s => (s.short || s.label).length)) : 0;
  const M = { t: 10, r: label ? Math.min(120, 18 + rlab * CH) : 14, b: 38, l: 64 };
  let iw = W - M.l - M.r, ih = H - M.t - M.b;

  let xlo = Infinity, xhi = -Infinity, ylo = Infinity, yhi = -Infinity;
  for (const s of S) for (const [x, y] of s.pts) {
    if (x < xlo) xlo = x; if (x > xhi) xhi = x;
    if (opt.log && !(y > 0)) continue;
    if (y < ylo) ylo = y; if (y > yhi) yhi = y;
  }
  if (opt.log) { ylo = Math.max(ylo, yhi * 1e-6); }
  else { ylo = Math.min(0, ylo); yhi = yhi * 1.06; }
  if (!(yhi > ylo)) yhi = ylo + 1;

  const X = v => M.l + (v - xlo) / (xhi - xlo) * iw;
  const Y = opt.log
    ? v => M.t + ih - (Math.log10(Math.max(v, ylo)) - Math.log10(ylo))
      / (Math.log10(yhi) - Math.log10(ylo)) * ih
    : v => M.t + ih - (v - ylo) / (yhi - ylo) * ih;

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  svg.setAttribute("aria-label", opt.aria || opt.ylab);
  const grid = css("--grid"), axis = css("--axis"), muted = css("--muted"),
    ink2 = css("--ink-2"), surf = css("--surface");

  const yt = opt.log ? logTicks(ylo, yhi) : niceTicks(ylo, yhi, 5);
  // scientific on a log axis, fixed on a linear one: one notation per axis, and
  // the left margin follows from the labels rather than being guessed
  const ylabel = opt.yfmt || (opt.log ? (v => sci(v)) : (v => fmt(v, 2)));
  M.l = Math.max(46, 26 + Math.max(...yt.map(v => ylabel(v).length)) * CH);
  iw = W - M.l - M.r;
  for (const t of yt) {
    svg.appendChild(svgEl("line", { x1: M.l, x2: M.l + iw, y1: Y(t), y2: Y(t), stroke: grid, "stroke-width": 1 }));
    const tx = svgEl("text", { x: M.l - 8, y: Y(t) + 3.5, "text-anchor": "end", fill: muted, "font-size": 10.5 });
    tx.textContent = ylabel(t);
    svg.appendChild(tx);
  }
  for (const t of niceTicks(xlo, xhi, 6)) {
    svg.appendChild(svgEl("line", { x1: X(t), x2: X(t), y1: M.t, y2: M.t + ih, stroke: grid, "stroke-width": 1 }));
    const tx = svgEl("text", { x: X(t), y: H - 20, "text-anchor": "middle", fill: muted, "font-size": 10.5 });
    tx.textContent = String(+t.toFixed(6));
    svg.appendChild(tx);
  }
  svg.appendChild(svgEl("line", { x1: M.l, x2: M.l + iw, y1: M.t + ih, y2: M.t + ih, stroke: axis, "stroke-width": 1 }));
  svg.appendChild(svgEl("line", { x1: M.l, x2: M.l, y1: M.t, y2: M.t + ih, stroke: axis, "stroke-width": 1 }));

  let t1 = svgEl("text", { x: M.l + iw / 2, y: H - 4, "text-anchor": "middle", fill: ink2, "font-size": 11.5 });
  t1.textContent = opt.xlab; svg.appendChild(t1);
  let t2 = svgEl("text", { x: 12, y: M.t + ih / 2, "text-anchor": "middle", fill: ink2, "font-size": 11.5, transform: `rotate(-90 12 ${M.t + ih / 2})` });
  t2.textContent = opt.ylab; svg.appendChild(t2);

  for (const s of S) {
    const d = s.pts.filter(p => !opt.log || p[1] > 0)
      .map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(2) + " " + Y(p[1]).toFixed(2)).join(" ");
    svg.appendChild(svgEl("path", {
      d, fill: "none", stroke: css(s.colour), "stroke-width": 2,
      "stroke-dasharray": s.dash, "stroke-linejoin": "round", "stroke-linecap": "round"
    }));
    if (label) {
      const last = s.pts[s.pts.length - 1];
      svg.appendChild(svgEl("rect", { x: M.l + iw + 6, y: Y(last[1]) - 4, width: 7, height: 7, rx: 2, fill: css(s.colour) }));
      const tt = svgEl("text", { x: M.l + iw + 17, y: Y(last[1]) + 3, fill: ink2, "font-size": 10.5 });
      tt.textContent = s.short || s.label; svg.appendChild(tt);
    }
  }

  /* hover: crosshair + one tooltip listing every series at that x */
  const hov = svgEl("g", { opacity: 0 });
  const rule = svgEl("line", { y1: M.t, y2: M.t + ih, stroke: axis, "stroke-width": 1, "stroke-dasharray": "3 3" });
  hov.appendChild(rule);
  const dots = S.map(s => {
    const g = svgEl("g", {});
    g.appendChild(svgEl("circle", { r: 5.5, fill: surf }));
    g.appendChild(svgEl("circle", { r: 4, fill: css(s.colour) }));
    hov.appendChild(g); return g;
  });
  svg.appendChild(hov);
  const hit = svgEl("rect", { x: M.l, y: M.t, width: iw, height: ih, fill: "transparent" });
  svg.appendChild(hit);
  const tip = $("tip");
  hit.addEventListener("pointermove", ev => {
    const pt = svg.getBoundingClientRect();
    const xv = xlo + (ev.clientX - pt.left) / pt.width * W;
    const xd = xlo + (xv - M.l) / iw * (xhi - xlo);
    let rows = "", any = false;
    S.forEach((s, i) => {
      let best = null, bd = Infinity;
      for (const p of s.pts) { const dd = Math.abs(p[0] - xd); if (dd < bd) { bd = dd; best = p; } }
      if (!best || (opt.log && !(best[1] > 0))) { dots[i].setAttribute("opacity", 0); return; }
      any = true;
      dots[i].setAttribute("opacity", 1);
      dots[i].setAttribute("transform", `translate(${X(best[0])} ${Y(best[1])})`);
      rule.setAttribute("x1", X(best[0])); rule.setAttribute("x2", X(best[0]));
      rows += `<div class="r"><span class="k" style="background:${css(s.colour)}"></span>` +
        `<span>${s.label}</span><span class="v">${fmt(best[1])}</span></div>`;
      if (i === 0) tip.dataset.x = String(best[0]);
    });
    hov.setAttribute("opacity", any ? 1 : 0);
    tip.innerHTML = `<div class="t">${opt.xlab.replace(/\s*\(.*\)/, "")} ` +
      `${fmt(+tip.dataset.x, 4)}${opt.xunit || ""}</div>${rows}`;
    tip.style.opacity = any ? 1 : 0;
    tip.style.left = Math.min(window.innerWidth - 240, ev.clientX + 14) + "px";
    tip.style.top = (ev.clientY - 12) + "px";
  });
  hit.addEventListener("pointerleave", () => { hov.setAttribute("opacity", 0); tip.style.opacity = 0; });
  host.appendChild(svg);
}

function legend(host, series) {
  host.innerHTML = "";
  if (series.length < 2) return;
  for (const s of series) {
    const d = document.createElement("div"); d.className = "it";
    d.innerHTML = `<svg viewBox="0 0 26 8"><line x1="1" y1="4" x2="25" y2="4" ` +
      `stroke="${css(s.colour)}" stroke-width="2" stroke-dasharray="${s.dash}"/></svg>` +
      `<span>${s.label}</span>`;
    host.appendChild(d);
  }
}

/* ------------------------------------------------------------- selection */
let FAMCOL = {};
function recolour() {
  /* Colour follows the entity, never its rank: a family keeps the slot it was
     first given for as long as anything from it is selected, so removing one
     configuration never repaints the survivors.  A slot is only reused once its
     family has left the selection entirely. */
  const have = new Set(chosen().map(s => s.family));
  for (const fam of Object.keys(FAMCOL)) if (!have.has(fam)) delete FAMCOL[fam];
  const used = new Set(Object.values(FAMCOL));
  for (const fam of have) {
    if (fam in FAMCOL) continue;
    const free = PAL.find(c => !used.has(c));
    if (!free) continue;                      // capped, not cycled
    FAMCOL[fam] = free; used.add(free);
  }
}
const familyColour = (fam) => FAMCOL[fam] || "--muted";
const ALL = () => [...CAT.systems, ...CAT.designs, ...CUSTOM];
function buildPicker() {
  const box = $("pick"); box.innerHTML = "";
  const add = (title, items, pre) => {
    const h = document.createElement("div"); h.className = "grp"; h.textContent = title;
    box.appendChild(h);
    items.forEach((sp, i) => {
      const id = pre + i;
      const l = document.createElement("label");
      l.innerHTML = `<input type="checkbox" id="${id}"><span class="swatch"></span>` +
        `<span>${sp.label}</span>`;
      box.appendChild(l);
      const cb = l.querySelector("input");
      cb.checked = !!(sp._cb && sp._cb.checked);
      cb.addEventListener("change", () => { onPick(); });
      sp._cb = cb; sp._sw = l.querySelector(".swatch");
    });
  };
  add("Published systems", CAT.systems, "sys");
  add("Generic detector designs", CAT.designs, "des");
  if (CUSTOM.length) add("Your own", CUSTOM, "cus");
}
function chosen() {
  return ALL().filter(s => s._cb && s._cb.checked);
}
function onPick() {
  const sel = chosen();
  recolour();
  for (const s of ALL()) {
    if (s._sw) s._sw.style.background = s._cb.checked
      ? `var(${familyColour(s.family)})` : "var(--axis)";
  }
  const fams = [...new Set(sel.map(s => s.family))];
  const perFam = {};
  for (const s of sel) perFam[s.family] = (perFam[s.family] || 0) + 1;
  for (const s of ALL()) {
    if (s._cb.checked) continue;
    s._cb.disabled = (!fams.includes(s.family) && fams.length >= MAXFAM)
      || (perFam[s.family] || 0) >= MAXCFG;
  }
  $("pickhint").textContent = sel.length
    ? `${sel.length} selected — colour is the crystal, dash is the configuration ` +
      `(at most ${MAXFAM} crystals and ${MAXCFG} configurations each).`
    : "Pick one or more. Colour distinguishes the crystal, dash the configuration.";
  markStale();
}
function assignStyles(sel) {
  /* Solid is the longest system of its family, as in Figure 7 of the paper, so
     that dash length reads as axial length rather than as selection order. */
  const rank = {};
  for (const fam of new Set(sel.map(s => s.family))) {
    rank[fam] = sel.filter(s => s.family === fam)
      .sort((a, b) => b.L_pet - a.L_pet).map(s => s.label);
  }
  const seen = {};
  return sel.map(sp => {
    seen[sp.family] = true;
    const n = rank[sp.family].indexOf(sp.label);
    return {
      spec: sp, key: sp.label, label: sp.label,
      short: (sp.L_pet / 10).toFixed(0) + " cm" + (sp.L_mrd ? " (MRD)" : ""),
      colour: familyColour(sp.family), dash: DASH[n % MAXCFG], pts: []
    };
  });
}

/* -------------------------------------------------------------- compute */
function markStale() {
  STALE = true;
  $("go").textContent = "Compute";
  $("status").textContent = SERIES.length ? "Settings changed — recompute." : "";
}
const sleep = () => new Promise(r => setTimeout(r, 0));

async function compute() {
  if (BUSY) return;
  const sel = chosen();
  if (!sel.length) { $("status").textContent = "Nothing selected."; return; }
  BUSY = true; $("go").disabled = true; $("prog").hidden = false;
  const F_o = +$("fo").value, D_cyl = +$("dcyl").value * 10;
  const nS = +$("npts").value, Smax = +$("smax").value * 10;
  const Ss = Array.from({ length: nS }, (_, i) => 50 + i * (Smax - 50) / (nS - 1));
  SERIES = assignStyles(sel);

  let done = 0;
  const total = SERIES.length * 2;
  for (const s of SERIES) {
    const chunks = [];
    for (let i = 0; i < Ss.length; i += 6) chunks.push(Ss.slice(i, i + 6));
    s.pts = []; s.beds = [];
    for (const c of chunks) {
      const r = JSON.parse(PY.sweep(JSON.stringify(s.spec), F_o, D_cyl, c));
      r.S.forEach((S, i) => { s.pts.push([S / 10, r.snr2[i]]); s.beds.push(r.n_beds[i]); });
      $("progbar").style.width = (100 * (done + 0.5) / total) + "%";
      await sleep();
    }
    done++; $("progbar").style.width = (100 * done / total) + "%";
  }
  await refreshAtS(done, total);
  $("prog").hidden = true; $("progbar").style.width = "0";
  BUSY = false; $("go").disabled = false; STALE = false;
  $("go").textContent = "Recompute";
  $("status").textContent = "";
  draw();
}

async function refreshAtS(done, total) {
  const F_o = +$("fo").value, D_cyl = +$("dcyl").value * 10, S = +$("sat").value * 10;
  for (const s of SERIES) {
    s.prof = JSON.parse(PY.profile(JSON.stringify(s.spec), F_o, D_cyl, S));
    s.sum = JSON.parse(PY.summary(JSON.stringify(s.spec), F_o, D_cyl, S));
    if (total) { done++; $("progbar").style.width = (100 * done / total) + "%"; }
    await sleep();
  }
}

/* ----------------------------------------------------------------- draw */
function draw() {
  if (!SERIES.length) return;
  const F_o = +$("fo").value, D = +$("dcyl").value, S = +$("sat").value;
  lineChart($("chart1"), {
    series: SERIES, log: LOGY, xlab: "scan length S (cm)", xunit: " cm",
    ylab: "SNR² (min over the range)",
    aria: "minimum squared signal to noise ratio against scan length"
  });
  legend($("leg1"), SERIES);
  $("cap1").textContent =
    `SLoG ${F_o.toFixed(1)} mm FWHM in a ${D} cm cylinder. Each point uses the ` +
    `bed protocol that maximises the worst point of the scan range.`;

  lineChart($("chart2"), {
    series: SERIES.map(s => ({
      ...s, pts: s.prof.z.map((z, i) => [z / 10, s.prof.snr2[i]])
    })),
    log: LOGY, xlab: "axial position z (cm)", xunit: " cm",
    ylab: "SNR²(z)", endLabels: false,
    aria: "squared signal to noise ratio along the axis"
  });
  legend($("leg2"), SERIES);
  $("cap2").textContent = `Scan length ${S} cm. The ripple is the bed structure; ` +
    `the curves in the left panel are the minima of these.`;
  table();
}

function table() {
  const best = Math.max(...SERIES.map(s => s.sum.snr2_min));
  const head = ["configuration", "L<sub>PET</sub><br>(cm)", "D<sub>PET</sub><br>(cm)",
    "&epsilon;", "r", "beds", "overlap<br>(%)", "min &eta;<sub>N</sub>",
    "SNR&sup2;<sub>min</sub>", "rel."];
  let h = "<thead><tr>" + head.map(t => `<th>${t}</th>`).join("") + "</tr></thead><tbody>";
  for (const s of SERIES) {
    const u = s.sum, sp = s.spec;
    h += "<tr><td><span class='name'><span class='swatch' style='background:" +
      `${css(s.colour)}'></span>${s.label}</span></td>` +
      `<td>${(sp.L_pet / 10).toFixed(1)}</td><td>${(sp.D_pet / 10).toFixed(1)}</td>` +
      `<td>${u.epsilon.toFixed(3)}</td><td>${fmt(u.r)}</td><td>${u.n_beds}</td>` +
      `<td>${u.overlap === null ? "—" : u.overlap.toFixed(0)}</td>` +
      `<td>${fmt(u.min_eta)}</td><td>${fmt(u.snr2_min)}</td>` +
      `<td>${(u.snr2_min / best).toFixed(2)}</td></tr>`;
  }
  $("tab").innerHTML = h + "</tbody>";
  $("tabnote").textContent = `at S = ${$("sat").value} cm; "rel." is relative to the ` +
    `best of the selected configurations`;
}

/* ------------------------------------------------------- custom systems */
function fillCustomFrom() {
  const sel = $("c-from");
  sel.innerHTML = ALL().map((s, i) => `<option value="${i}">${s.label}</option>`).join("");
  sel.onchange = () => {
    const s = ALL()[+sel.value];
    $("c-L").value = (s.L_pet / 10).toFixed(0);
    $("c-D").value = (s.D_pet / 10).toFixed(1);
    $("c-fy").value = s.F_y; $("c-fz").value = s.F_z;
    $("c-ctr").value = s.ctr !== null && s.ctr !== undefined ? s.ctr
      : (s.F_t ? Math.round(s.F_t / 0.15) : "");
    $("c-eps").value = (+s.epsilon).toFixed(3);
    $("c-mrd").value = s.L_mrd ? (s.L_mrd / 10).toFixed(0) : "";
    $("c-name").value = "like " + s.label.split(" (")[0];
  };
  sel.value = "2"; sel.onchange();
}
function addCustom() {
  const num = id => $(id).value === "" ? null : +$(id).value;
  const base = ALL()[+$("c-from").value];
  const spec = {
    kind: "custom", family: "your own", label: $("c-name").value || "custom",
    name: $("c-name").value || "custom",
    L_pet: num("c-L") * 10, D_pet: num("c-D") * 10,
    F_y: num("c-fy"), F_z: num("c-fz"),
    ctr: num("c-ctr"), F_t: null,
    L_mrd: num("c-mrd") === null ? null : num("c-mrd") * 10,
    epsilon: num("c-eps"), S_nema: null,
    crystal: base.crystal, crystal_size: base.crystal_size,
    energy_resolution: null, energy_window: null, reference: "",
    assumed: [], note: "defined in the browser"
  };
  for (const k of ["L_pet", "D_pet", "F_y", "F_z", "epsilon"]) {
    if (!(spec[k] > 0)) { $("status").textContent = "Fill in " + k + "."; return; }
  }
  CUSTOM.push(spec);
  buildPicker();
  spec._cb.checked = true;
  onPick();
  spec._cb.scrollIntoView({ block: "nearest" });
  fillCustomFrom();
  $("status").textContent = `Added "${spec.label}" — press Compute.`;
}

/* ----------------------------------------------------------------- boot */
async function main() {
  const bar = $("bootbar");
  bar.style.width = "12%";
  const pyodide = await loadPyodide({
    stdout: () => { }, stderr: (m) => console.warn(m)
  });
  bar.style.width = "35%";
  $("bootmsg").textContent = "Loading NumPy and SciPy…";
  await pyodide.loadPackage(["numpy", "scipy"]);
  bar.style.width = "80%";
  $("bootmsg").textContent = "Installing slogpet…";
  for (const dir of ["/app", "/app/slogpet", "/app/slogpet/data"]) {
    try { pyodide.FS.mkdir(dir); } catch (e) { }
  }
  for (const [path, src] of Object.entries(window.SLOGPET_FILES)) {
    pyodide.FS.writeFile("/app/" + path, src);
  }
  pyodide.runPython("import sys; sys.path.insert(0, '/app')");
  PY = pyodide.pyimport("api");
  CAT = JSON.parse(PY.catalogue());
  bar.style.width = "100%";

  buildPicker();
  CAT.systems[2]._cb.checked = true;      // Quadra
  CAT.systems[6]._cb.checked = true;      // Omni Legend 32
  CAT.systems[9]._cb.checked = true;      // uMI Panorama GS
  onPick();
  fillCustomFrom();
  $("c-add").onclick = addCustom;
  $("provbody").innerHTML = Object.entries(CAT.references)
    .map(([k, v]) => `<div style="margin-bottom:4px"><b>${k}.</b> ${v}</div>`).join("");
  $("boot").hidden = true; $("app").hidden = false;
  await compute();
}

/* --------------------------------------------------------------- wiring */
function bindTheme() {
  const set = d => {
    document.documentElement.dataset.theme = d ? "dark" : "light";
    $("th-d").setAttribute("aria-pressed", d); $("th-l").setAttribute("aria-pressed", !d);
    if (SERIES.length) draw();
  };
  $("th-d").onclick = () => set(true); $("th-l").onclick = () => set(false);
}
function bindScale() {
  const set = l => {
    LOGY = l; $("sc-log").setAttribute("aria-pressed", l);
    $("sc-lin").setAttribute("aria-pressed", !l); if (SERIES.length) draw();
  };
  $("sc-log").onclick = () => set(true); $("sc-lin").onclick = () => set(false);
}
window.addEventListener("DOMContentLoaded", () => {
  bindTheme(); bindScale();
  const live = { fo: v => (+v).toFixed(1), dcyl: v => v, smax: v => v, sat: v => v };
  for (const id in live) {
    const el = $(id), out = $(id + "-v");
    const upd = () => { out.textContent = live[id](el.value); };
    el.addEventListener("input", upd); upd();
  }
  for (const id of ["fo", "dcyl", "smax", "npts"]) $(id).addEventListener("change", markStale);
  $("sat").addEventListener("change", async () => {
    if (!SERIES.length || BUSY) return;
    BUSY = true; $("status").textContent = "…"; await refreshAtS(0, 0);
    BUSY = false; $("status").textContent = ""; draw();
  });
  $("go").onclick = compute;
  window.addEventListener("resize", () => { if (SERIES.length) draw(); });
  main().catch(e => {
    $("bootmsg").innerHTML = "<b>Could not start.</b> " + e +
      "<br>This page needs an internet connection the first time, to fetch Pyodide.";
    console.error(e);
  });
});
