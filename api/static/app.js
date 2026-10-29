"use strict";
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
let MAX_BYTES = 15 * 1024 * 1024, file = null, reqId = 0, ctrl = null, timer = null, last = null, cmpKey = null, cmpId = 0, cctrl = null, thumbUrl = null;
const FALLBACK_METHODS = ["Grad-CAM", "Grad-CAM++", "HiRes-CAM", "XGrad-CAM", "Layer-CAM", "Eigen-CAM", "Score-CAM (slow)"];

/* ---------------- start-up: health, integrity ---------------- */
async function health() {
  try {
    const h = await (await fetch("/api/health")).json();
    MAX_BYTES = h.max_mb * 1024 * 1024; $("maxmb").textContent = h.max_mb; $("disc").textContent = h.disclaimer + " Images are processed in memory on this computer and are not stored.";
    const sel = $("method"); sel.replaceChildren(...h.methods.map((m) => { const o = el("option", "", m); o.value = m; return o; }));
    const ig = h.integrity || { status: "unknown" }, pill = $("pill");
    if (!h.ready) { pill.className = "pill bad"; $("pillText").textContent = "Models not loaded"; const b = $("sysErr"); b.hidden = false; b.textContent = (h.error || "Models are not loaded.") + "  Copy your checkpoints into models/deploy/classifier_best.pt and the segmenter named in configs/segmentation_primary.json, then restart."; }
    else if (ig.status === "mismatch") { pill.className = "pill bad"; $("pillText").textContent = "Model checksum MISMATCH"; }
    else { pill.className = "pill ok"; $("pillText").textContent = "Models ready · " + h.device; }
    $("integrity").textContent = ig.status === "verified" ? "Model files verified (SHA-256)" : ig.status === "mismatch" ? "WARNING: a model file does not match its recorded SHA-256. Do not trust results." : "Model files could not be verified (no checksum file)";
  } catch (e) {
    $("pill").className = "pill bad"; $("pillText").textContent = "Server not reachable";
    $("method").replaceChildren(...FALLBACK_METHODS.map((m) => { const o = el("option", "", m); o.value = m; return o; })); syncControls();
  }
}

/* ---------------- input handling ---------------- */
function toast(msg) { const t = $("toast"); t.textContent = msg; t.hidden = false; clearTimeout(toast.t); toast.t = setTimeout(() => (t.hidden = true), 6500); }
function setFile(f) {
  if (!f) return;
  if (!/^image\//.test(f.type) && !/\.(png|jpe?g|bmp|tiff?|webp)$/i.test(f.name)) return toast("Please choose a PNG, JPEG, BMP, TIFF or WebP image.");
  if (f.size > MAX_BYTES) return toast("That file is larger than " + Math.round(MAX_BYTES / 1048576) + " MB.");
  file = f; cmpKey = null; cctrl && cctrl.abort(); if (thumbUrl) URL.revokeObjectURL(thumbUrl); thumbUrl = URL.createObjectURL(f);
  const t = $("thumb"); t.src = thumbUrl; t.hidden = false; run();
}
const drop = $("drop"), inp = $("file");
drop.addEventListener("click", () => inp.click());
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); inp.click(); } });
inp.addEventListener("change", () => setFile(inp.files[0]));
["dragenter", "dragover"].forEach((n) => drop.addEventListener(n, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((n) => drop.addEventListener(n, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
window.addEventListener("paste", (e) => { const it = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/")); if (it) setFile(it.getAsFile()); });

function syncControls() { const own = $("backend").value === "own"; if (own) $("method").value = "Grad-CAM"; $("method").disabled = own; $("hint").textContent = /Score/.test($("method").value) ? "Score-CAM runs the network many times; expect a long wait on a CPU." : ""; }
["method", "layer", "backend"].forEach((id) => $(id).addEventListener("change", () => { syncControls(); schedule(); }));
function schedule() { clearTimeout(timer); if (file) timer = setTimeout(run, 200); }

/* ---------------- analysis request ---------------- */
async function post(url, extra, signal) {
  const fd = new FormData(); fd.append("file", file); for (const [k, v] of Object.entries(extra)) fd.append(k, v);
  const r = await fetch(url, { method: "POST", body: fd, signal }); let body = null;
  try { body = await r.json(); } catch (_) { /* non-JSON error page */ }
  if (!r.ok) { const d = body && body.detail; throw new Error(typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join("; ") : "Request failed (" + r.status + ")"); }
  return body;
}
async function run() {
  if (!file) return; if (ctrl) ctrl.abort(); ctrl = new AbortController(); const my = ++reqId;
  $("spin").hidden = false; if (!last) $("empty").textContent = "Analysing…";
  try {
    const r = await post("/api/analyze", { method: $("method").value, layer: $("layer").value, backend: $("backend").value }, ctrl.signal);
    if (my !== reqId) return; last = r; render(r); $("spin").hidden = true;
    if (!$("tab-cmp").hidden) loadCompare();
  } catch (e) { if (e.name === "AbortError") return; if (my !== reqId) return; $("spin").hidden = true; $("empty").textContent = "Analysis failed."; toast(e.message); }
}

/* ---------------- rendering ---------------- */
function render(r) {
  $("empty").hidden = true; $("result").hidden = false; const p = r.prediction;
  $("cls").textContent = p["class"].replace(/_/g, " "); $("conf").textContent = pct(p.confidence); $("band").textContent = p.calibration_band;
  const b = $("trust"); b.textContent = r.trust.level; b.className = "badge " + r.trust.level; $("trustSum").textContent = r.trust.summary;
  const wl = $("warnings"); wl.replaceChildren(...r.warnings.map((w) => el("li", "", w)));
  const im = r.images; $("base").src = im.base; $("heat").src = im.heat;
  for (const [id, key] of [["fill", "mask_fill"], ["edge", "mask_edge"]]) { const e = $(id); if (im[key]) { e.src = im[key]; e.dataset.has = "1"; } else { e.removeAttribute("src"); e.dataset.has = ""; } }
  applyView();
  const pr = $("probs"); pr.replaceChildren();
  Object.entries(p.probabilities).sort((a, b) => b[1] - a[1]).forEach(([k, v], i) => {
    const row = el("div", "prow" + (i === 0 ? " top" : "")); row.title = "uncalibrated: " + pct(p.probabilities_uncalibrated[k]);
    const t = el("div", "t"), f = el("div", "f"); t.append(f); row.append(el("span", "", k.replace(/_/g, " ")), t, el("span", "", pct(v))); pr.append(row); requestAnimationFrame(() => (f.style.width = (v * 100).toFixed(1) + "%"));
  });
  const ck = $("checks"); ck.replaceChildren();
  r.trust.checks.forEach((c) => { const li = el("li"), d = el("div"); li.append(el("span", "ico " + c.status, { pass: "✓", warn: "!", fail: "✕", na: "–" }[c.status])); d.append(el("span", "", c.label), el("small", "", "rule: " + c.rule)); li.append(d, el("b", "v", c.value)); ck.append(li); });
  const s = r.segmentation, m = $("meas"); m.replaceChildren(); const kv = (k, v) => m.append(el("dt", "", k), el("dd", "", v));
  kv("Region found", s.region_found ? "yes" : "no"); kv("Area", s.area_px + " px (" + s.area_percent + "% of slice)");
  if (s.present) { kv("Bounding box", s.bbox.w + " × " + s.bbox.h + " px"); kv("Centre", "x " + s.centroid.x + ", y " + s.centroid.y); }
  if (s.disagrees_with_classifier) kv("Note", "region and class disagree");
  $("text").textContent = r.text; const x = r.explanation;
  $("expNote").textContent = x.note + (x.library_vs_project_gradcam_corr !== null ? "  Library vs project Grad-CAM correlation: " + x.library_vs_project_gradcam_corr.toFixed(3) + "." : "");
  const t = $("tech"); t.replaceChildren(); const tk = (k, v) => t.append(el("dt", "", k), el("dd", "", String(v)));
  tk("Explanation", x.label); tk("Temperature", p.temperature); tk("Input", r.meta.input.width + "×" + r.meta.input.height + " " + r.meta.input.format + " (" + r.meta.input.mode + ")"); tk("Input SHA-256", r.meta.input_sha256 + "…"); tk("Request", r.meta.request_id); tk("Runtime", r.meta.runtime_ms + " ms");
  const slim = JSON.parse(JSON.stringify(r)); for (const k of Object.keys(slim.images)) if (typeof slim.images[k] === "string") slim.images[k] = "[PNG data omitted]"; $("json").textContent = JSON.stringify(slim, null, 2);
}

/* ---------------- viewer controls ---------------- */
function applyView() {
  $("heat").style.display = $("cHeat").checked ? "" : "none"; $("heat").style.opacity = $("oHeat").value / 100;
  $("fill").style.display = $("cFill").checked && $("fill").dataset.has ? "" : "none"; $("edge").style.display = $("cEdge").checked && $("edge").dataset.has ? "" : "none";
  $("ov").style.clipPath = "inset(0 " + (100 - $("swipe").value) + "% 0 0)";
  $("zoom").style.transform = "scale(" + $("zoomR").value / 100 + ")";
}
["cHeat", "cFill", "cEdge", "oHeat", "swipe", "zoomR"].forEach((id) => $(id).addEventListener("input", applyView));
$("stage").addEventListener("mousemove", (e) => { const b = $("stage").getBoundingClientRect(); $("zoom").style.transformOrigin = ((e.clientX - b.left) / b.width * 100).toFixed(1) + "% " + ((e.clientY - b.top) / b.height * 100).toFixed(1) + "%"; });

/* ---------------- tabs and comparison ---------------- */
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("on", x === t)); ["over", "cmp", "det"].forEach((n) => ($("tab-" + n).hidden = n !== t.dataset.tab));
  if (t.dataset.tab === "cmp") loadCompare();
}));
function grid(host, items) { host.replaceChildren(...items.map((it) => { const c = el("div", "cg"); if (it.image) { const i = el("img"); i.src = it.image; i.alt = it.label; c.append(i, el("div", "", it.label)); } else c.append(el("div", "er", it.label + ": " + it.error)); return c; })); }
async function loadCompare() {
  if (!file) return; const key = file.name + file.size + file.lastModified + $("layer").value; if (key === cmpKey) return;
  const mine = ++cmpId; cctrl && cctrl.abort(); cctrl = new AbortController(); $("cmpMethods").replaceChildren(el("p", "hint", "Computing explanations…")); $("cmpLayers").replaceChildren();
  try { const r = await post("/api/compare", { layer: $("layer").value }, cctrl.signal); if (mine !== cmpId) return; cmpKey = key; grid($("cmpMethods"), r.methods); grid($("cmpLayers"), r.layers); }
  catch (e) { if (e.name !== "AbortError") { $("cmpMethods").replaceChildren(el("p", "hint", "Comparison failed.")); toast(e.message); } }
}

/* ---------------- downloads ---------------- */
function save(href, name) { const a = el("a"); a.href = href; a.download = name; document.body.append(a); a.click(); a.remove(); }
$("dlPng").addEventListener("click", () => last && save(last.images.panel, "report_" + last.meta.request_id + ".png"));
$("dlJson").addEventListener("click", () => { if (!last) return; const s = JSON.parse(JSON.stringify(last)); delete s.images; const u = URL.createObjectURL(new Blob([JSON.stringify(s, null, 2)], { type: "application/json" })); save(u, "result_" + last.meta.request_id + ".json"); setTimeout(() => URL.revokeObjectURL(u), 2000); });

syncControls(); health();