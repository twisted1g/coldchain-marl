"use strict";

const KIND_ORDER = ["farm", "hub", "dc", "retail"];
const KIND_LABEL_NODE = { farm: "farm", hub: "hub", dc: "DC", retail: "retail" };
const KIND_COLOR = { farm: "#16a34a", hub: "#2563eb", dc: "#d97706", retail: "#db2777" };
const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const WEATHER_ICON = { sunny: "☀", cloudy: "☁", rainy: "☔", stormy: "⛈" };
const PLOT_BG = "#ffffff";
const PLOT_FONT = { color: "#64748b", size: 10 };
const GRID = "#e4e9f2";
const INK = "#1b2436";
const VEHICLE_COLOR = ["#2563eb", "#0891b2", "#7c3aed"];

const AGENT_KINDS = ["temperature", "routing", "spoilage", "inventory", "delivery"];
const KIND_LABEL = {
  temperature: "temperature", routing: "routing", spoilage: "spoilage",
  inventory: "inventory (retailers)", delivery: "delivery (vehicles)",
};
const METRICS = {
  temperature: [["deviation", "temp_deviation"]],
  routing: [["route cost", "route_cost"], ["delivered", "delivered"]],
  spoilage: [["false-neg", "fn_rate"], ["pred", "y_pred"]],
  inventory: [["order", "order"], ["unmet", "unmet_demand"], ["stock", "inventory_level"]],
  delivery: [["slot cost", "slot_cost"], ["delay", "delay"],
             ["sla", "sla_violated"], ["conflict", "conflict"]],
};

let EP = null;
let IDX = 0;
let TIMER = null;
let BASE_SHAPES = {};
let STREAM = null;
let LIVE = false;
let SELECTED = null; // agent name shown in the detail drawer, or null

// graph trace order (fixed) + animation state
const G = { EDGE: 0, SCHED: 1, ROUTE: 2, NODE: 3, TARGET: 4, CARGO: 5, SHIP: 6 };
let SPREV = null, RAF = null, GRAPH_READY = false, LAST_SHIP_NO = null;
const TWEEN_MS = 800;

const $ = (id) => document.getElementById(id);
const agentKind = (a) => a.replace(/_\d+$/, "");
const nodeKind = (n) => n.replace(/_\d+$/, "");
const fmt = (v, d = 3) => (typeof v === "number" ? v.toFixed(d) : v);
const setStatus = (s) => { $("status").textContent = s; };
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

// ---- data loading ---------------------------------------------------------
async function fetchEpisodes() {
  const { episodes } = await (await fetch("/api/episodes")).json();
  const sel = $("episodeSelect");
  sel.innerHTML = "";
  episodes.forEach((name) => {
    const o = document.createElement("option");
    o.value = o.textContent = name;
    sel.appendChild(o);
  });
  return episodes;
}

async function loadEpisode(name) {
  setStatus("loading " + name + " …");
  const data = await (await fetch("/api/episode/" + encodeURIComponent(name))).json();
  if (data.error) { setStatus(data.error); return; }
  setEpisode(data);
  setStatus(name + " · " + EP.ticks.length + " ticks");
}

async function runEpisode() {
  const body = {
    seed: Number($("runSeed").value) || 90000,
    episodes: Number($("runEpisodes").value) || 1,
    tag: $("runTag").value.trim(),
    scenario: $("runScenario").value.trim(),
    max_steps: $("runMaxSteps").value.trim(),
    mediator: $("runMediator").value,
  };
  setStatus("rolling out … (loading policies)");
  $("runBtn").disabled = true;
  try {
    const r = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (data.error) { setStatus("error: " + data.error); return; }
    await fetchEpisodes();
    $("episodeSelect").value = data.name;
    setEpisode(data);
    setStatus("ran " + data.name + " · " + EP.ticks.length + " ticks");
  } catch (e) {
    setStatus("request failed: " + e);
  } finally {
    $("runBtn").disabled = false;
  }
}

// ---- live streaming inference ---------------------------------------------
function toggleLive() {
  if (LIVE) { stopLive(); return; }
  startLive();
}

function startLive() {
  stopLive();
  stopPlay();
  const seed = Number($("runSeed").value) || 90000;
  const horizon = Number($("runHorizon").value) || 60;
  const tag = $("runTag").value.trim();
  const maxSteps = $("runMaxSteps").value.trim();
  const qs = new URLSearchParams({ seed, horizon, mediator: $("runMediator").value });
  if (tag) qs.set("tag", tag);
  if (maxSteps) qs.set("max_steps", maxSteps);

  LIVE = true;
  $("liveBtn").classList.add("on");
  $("liveBtn").textContent = "■ stop";
  setStatus("connecting live inference …");

  STREAM = new EventSource("/api/stream?" + qs.toString());
  STREAM.onmessage = (e) => {
    const rec = JSON.parse(e.data);
    if (rec.type === "meta") liveInit(rec);
    else if (rec.type === "tick") liveAppend(rec);
  };
  STREAM.addEventListener("end", () => {
    setStatus("live inference finished · " + (EP ? EP.ticks.length : 0) + " ticks");
    stopLive();
  });
  STREAM.addEventListener("error", () => {
    if (LIVE) setStatus("live stream error / disconnected");
    stopLive();
  });
}

function stopLive() {
  if (STREAM) { STREAM.close(); STREAM = null; }
  LIVE = false;
  $("liveBtn").classList.remove("on");
  $("liveBtn").textContent = "● live";
}

function liveInit(meta) {
  EP = { name: "live", meta, ticks: [], _pos: layout(meta),
         _depots: depotList(meta), _agents: [] };
  IDX = 0;
  GRAPH_READY = false;
  LAST_SHIP_NO = null;
  $("tickSlider").max = 0;
  renderLegend();
  BASE_SHAPES = {};
  setStatus("live · streaming …");
}

function liveAppend(rec) {
  if (!EP) return;
  EP.ticks.push(rec);
  if (!EP._agents.length) EP._agents = agentList(EP.ticks);
  const atEnd = IDX >= EP.ticks.length - 2; // follow the head unless user scrubbed back
  $("tickSlider").max = EP.ticks.length - 1;
  buildCharts();
  if (atEnd) IDX = EP.ticks.length - 1;
  render(atEnd);
  const problems = liveProblemCount(rec);
  setStatus(`live · tick ${rec.tick} · shipment #${rec.shipment_no || 1}` +
            (problems ? ` · ${problems} active problem(s)` : ""));
}

function liveProblemCount(t) {
  let n = t.disruptions.length;
  n += t.vehicles.filter((v) => v.conflict || v.sla_violated).length;
  n += t.inventory.unmet.filter((u) => u > 0.01).length;
  n += (t.order_queue || []).length;
  return n;
}

function setEpisode(data) {
  stopLive();
  stopPlay();
  data._pos = layout(data.meta);
  data._depots = depotList(data.meta);
  data._agents = agentList(data.ticks);
  EP = data;
  IDX = 0;
  GRAPH_READY = false;
  LAST_SHIP_NO = null;
  const slider = $("tickSlider");
  slider.max = EP.ticks.length - 1;
  slider.value = 0;
  renderLegend();
  buildCharts();
  render();
}

// ---- layout / helpers -----------------------------------------------------
function layout(meta) {
  const byKind = { farm: [], hub: [], dc: [], retail: [] };
  meta.nodes.forEach((n) => byKind[n.kind].push(n.name));
  const pos = {};
  KIND_ORDER.forEach((k, col) => {
    const names = byKind[k].slice().sort();
    const n = names.length;
    names.forEach((nm, row) => { pos[nm] = [col, (n - 1) / 2 - row]; });
  });
  return pos;
}

function depotList(meta) {
  // Every restock vehicle departs from the source farm (meta.source — the exact
  // node the sim measures retailer transit from) and, after dropping its load,
  // returns to it instantly (the sim defers the return trip). Staging the fleet
  // anywhere else would desync the drawn origin from the modelled transit cost.
  return [meta.source];
}
const vehicleDepot = () => EP._depots[0];

// The weighted farm->hub->dc->retail route the truck actually follows (same
// path the transit cost is built from). Falls back to a direct hop for old
// episodes recorded before restock_paths existed.
function restockPath(instance) {
  const paths = EP.meta.restock_paths;
  const p = paths && paths[instance];
  return p && p.length ? p : [vehicleDepot(), "retail_" + instance];
}

// Point along a node-path polyline at fraction ``p`` in [0,1], moving one graph
// hop per equal slice so the truck steps neighbour-to-neighbour up the chain.
function pathPoint(path, p) {
  const pos = EP._pos;
  const segs = path.length - 1;
  if (segs <= 0) return pos[path[0]];
  const x = clamp(p, 0, 1) * segs;
  const i = Math.min(Math.floor(x), segs - 1);
  const f = x - i;
  const a = pos[path[i]], b = pos[path[i + 1]];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
}

// Flatten a node path into Plotly line arrays (x, y, null-separated).
function pathSegments(path, rx, ry) {
  const pos = EP._pos;
  for (let i = 0; i < path.length - 1; i++) {
    if (!pos[path[i]] || !pos[path[i + 1]]) continue;
    rx.push(pos[path[i]][0], pos[path[i + 1]][0], null);
    ry.push(pos[path[i]][1], pos[path[i + 1]][1], null);
  }
}

function agentList(ticks) {
  const seen = new Set();
  ticks.forEach((t) => {
    Object.keys(t.rewards || {}).forEach((a) => seen.add(a));
    Object.keys(t.infos || {}).forEach((a) => seen.add(a));
  });
  return [...seen].sort((a, b) => {
    const ka = AGENT_KINDS.indexOf(agentKind(a));
    const kb = AGENT_KINDS.indexOf(agentKind(b));
    return ka !== kb ? ka - kb : a.localeCompare(b);
  });
}

function riskColor(r) {
  const hue = (1 - clamp(r, 0, 1)) * 120;
  return `hsl(${hue}, 65%, 48%)`;
}

// ---- render root ----------------------------------------------------------
function render(animate) {
  if (!EP) return;
  const t = EP.ticks[IDX];
  $("tickSlider").value = IDX;
  $("tickLabel").textContent = `tick ${t.tick} / ${EP.meta.max_steps}`;
  renderKPIs(t);
  renderHeadline(t);
  renderGraph(t, animate);
  renderInventory(t);
  renderFleet(t);
  renderQueue(t);
  renderNegotiation(t);
  renderAgents(t);
  updateDetail(t);
  moveCursor(t.tick);
}

function renderLegend() {
  const chip = (c, l) => `<span><i style="background:${c}"></i>${l}</span>`;
  const ring = (c, l) => `<span><i class="ring" style="border-color:${c}"></i>${l}</span>`;
  const dia = (c, l) => `<span><i class="diamond" style="background:${c}"></i>${l}</span>`;
  const dash = (c, l) => `<span><i class="dash" style="background-image:repeating-linear-gradient(90deg,${c} 0 3px,transparent 3px 6px)"></i>${l}</span>`;
  $("graphLegend").innerHTML =
    KIND_ORDER.map((k) => chip(KIND_COLOR[k], KIND_LABEL_NODE[k])).join("") +
    dia(INK, "cold-chain shipment") + chip(VEHICLE_COLOR[0], "restock truck") +
    dash("#94a3b8", "scheduled leg") + ring("#dc2626", "disrupted");
}

function kpiTile(cls, label, val, sub) {
  return `<div class="kpi ${cls}"><span class="k-label">${label}</span>` +
    `<span class="k-val">${val}</span><span class="k-sub">${sub || ""}</span></div>`;
}

function renderKPIs(t) {
  const th = EP.meta.thresholds;
  const s = t.shipment;
  const inBand = s.sensor_temp >= th.optimal_temp_low && s.sensor_temp <= th.optimal_temp_high;
  const risk = s.spoilage_risk;
  const unmet = t.inventory.unmet.reduce((a, b) => a + b, 0);
  const demand = t.inventory.demand_today.reduce((a, b) => a + b, 0);
  const nVeh = t.vehicles.length;
  const onTime = nVeh - t.vehicles.filter((v) => v.sla_violated).length;
  const conflicts = t.vehicles.filter((v) => v.conflict).length;
  const problems = liveProblemCount(t);
  const riskCls = risk > 0.5 ? "bad" : risk > 0.25 ? "warn" : "good";

  $("kpibar").innerHTML = [
    kpiTile("", "shipment", EP.meta.fruit.split(".").pop(),
      `${EP.meta.source} → ${EP.meta.target}`),
    kpiTile(riskCls, "spoilage risk", risk.toFixed(2),
      `freshness ${s.freshness_score.toFixed(2)}`),
    kpiTile(inBand ? "good" : "warn", "cargo temp", `${s.sensor_temp.toFixed(1)}°C`,
      `set ${s.desired_temp.toFixed(1)}° · band ${th.optimal_temp_low}–${th.optimal_temp_high}`),
    kpiTile(unmet > 0.01 ? "bad" : "good", "unmet demand", unmet.toFixed(2),
      `demand today ${demand.toFixed(1)}`),
    kpiTile(onTime < nVeh ? "warn" : "good", "on-time fleet", `${onTime}/${nVeh}`,
      conflicts ? `${conflicts} conflict` : "no conflicts"),
    kpiTile("", "energy use", t.energy_usage.toFixed(2), `tick ${t.tick}/${EP.meta.max_steps}`),
    kpiTile(problems ? "bad" : "good", "active problems", problems,
      t.disruptions.length ? `${t.disruptions.length} disruption(s)` : "nominal"),
  ].join("");
}

function renderHeadline(t) {
  const c = t.calendar;
  const a = t.ambient;
  const disr = t.disruptions.length
    ? t.disruptions.map((d) => `${d.type}@${d.target}`).join(", ")
    : "none";
  $("headline").textContent =
    `${WEATHER_ICON[a.weather] || ""} ${a.weather} ${a.temp.toFixed(1)}°C · ` +
    `${WEEKDAYS[c.weekday] || c.weekday} day ${c.day_of_year} · ` +
    `event ×${c.event_multiplier.toFixed(2)} · disruptions: ${disr}`;
}

// ---- fleet mechanics ------------------------------------------------------
// Vehicle state read straight from the sim: a truck carrying an order that is
// still at the depot is waiting for its slot window (scheduled); once it leaves
// the depot it is in transit, stepping one graph edge per tick; no cargo means
// it is idle back at the depot.
function vehicleStatus(t, i) {
  const v = t.vehicles[i];
  const inst = v.carrying;
  if (inst === null || inst === undefined) return { state: "idle", instance: null };
  if (v.current_node === vehicleDepot()) return { state: "scheduled", instance: inst };
  return { state: "transit", instance: inst };
}

// ---- graph ----------------------------------------------------------------
function edgeArrays() {
  const pos = EP._pos;
  const ex = [], ey = [];
  EP.meta.edges.forEach(([u, v]) => {
    ex.push(pos[u][0], pos[v][0], null);
    ey.push(pos[u][1], pos[v][1], null);
  });
  return [ex, ey];
}

// Remaining route of a moving truck: the farm->hub->dc->retail path sliced from
// wherever the truck currently sits.
function remainingPath(v) {
  const path = restockPath(v.carrying);
  const idx = path.indexOf(v.current_node);
  return idx >= 0 ? path.slice(idx) : path;
}

function routeArrays(t) {
  // active delivery legs traced along the real path ahead of each moving truck
  const rx = [], ry = [];
  t.vehicles.forEach((v) => {
    if (vehicleStatus(t, t.vehicles.indexOf(v)).state === "transit") {
      pathSegments(remainingPath(v), rx, ry);
    }
  });
  return [rx, ry];
}

function schedArrays(t) {
  // scheduled legs: a truck holds an order but its slot window has not opened
  // yet (still at the depot). Drawn dashed so it reads as "about to leave".
  const sx = [], sy = [];
  t.vehicles.forEach((v, i) => {
    if (vehicleStatus(t, i).state === "scheduled") {
      pathSegments(restockPath(v.carrying), sx, sy);
    }
  });
  return [sx, sy];
}

// Positions for the WHOLE fleet, read from each truck's real ``current_node`` —
// an in-transit truck sits on the graph node it has reached (one hop per tick);
// idle / scheduled trucks fan out at the shared depot. Trucks sharing a node are
// fanned vertically so all N stay visible.
function fleetPositions(t) {
  const pos = EP._pos, x = [], y = [], c = [], ht = [], tx = [];
  const depot = vehicleDepot();
  const atNode = {};
  t.vehicles.forEach((v) => {
    (atNode[v.current_node] = atNode[v.current_node] || []).push(v);
  });
  t.vehicles.forEach((v, i) => {
    const st = vehicleStatus(t, i);
    const base = pos[v.current_node] || pos[depot];
    const group = atNode[v.current_node];
    const k = group.indexOf(v), n = group.length;
    const off = (k - (n - 1) / 2) * 0.22;
    const px = base[0] + (v.current_node === depot ? -0.34 : 0);
    const py = base[1] + off;
    const node = v.current_node.replace("_", " ");
    const label = st.state === "transit"
      ? `v${i} → R${st.instance} · at ${node}`
      : st.state === "scheduled"
        ? `v${i} ⏱ slot ${v.chosen_slot} → R${st.instance}`
        : `v${i} · idle @ ${node}`;
    x.push(px); y.push(py);
    c.push(VEHICLE_COLOR[i % VEHICLE_COLOR.length]);
    tx.push("v" + i); ht.push(label);
  });
  return { x, y, c, ht, tx };
}

function graphInit(t) {
  const pos = EP._pos;
  const names = Object.keys(pos);
  const [ex, ey] = edgeArrays();
  const ship = t.shipment;

  const traces = [];
  traces[G.EDGE] = { x: ex, y: ey, mode: "lines", type: "scatter", hoverinfo: "skip",
    line: { color: GRID, width: 1.4 } };
  traces[G.SCHED] = { x: [], y: [], mode: "lines", type: "scatter", hoverinfo: "skip",
    line: { color: "#94a3b8", width: 2, dash: "dot" }, opacity: 0.7 };
  traces[G.ROUTE] = { x: [], y: [], mode: "lines", type: "scatter", hoverinfo: "skip",
    line: { color: VEHICLE_COLOR[0], width: 3 }, opacity: 0.55 };
  traces[G.NODE] = {
    x: names.map((n) => pos[n][0]), y: names.map((n) => pos[n][1]),
    mode: "markers+text", type: "scatter",
    text: names.map((n) => n.replace("_", " ")), textposition: "bottom center",
    textfont: { color: "#475569", size: 9 }, hoverinfo: "text", hovertext: names,
    marker: { size: 28, color: names.map((n) => KIND_COLOR[nodeKind(n)]),
              line: { width: names.map(() => 2), color: names.map(() => "#ffffff") } },
  };
  traces[G.TARGET] = { x: [pos[ship.target_node][0]], y: [pos[ship.target_node][1]],
    mode: "markers", type: "scatter", hoverinfo: "skip",
    marker: { size: 44, color: "rgba(0,0,0,0)", line: { width: 2.5, color: INK } } };
  traces[G.CARGO] = { x: [], y: [], mode: "markers+text", type: "scatter",
    hoverinfo: "text", hovertext: [], text: [], textposition: "top center",
    textfont: { color: INK, size: 9 },
    marker: { size: 17, symbol: "square", color: [],
      line: { width: 2, color: "#ffffff" } } };
  traces[G.SHIP] = { x: [pos[ship.current_node][0]], y: [pos[ship.current_node][1]],
    mode: "markers", type: "scatter", hoverinfo: "text",
    hovertext: [`shipment · risk ${ship.spoilage_risk.toFixed(2)}`],
    marker: { size: 22, symbol: "diamond", color: riskColor(ship.spoilage_risk),
              line: { width: 2.5, color: "#ffffff" } } };

  const ymax = Math.max(...names.map((n) => Math.abs(pos[n][1]))) + 1.3;
  Plotly.newPlot("graph", traces, {
    showlegend: false, margin: { l: 8, r: 8, t: 8, b: 8 },
    paper_bgcolor: PLOT_BG, plot_bgcolor: PLOT_BG,
    xaxis: { visible: false, range: [-1.0, 3.9] },
    yaxis: { visible: false, range: [-ymax, ymax] },
    annotations: graphAnnotations(t),
  }, { displayModeBar: false, responsive: true });

  SPREV = [pos[ship.current_node][0], pos[ship.current_node][1]];
  GRAPH_READY = true;
  setCargo(fleetPositions(t)); // draw the whole fleet on first paint
}

function renderGraph(t, animate) {
  if (!GRAPH_READY) { graphInit(t); return; }
  const pos = EP._pos;
  const names = Object.keys(pos);
  const disrupted = new Set(t.disruptions.map((d) => d.target));
  const ship = t.shipment;
  const [rx, ry] = routeArrays(t);
  const [sx, sy] = schedArrays(t);

  Plotly.restyle("graph", { x: [rx], y: [ry] }, [G.ROUTE]);
  Plotly.restyle("graph", { x: [sx], y: [sy] }, [G.SCHED]);
  Plotly.restyle("graph", {
    "marker.line.width": [names.map((n) => (disrupted.has(n) ? 4 : 2))],
    "marker.line.color": [names.map((n) => (disrupted.has(n) ? "#dc2626" : "#ffffff"))],
  }, [G.NODE]);
  Plotly.restyle("graph",
    { x: [[pos[ship.target_node][0]]], y: [[pos[ship.target_node][1]]] }, [G.TARGET]);
  Plotly.restyle("graph",
    { "marker.color": [[riskColor(ship.spoilage_risk)]] }, [G.SHIP]);
  Plotly.relayout("graph", { annotations: graphAnnotations(t) });

  // a respawned shipment teleports to a new farm — snap it there instead of
  // gliding the marker backwards across the whole graph
  const snapShip = t.shipment_no !== undefined && t.shipment_no !== LAST_SHIP_NO;
  LAST_SHIP_NO = t.shipment_no;
  const sTarget = [pos[ship.current_node][0], pos[ship.current_node][1]];
  animateGraph(t, sTarget, animate, snapShip);
}

function setShipmentPos(sPos) {
  Plotly.restyle("graph", { x: [[sPos[0]]], y: [[sPos[1]]] }, [G.SHIP]);
}
function setCargo(cd) {
  Plotly.restyle("graph",
    { x: [cd.x], y: [cd.y], "marker.color": [cd.c], hovertext: [cd.ht], text: [cd.tx] },
    [G.CARGO]);
}

function animateGraph(t, sTarget, animate, snapShip) {
  if (RAF) cancelAnimationFrame(RAF);
  if (!animate) {
    setShipmentPos(sTarget); setCargo(fleetPositions(t));
    SPREV = sTarget; return;
  }
  // Trucks sit on discrete graph nodes (one hop per tick) — place them once;
  // only the cold-chain shipment tweens between nodes.
  setCargo(fleetPositions(t));
  const sFrom = snapShip ? sTarget : SPREV; // snap on respawn, else glide
  const t0 = performance.now();
  function frame(now) {
    const k = clamp((now - t0) / TWEEN_MS, 0, 1);
    const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    setShipmentPos([sFrom[0] + (sTarget[0] - sFrom[0]) * e,
                    sFrom[1] + (sTarget[1] - sFrom[1]) * e]);
    if (k < 1) RAF = requestAnimationFrame(frame);
    else { SPREV = sTarget; RAF = null; }
  }
  RAF = requestAnimationFrame(frame);
}

function graphAnnotations(t) {
  // Every agent type is made legible on the network:
  //   routing            -> the moving shipment diamond (positioned elsewhere)
  //   temperature/spoilage -> a badge riding above the shipment
  //   inventory          -> per-retailer stock/order/unmet badge on retail nodes
  //   delivery           -> the vehicle markers (drawn in the fleet trace)
  const pos = EP._pos;
  const layerY = Math.max(...Object.values(pos).map((p) => Math.abs(p[1]))) + 1.05;
  const span = (c, txt) => `<span style="color:${c}">${txt}</span>`;

  const headers = KIND_ORDER.map((k, col) => ({
    x: col, y: layerY, xanchor: "center", yanchor: "bottom",
    text: KIND_LABEL_NODE[k].toUpperCase(), showarrow: false,
    font: { size: 10, color: KIND_COLOR[k], family: "Inter" },
  }));

  // inventory agents: stock (colour-graded), order, and an unmet-demand flag
  const inv = t.inventory;
  const retail = inv.levels.map((lvl, i) => {
    const p = pos["retail_" + i];
    if (!p) return null;
    const col = lvl < 0.2 ? "#dc2626" : lvl < 0.4 ? "#d97706" : "#16a34a";
    const unmet = inv.unmet[i] > 0.01 ? " · " + span("#dc2626", "⚠" + inv.unmet[i].toFixed(1)) : "";
    return {
      x: p[0], y: p[1], xshift: 30, xanchor: "left", align: "left", showarrow: false,
      text: `R${i} ${span(col, lvl.toFixed(2))} · ord ${inv.order[i].toFixed(1)}${unmet}`,
      font: { size: 9, color: "#94a3b8" },
    };
  }).filter(Boolean);

  // temperature + spoilage agents: a readout that rides above the shipment
  const s = t.shipment;
  const th = EP.meta.thresholds;
  const sp = pos[s.current_node];
  const inBand = s.sensor_temp >= th.optimal_temp_low && s.sensor_temp <= th.optimal_temp_high;
  const shipBadge = {
    x: sp[0], y: sp[1], yshift: 34, xanchor: "center", yanchor: "bottom", showarrow: false,
    text: `🌡 ${span(inBand ? "#16a34a" : "#d97706", s.sensor_temp.toFixed(1) + "°")} · ` +
      `risk ${span(riskColor(s.spoilage_risk), s.spoilage_risk.toFixed(2))} · ` +
      `pred ${t.spoilage_prediction.toFixed(2)}`,
    font: { size: 9, color: "#64748b" },
    bgcolor: "rgba(255,255,255,0.9)", bordercolor: "#e4e9f2", borderwidth: 1, borderpad: 3,
  };

  return [...headers, ...retail, shipBadge];
}

// ---- side panels ----------------------------------------------------------
function renderInventory(t) {
  const inv = t.inventory;
  let ih = `<table><thead><tr><th>retailer</th><th>stock</th><th>order</th>` +
           `<th>demand</th><th>unmet</th></tr></thead><tbody>`;
  inv.levels.forEach((lvl, i) => {
    const pct = clamp(Math.round(lvl * 100), 0, 100);
    const barCls = lvl < 0.2 ? "low" : lvl < 0.4 ? "mid" : "";
    ih += `<tr><td>R${i}</td>` +
      `<td><span class="stockcell"><span class="bar ${barCls}" style="--p:${pct}%"></span>` +
      `${lvl.toFixed(2)}</span></td>` +
      `<td>${inv.order[i].toFixed(2)}</td><td>${inv.demand_today[i].toFixed(2)}</td>` +
      `<td class="${inv.unmet[i] > 0.01 ? "flag-bad" : ""}">${inv.unmet[i].toFixed(2)}</td></tr>`;
  });
  ih += `</tbody></table>`;
  $("invTable").innerHTML = ih;
}

const STATUS_LABEL = {
  transit: "en route", scheduled: "waiting slot", idle: "idle",
};

function renderFleet(t) {
  const box = $("fleet");
  box.innerHTML = "";
  const depot = vehicleDepot().replace("_", " ");
  t.vehicles.forEach((v, i) => {
    const st = vehicleStatus(t, i);
    const color = VEHICLE_COLOR[i % VEHICLE_COLOR.length];
    const here = v.current_node.replace("_", " ");
    const dest = st.state === "idle"
      ? `parked @ ${depot}`
      : st.state === "scheduled"
        ? `${depot} → R${st.instance} · awaiting slot ${v.chosen_slot}`
        : `→ R${st.instance} · at ${here}`;
    let prog = "";
    if (st.state === "transit") {
      const rem = restockPath(v.carrying);
      const idx = rem.indexOf(v.current_node);
      const p = idx >= 0 && rem.length > 1 ? idx / (rem.length - 1) : 0;
      prog = `<div class="vprog" style="--c:${color}"><span style="--p:${Math.round(p * 100)}%"></span></div>`;
    }
    const flags = [];
    if (v.conflict) flags.push(`<span class="tag bad">conflict</span>`);
    if (v.sla_violated) flags.push(`<span class="tag bad">SLA</span>`);
    if (v.delay > 0.01) flags.push(`<span class="tag warn">delay ${v.delay.toFixed(1)}</span>`);
    const el = document.createElement("div");
    el.className = "veh state-" + st.state;
    el.innerHTML =
      `<div class="vhead"><span class="vdot" style="background:${color}"></span>` +
      `<span class="vname">v${i}</span>` +
      `<span class="vstate">${STATUS_LABEL[st.state]}</span>` +
      `<span class="vslot">slot ${v.chosen_slot}</span></div>` +
      `<div class="vdest">${dest}</div>` + prog +
      (flags.length ? `<div class="vflags">${flags.join("")}</div>` : "");
    box.appendChild(el);
  });
}

function renderQueue(t) {
  const q = t.order_queue || [];
  $("queueCount").textContent = q.length ? q.length : "";
  const box = $("queue");
  if (!q.length) {
    box.innerHTML = `<div class="empty">— no orders waiting for a vehicle —</div>`;
    return;
  }
  const cap = 12;
  let html = q.slice(0, cap)
    .map(([i, qty], k) =>
      `<span class="qchip"><b>#${k + 1}</b> R${i} · ${qty.toFixed(2)}</span>`)
    .join("");
  if (q.length > cap) html += `<span class="qchip">+${q.length - cap} more</span>`;
  box.innerHTML = html;
}

// ---- LLM slot negotiation -------------------------------------------------
function renderNegotiation(t) {
  const mode = (EP.meta && EP.meta.mediator) || "off";
  $("negoMediator").textContent = mode === "off" ? "mediator off" : `${mode} mediator · Alg 6`;
  const box = $("negotiation");
  const events = t.negotiations || [];
  $("negoCount").textContent = events.length ? events.length : "";
  $("negoCard").classList.toggle("active", events.length > 0);

  if (mode === "off") {
    box.innerHTML = `<div class="empty">— mediator disabled for this run —</div>`;
    return;
  }
  if (!events.length) {
    box.innerHTML = `<div class="empty">— no slot conflict this tick · fleet slots clear —</div>`;
    return;
  }
  box.innerHTML = events.map(negoEvent).join("");
}

function negoEvent(ev) {
  const changed = (a) => ev.initial[a] !== ev.final[a];
  const rows = ev.agents.map((a) => {
    const from = ev.initial[a], to = ev.final[a];
    const arrow = changed(a)
      ? `slot ${from} → <b class="moved">${to}</b>`
      : `slot <b>${to}</b> <span class="kept">(kept)</span>`;
    return `<div class="nrow"><span class="vdot" style="background:${
      VEHICLE_COLOR[(+a.split("_")[1] || 0) % VEHICLE_COLOR.length]}"></span>` +
      `<span class="nname">${a}</span><span class="nslot">${arrow}</span></div>`;
  }).join("");
  const badge = ev.resolved
    ? `<span class="tag good">resolved · ${ev.rounds} round${ev.rounds === 1 ? "" : "s"}</span>`
    : `<span class="tag bad">unresolved · penalty stands</span>`;
  const summary = ev.summary
    ? `<div class="nsummary">“${escapeHtml(ev.summary)}”</div>`
    : "";
  return `<div class="nevent ${ev.resolved ? "ok" : "fail"}">` +
    `<div class="nhead"><span class="ncontest">⚔ slot ${ev.slot} contested by ${ev.agents.length} trucks</span>${badge}</div>` +
    rows + summary + `</div>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// ---- agents ---------------------------------------------------------------
function renderAgents(t) {
  const box = $("agents");
  box.innerHTML = "";
  AGENT_KINDS.forEach((kind) => {
    EP._agents
      .filter((a) => agentKind(a) === kind)
      .forEach((name) => box.appendChild(agentCard(kind, name, t)));
  });
}

function agentCard(kind, name, t) {
  const reward = t.rewards ? t.rewards[name] : undefined;
  const info = (t.infos && t.infos[name]) || {};
  const action = t.actions ? t.actions[name] : undefined;
  const rcls = reward === undefined ? "" : reward >= 0 ? "pos" : "neg";

  const metrics = (METRICS[kind] || [])
    .filter(([, key]) => info[key] !== undefined)
    .map(([label, key]) => {
      const v = info[key];
      const bad = (key === "sla_violated" || key === "conflict" || key === "fn_rate") && v > 0;
      return `<span>${label} <b class="${bad ? "flag-bad" : ""}">${fmt(v)}</b></span>`;
    })
    .join("");

  const el = document.createElement("div");
  el.className = "agent " + kind + (name === SELECTED ? " selected" : "");
  el.innerHTML =
    `<div class="ahead"><span class="aname">${name}</span>` +
    `<span class="areward ${rcls}">r ${reward === undefined ? "–" : fmt(reward, 2)}</span></div>` +
    `<span class="akind">${KIND_LABEL[kind]}</span>` +
    `<div class="aaction">${actionText(kind, action)}</div>` +
    `<div class="ametrics">${metrics || "<span>—</span>"}</div>`;
  el.addEventListener("click", () => openDetail(name));
  return el;
}

function actionText(kind, action) {
  if (action === undefined) return "action: –";
  if (Array.isArray(action)) action = action.length === 1 ? action[0] : action;
  if (kind === "routing") return `action: next-node ${action}`;
  if (kind === "delivery") return `action: slot ${action}`;
  if (Array.isArray(action)) return "action: [" + action.map((x) => fmt(x)).join(", ") + "]";
  return "action: " + fmt(action);
}

// ---- agent detail drawer --------------------------------------------------
function openDetail(name) {
  SELECTED = name;
  $("detail").hidden = false;
  document.body.classList.add("has-detail");
  buildDetailCharts();
  if (EP) { updateDetail(EP.ticks[IDX]); moveCursor(EP.ticks[IDX].tick); }
  renderAgents(EP.ticks[IDX]); // refresh selected highlight
}

function closeDetail() {
  const t = EP ? EP.ticks[IDX] : null;
  SELECTED = null;
  $("detail").hidden = true;
  document.body.classList.remove("has-detail");
  if (t) renderAgents(t);
}

function updateDetail(t) {
  if (!SELECTED) return;
  const kind = agentKind(SELECTED);
  const reward = t.rewards ? t.rewards[SELECTED] : undefined;
  const action = t.actions ? t.actions[SELECTED] : undefined;
  const info = (t.infos && t.infos[SELECTED]) || {};
  $("detailName").textContent = SELECTED;
  $("detailKind").textContent = KIND_LABEL[kind] || kind;

  const rcls = reward === undefined ? "" : reward >= 0 ? "pos" : "neg";
  const rows = Object.entries(info)
    .map(([k, v]) => {
      const bad = (k === "sla_violated" || k === "conflict" || k === "fn_rate") && v > 0;
      return `<div class="drow"><span>${k}</span><b class="${bad ? "flag-bad" : ""}">${fmt(v)}</b></div>`;
    })
    .join("");
  $("detailBody").innerHTML =
    `<div class="dstats">` +
      `<div class="dstat"><span class="k-label">reward</span>` +
        `<span class="k-val ${rcls}">${reward === undefined ? "–" : fmt(reward, 3)}</span></div>` +
      `<div class="dstat"><span class="k-label">action</span>` +
        `<span class="k-val">${actionText(kind, action).replace("action: ", "")}</span></div>` +
    `</div>` +
    `<div class="drows">${rows || '<div class="empty">no info this tick</div>'}</div>`;
}

function buildDetailCharts() {
  if (!SELECTED || !EP) return;
  const ticks = EP.ticks, xs = ticks.map((t) => t.tick);
  const cfg = { displayModeBar: false, responsive: true };

  Plotly.newPlot("detailChartReward", [
    line(xs, ticks.map((t) => (t.rewards ? t.rewards[SELECTED] : null)), "reward", "#2563eb"),
  ], chartLayout([]), cfg);

  const keys = new Set();
  ticks.forEach((t) => {
    const inf = t.infos && t.infos[SELECTED];
    if (inf) Object.entries(inf).forEach(([k, v]) => { if (typeof v === "number") keys.add(k); });
  });
  const traces = [...keys].map((k, i) =>
    line(xs, ticks.map((t) => {
      const inf = t.infos && t.infos[SELECTED];
      return inf && inf[k] !== undefined ? inf[k] : null;
    }), k, AGENT_PALETTE[i % AGENT_PALETTE.length]));
  Plotly.newPlot("detailChartMetric",
    traces.length ? traces : [line(xs, xs.map(() => null), "—", "#94a3b8")],
    chartLayout([]), cfg);
}

// ---- charts ---------------------------------------------------------------
function chartLayout(shapes) {
  return {
    margin: { l: 34, r: 8, t: 6, b: 20 },
    paper_bgcolor: PLOT_BG, plot_bgcolor: PLOT_BG, font: PLOT_FONT,
    showlegend: true, legend: { orientation: "h", y: 1.28, font: { size: 9, color: "#64748b" } },
    xaxis: { gridcolor: GRID, zeroline: false, linecolor: GRID, tickfont: { color: "#94a3b8" } },
    yaxis: { gridcolor: GRID, zeroline: false, linecolor: GRID, tickfont: { color: "#94a3b8" } },
    shapes: shapes || [],
  };
}

function buildCharts() {
  const ticks = EP.ticks;
  const xs = ticks.map((t) => t.tick);
  const th = EP.meta.thresholds;
  const cfg = { displayModeBar: false, responsive: true };

  const tempShapes = [band(th.optimal_temp_low, th.optimal_temp_high, "#57b083")];
  if (th.chill_injury !== null) tempShapes.push(hline(th.chill_injury, "#6b93d6"));
  BASE_SHAPES.temp = tempShapes;
  Plotly.newPlot("chartTemp", [
    line(xs, ticks.map((t) => t.shipment.sensor_temp), "sensor", "#e0574f"),
    line(xs, ticks.map((t) => t.shipment.desired_temp), "setpoint", "#b07ad9", "dash"),
    line(xs, ticks.map((t) => t.ambient.temp), "ambient", "#8493ad", "dot"),
  ], chartLayout(tempShapes), cfg);

  BASE_SHAPES.spoil = [hline(0.5, "#8493ad")];
  Plotly.newPlot("chartSpoil", [
    line(xs, ticks.map((t) => t.shipment.spoilage_risk), "risk", "#e0574f"),
    line(xs, ticks.map((t) => t.shipment.freshness_score), "freshness", "#57b083"),
    line(xs, ticks.map((t) => t.spoilage_prediction), "pred", "#b07ad9", "dot"),
  ], chartLayout(BASE_SHAPES.spoil), cfg);

  BASE_SHAPES.reward = [];
  const rewardTraces = EP._agents.map((a, i) =>
    line(xs, ticks.map((t) => (t.rewards ? t.rewards[a] : null)), a,
         AGENT_PALETTE[i % AGENT_PALETTE.length]));
  Plotly.newPlot("chartReward", rewardTraces, chartLayout([]), cfg);
  if (SELECTED) buildDetailCharts();
}

const AGENT_PALETTE = ["#2563eb", "#0891b2", "#db2777", "#d97706", "#16a34a",
  "#7c3aed", "#0d9488", "#ca8a04", "#e11d48", "#475569", "#9333ea"];

function line(x, y, name, color, dash) {
  return { x, y, name, mode: "lines+markers", type: "scatter",
           marker: { size: 3 }, line: { color, width: 1.6, dash: dash || "solid" } };
}
function band(lo, hi, color) {
  return { type: "rect", xref: "paper", x0: 0, x1: 1, y0: lo, y1: hi,
           fillcolor: color, opacity: 0.12, line: { width: 0 } };
}
function hline(y, color) {
  return { type: "line", xref: "paper", x0: 0, x1: 1, y0: y, y1: y,
           line: { color, width: 1, dash: "dot" } };
}
function cursor(x) {
  return { type: "line", yref: "paper", x0: x, x1: x, y0: 0, y1: 1,
           line: { color: "#94a3b8", width: 1.4, dash: "dot" } };
}
function moveCursor(tick) {
  const c = cursor(tick);
  Plotly.relayout("chartTemp", { shapes: BASE_SHAPES.temp.concat(c) });
  Plotly.relayout("chartSpoil", { shapes: BASE_SHAPES.spoil.concat(c) });
  Plotly.relayout("chartReward", { shapes: BASE_SHAPES.reward.concat(c) });
  if (SELECTED && !$("detail").hidden) {
    Plotly.relayout("detailChartReward", { shapes: [c] });
    Plotly.relayout("detailChartMetric", { shapes: [c] });
  }
}

// ---- playback -------------------------------------------------------------
function step(delta, animate) {
  if (!EP) return;
  IDX = clamp(IDX + delta, 0, EP.ticks.length - 1);
  render(animate);
}
function togglePlay() {
  if (TIMER) return stopPlay();
  $("playBtn").textContent = "❚❚ pause";
  TIMER = setInterval(() => {
    if (IDX >= EP.ticks.length - 1) return stopPlay();
    step(1, true);
  }, 900);
}
function stopPlay() {
  if (TIMER) clearInterval(TIMER);
  TIMER = null;
  $("playBtn").textContent = "▶ play";
}

// ---- wire up --------------------------------------------------------------
function init() {
  $("episodeSelect").addEventListener("change", (e) => loadEpisode(e.target.value));
  $("runBtn").addEventListener("click", runEpisode);
  $("liveBtn").addEventListener("click", toggleLive);
  $("prevBtn").addEventListener("click", () => { stopPlay(); step(-1); });
  $("nextBtn").addEventListener("click", () => { stopPlay(); step(1); });
  $("playBtn").addEventListener("click", togglePlay);
  $("tickSlider").addEventListener("input", (e) => {
    stopPlay(); IDX = Number(e.target.value); render();
  });
  $("detailClose").addEventListener("click", closeDetail);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && SELECTED) closeDetail(); });
  boot();
}

// Paint a recorded episode instantly (if any) for an immediate view, then roll
// a live inference on top so the dashboard is animated the moment it opens.
async function boot() {
  const episodes = await fetchEpisodes();
  if (episodes.length) {
    $("episodeSelect").value = episodes[0];
    await loadEpisode(episodes[0]);
  } else {
    setStatus("no recorded episodes — starting live");
  }
  startLive();
}

init();
