"use strict";

const KIND_ORDER = ["farm", "hub", "dc", "retail"];
const KIND_COLOR = { farm: "#4c9f70", hub: "#6689c9", dc: "#c9a14a", retail: "#c96666" };
const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const PLOT_BG = "#182031";
const PLOT_FONT = { color: "#8fa0bd", size: 10 };

let EP = null;        // {name, meta, ticks, _pos, _agents}
let IDX = 0;
let TIMER = null;
let BASE_SHAPES = {}; // per-chart constant shapes (bands, thresholds)

const $ = (id) => document.getElementById(id);

// ---- data loading ---------------------------------------------------------
async function fetchEpisodes() {
  const r = await fetch("/api/episodes");
  const { episodes } = await r.json();
  const sel = $("episodeSelect");
  sel.innerHTML = "";
  episodes.forEach((name) => {
    const o = document.createElement("option");
    o.value = o.textContent = name;
    sel.appendChild(o);
  });
  if (episodes.length) loadEpisode(episodes[0]);
  else setStatus("no episodes — run one");
}

async function loadEpisode(name) {
  setStatus("loading " + name + " …");
  const r = await fetch("/api/episode/" + encodeURIComponent(name));
  const data = await r.json();
  if (data.error) { setStatus(data.error); return; }
  setEpisode(data);
  setStatus(name + " · " + EP.ticks.length + " ticks");
}

async function runEpisode() {
  const body = {
    seed: Number($("runSeed").value) || 90000,
    tag: $("runTag").value.trim(),
    scenario: $("runScenario").value.trim(),
    max_steps: $("runMaxSteps").value.trim(),
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
    const sel = $("episodeSelect");
    sel.value = data.name;
    setEpisode(data);
    setStatus("ran " + data.name + " · " + EP.ticks.length + " ticks");
  } catch (e) {
    setStatus("request failed: " + e);
  } finally {
    $("runBtn").disabled = false;
  }
}

function setEpisode(data) {
  stopPlay();
  data._pos = layout(data.meta);
  data._agents = agentList(data.ticks);
  EP = data;
  IDX = 0;
  const slider = $("tickSlider");
  slider.max = EP.ticks.length - 1;
  slider.value = 0;
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

function agentList(ticks) {
  const seen = new Set();
  ticks.forEach((t) => {
    Object.keys(t.rewards || {}).forEach((a) => seen.add(a));
    Object.keys(t.infos || {}).forEach((a) => seen.add(a));
  });
  const order = { temperature: 0, routing: 1, spoilage: 2, inventory: 3, delivery: 4 };
  return [...seen].sort((a, b) => {
    const ka = order[agentKind(a)], kb = order[agentKind(b)];
    return ka !== kb ? ka - kb : a.localeCompare(b);
  });
}

const agentKind = (a) => a.replace(/_\d+$/, "");
const fmt = (v, d = 3) => (typeof v === "number" ? v.toFixed(d) : v);
const setStatus = (s) => { $("status").textContent = s; };

function riskColor(r) {
  r = Math.max(0, Math.min(1, r));
  const hue = (1 - r) * 120; // 120=green, 0=red
  return `hsl(${hue}, 70%, 45%)`;
}

// ---- render ---------------------------------------------------------------
function render() {
  if (!EP) return;
  const t = EP.ticks[IDX];
  $("tickSlider").value = IDX;
  $("tickLabel").textContent = `tick ${t.tick} / ${EP.meta.max_steps}`;
  renderHeadline(t);
  renderGraph(t);
  renderSysparams(t);
  renderAgents(t);
  moveCursor(t.tick);
}

function renderHeadline(t) {
  const a = t.ambient, c = t.calendar;
  const disr = t.disruptions.length
    ? t.disruptions.map((d) => `${d.type}@${d.target}`).join(", ")
    : "none";
  $("headline").textContent =
    `${EP.meta.fruit.split(".").pop()} · ${EP.meta.source}→${EP.meta.target} · ` +
    `${a.weather} ${a.temp.toFixed(1)}°C · ${WEEKDAYS[c.weekday] || c.weekday} ` +
    `day ${c.day_of_year} · disruptions: ${disr}`;
}

function renderGraph(t) {
  const pos = EP._pos;
  const disrupted = new Set(t.disruptions.map((d) => d.target));
  const ex = [], ey = [];
  EP.meta.edges.forEach(([u, v]) => {
    ex.push(pos[u][0], pos[v][0], null);
    ey.push(pos[u][1], pos[v][1], null);
  });
  const names = Object.keys(pos);
  const nodeTrace = {
    x: names.map((n) => pos[n][0]),
    y: names.map((n) => pos[n][1]),
    mode: "markers+text",
    type: "scatter",
    text: names,
    textposition: "bottom center",
    textfont: { color: "#8fa0bd", size: 9 },
    hovertext: names.map((n) => n),
    hoverinfo: "text",
    marker: {
      size: 26,
      color: names.map((n) => KIND_COLOR[nodeKind(n)]),
      line: {
        width: names.map((n) => (disrupted.has(n) ? 3 : 1)),
        color: names.map((n) => (disrupted.has(n) ? "#d9524a" : "#0f1420")),
      },
    },
  };
  const ship = t.shipment;
  const [tx, ty] = pos[ship.target_node];
  const targetTrace = {
    x: [tx], y: [ty], mode: "markers", type: "scatter", hoverinfo: "skip",
    marker: { size: 40, color: "rgba(0,0,0,0)", line: { width: 2, color: "#dfe6f2" } },
  };
  const [cx, cy] = pos[ship.current_node];
  const cargoTrace = {
    x: [cx], y: [cy], mode: "markers", type: "scatter",
    hovertext: [`cargo · risk ${ship.spoilage_risk.toFixed(2)}`], hoverinfo: "text",
    marker: { size: 16, symbol: "diamond", color: riskColor(ship.spoilage_risk),
              line: { width: 1.5, color: "#0f1420" } },
  };
  const edgeTrace = {
    x: ex, y: ey, mode: "lines", type: "scatter", hoverinfo: "skip",
    line: { color: "#2c3650", width: 1 },
  };
  // vehicle annotations near their assigned retailers
  const ann = t.vehicles.map((v, i) => {
    const [rx, ry] = pos[v.assigned_node];
    const flag = v.sla_violated ? " !" : v.conflict ? " ×" : "";
    return {
      x: rx + 0.28, y: ry - 0.35 - 0.22 * i, xanchor: "left",
      text: `v${i} s${v.chosen_slot}${flag}`, showarrow: false,
      font: { size: 9, color: v.sla_violated ? "#d9524a" : v.conflict ? "#e0a63a" : "#8fa0bd" },
    };
  });
  const ymax = Math.max(...names.map((n) => Math.abs(pos[n][1]))) + 1.2;
  Plotly.react("graph",
    [edgeTrace, nodeTrace, targetTrace, cargoTrace],
    {
      showlegend: false, margin: { l: 8, r: 8, t: 8, b: 8 },
      paper_bgcolor: PLOT_BG, plot_bgcolor: PLOT_BG,
      xaxis: { visible: false, range: [-0.6, 3.9] },
      yaxis: { visible: false, range: [-ymax, ymax] },
      annotations: ann,
    },
    { displayModeBar: false, responsive: true });
}

const nodeKind = (n) => n.replace(/_\d+$/, "");

function renderSysparams(t) {
  const a = t.ambient, c = t.calendar, inv = t.inventory;
  const kv = (k, v) => `<div class="kv"><span>${k}</span><span>${v}</span></div>`;
  let html = "";
  html += kv("weather", a.weather);
  html += kv("ambient °C", a.temp.toFixed(1));
  html += kv("ambient RH", a.humidity.toFixed(2));
  html += kv("event ×", c.event_multiplier.toFixed(2));
  html += kv("energy", t.energy_usage.toFixed(3));
  html += kv("spoil pred", t.spoilage_prediction.toFixed(3));

  html += `<div class="full"><table><tr><th>retailer</th><th>stock</th><th>order</th>` +
          `<th>demand</th><th>unmet</th></tr>`;
  inv.levels.forEach((_, i) => {
    html += `<tr><td>R${i}</td><td>${inv.levels[i].toFixed(2)}</td>` +
      `<td>${inv.order[i].toFixed(2)}</td><td>${inv.demand_today[i].toFixed(2)}</td>` +
      `<td class="${inv.unmet[i] > 0 ? "flag-warn" : ""}">${inv.unmet[i].toFixed(2)}</td></tr>`;
  });
  html += `</table></div>`;

  html += `<div class="full"><table><tr><th>cargo→</th><th>veh</th><th>arrive</th><th>qty</th></tr>`;
  if (t.cargo.length) {
    t.cargo.forEach((c2) => {
      html += `<tr><td>R${c2.instance}</td><td>v${c2.vehicle}</td>` +
        `<td>t${c2.arrival_tick}</td><td>${c2.qty.toFixed(2)}</td></tr>`;
    });
  } else {
    html += `<tr><td colspan="4" style="color:#8fa0bd">— none in transit —</td></tr>`;
  }
  html += `</table></div>`;
  $("sysparams").innerHTML = html;
}

function renderAgents(t) {
  const box = $("agents");
  box.innerHTML = "";
  EP._agents.forEach((name) => {
    const kind = agentKind(name);
    const reward = t.rewards ? t.rewards[name] : undefined;
    const info = (t.infos && t.infos[name]) || {};
    const action = t.actions ? t.actions[name] : undefined;

    const rcls = reward === undefined ? "" : reward >= 0 ? "pos" : "neg";
    const metrics = Object.entries(info)
      .map(([k, v]) => {
        let cls = "";
        if ((k === "sla_violated" || k === "conflict" || k === "fn_rate") && v > 0)
          cls = "flag-bad";
        return `<span>${k} <b class="${cls}">${fmt(v)}</b></span>`;
      })
      .join("");

    const el = document.createElement("div");
    el.className = "agent " + kind;
    el.innerHTML =
      `<div class="ahead"><span class="aname">${name}</span>` +
      `<span class="areward ${rcls}">r ${reward === undefined ? "–" : fmt(reward)}</span></div>` +
      `<div class="aaction">action: ${actionText(name, action)}</div>` +
      `<div class="ametrics">${metrics || "<span>—</span>"}</div>`;
    box.appendChild(el);
  });
}

function actionText(name, action) {
  if (action === undefined) return "–";
  const kind = agentKind(name);
  if (Array.isArray(action)) action = action.length === 1 ? action[0] : action;
  if (kind === "routing") return `next-node idx ${action}`;
  if (kind === "delivery") return `slot ${action}`;
  if (Array.isArray(action)) return "[" + action.map((x) => fmt(x)).join(", ") + "]";
  return fmt(action);
}

// ---- charts ---------------------------------------------------------------
function chartLayout(extraShapes) {
  return {
    margin: { l: 34, r: 8, t: 6, b: 20 },
    paper_bgcolor: PLOT_BG, plot_bgcolor: PLOT_BG, font: PLOT_FONT,
    showlegend: true, legend: { orientation: "h", y: 1.25, font: { size: 9 } },
    xaxis: { gridcolor: "#2c3650", zeroline: false },
    yaxis: { gridcolor: "#2c3650", zeroline: false },
    shapes: extraShapes || [],
  };
}

function buildCharts() {
  const ticks = EP.ticks;
  const xs = ticks.map((t) => t.tick);
  const th = EP.meta.thresholds;
  const cfg = { displayModeBar: false, responsive: true };

  // temperature
  const tempTraces = [
    line(xs, ticks.map((t) => t.shipment.sensor_temp), "sensor", "#d9524a"),
    line(xs, ticks.map((t) => t.shipment.desired_temp), "setpoint", "#b07ad9", "dash"),
    line(xs, ticks.map((t) => t.ambient.temp), "ambient", "#8fa0bd", "dot"),
  ];
  const tempShapes = [band(th.optimal_temp_low, th.optimal_temp_high, "#4c9f70")];
  if (th.chill_injury !== null) tempShapes.push(hline(th.chill_injury, "#6689c9"));
  BASE_SHAPES.temp = tempShapes;
  Plotly.newPlot("chartTemp", tempTraces, chartLayout(tempShapes), cfg);

  // spoilage
  const spoilTraces = [
    line(xs, ticks.map((t) => t.shipment.spoilage_risk), "risk", "#d9524a"),
    line(xs, ticks.map((t) => t.shipment.freshness_score), "freshness", "#4c9f70"),
    line(xs, ticks.map((t) => t.spoilage_prediction), "pred", "#b07ad9", "dot"),
  ];
  BASE_SHAPES.spoil = [hline(0.5, "#8fa0bd")];
  Plotly.newPlot("chartSpoil", spoilTraces, chartLayout(BASE_SHAPES.spoil), cfg);

  // rewards per agent
  const rewardTraces = EP._agents.map((a, i) =>
    line(xs, ticks.map((t) => (t.rewards ? t.rewards[a] : null)), a,
         AGENT_PALETTE[i % AGENT_PALETTE.length]));
  BASE_SHAPES.reward = [];
  Plotly.newPlot("chartReward", rewardTraces, chartLayout([]), cfg);
}

const AGENT_PALETTE = ["#5aa9ff", "#6689c9", "#c96666", "#c9a14a", "#e0a63a",
  "#d9a14a", "#4c9f70", "#5fbf8f", "#7fdfaf", "#8fa0bd", "#b07ad9"];

function line(x, y, name, color, dash) {
  return { x, y, name, mode: "lines+markers", type: "scatter",
           marker: { size: 3 }, line: { color, width: 1.6, dash: dash || "solid" } };
}
function band(lo, hi, color) {
  return { type: "rect", xref: "paper", x0: 0, x1: 1, y0: lo, y1: hi,
           fillcolor: color, opacity: 0.15, line: { width: 0 } };
}
function hline(y, color) {
  return { type: "line", xref: "paper", x0: 0, x1: 1, y0: y, y1: y,
           line: { color, width: 1, dash: "dot" } };
}
function cursor(x) {
  return { type: "line", yref: "paper", x0: x, x1: x, y0: 0, y1: 1,
           line: { color: "#dfe6f2", width: 1.4 } };
}

function moveCursor(tick) {
  const c = cursor(tick);
  Plotly.relayout("chartTemp", { shapes: BASE_SHAPES.temp.concat(c) });
  Plotly.relayout("chartSpoil", { shapes: BASE_SHAPES.spoil.concat(c) });
  Plotly.relayout("chartReward", { shapes: BASE_SHAPES.reward.concat(c) });
}

// ---- playback -------------------------------------------------------------
function step(delta) {
  if (!EP) return;
  IDX = Math.max(0, Math.min(EP.ticks.length - 1, IDX + delta));
  render();
}
function togglePlay() {
  if (TIMER) return stopPlay();
  $("playBtn").textContent = "❚❚ pause";
  TIMER = setInterval(() => {
    if (IDX >= EP.ticks.length - 1) return stopPlay();
    step(1);
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
  $("prevBtn").addEventListener("click", () => { stopPlay(); step(-1); });
  $("nextBtn").addEventListener("click", () => { stopPlay(); step(1); });
  $("playBtn").addEventListener("click", togglePlay);
  $("tickSlider").addEventListener("input", (e) => {
    stopPlay(); IDX = Number(e.target.value); render();
  });
  fetchEpisodes();
}

init();
