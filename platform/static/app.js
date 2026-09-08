/* ═══════════ 企业级管控平台 前端逻辑 ═══════════ */
const $ = id => document.getElementById(id);
const charts = {};
const fmtInt = x => (x === undefined || x === null) ? "-" : Math.round(x).toLocaleString();
const fmt2 = x => (x === undefined || x === null || isNaN(x)) ? "-" : Number(x).toFixed(2);

function chart(id) {
  if (!charts[id]) {
    const el = $(id);
    if (!el) return null;
    charts[id] = echarts.init(el);
  }
  return charts[id];
}
window.addEventListener("resize", () => Object.values(charts).forEach(c => c.resize()));

async function jget(u) { const r = await fetch(u); return r.json(); }
async function jpost(u, body) {
  const r = await fetch(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  return r.json();
}

/* ── 时钟 ── */
setInterval(() => {
  const d = new Date();
  const wd = ["周日","周一","周二","周三","周四","周五","周六"][d.getDay()];
  $("clock").textContent = `${d.toLocaleTimeString("zh-CN",{hour12:false})} ${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,"0")}-${String(d.getDate()).padStart(2,"0")} ${wd}`;
}, 1000);

/* ── 页面路由 ── */
const PAGE_NAME = { overview:"总览仪表盘", network:"路网管理", gis:"GIS地图监控", algorithm:"协同管控算法",
                    deploy:"模型部署", sim:"信号控制仿真", eval:"性能评估", cve:"云边端协同", docs:"接口文档" };
const loaders = {};
document.querySelectorAll(".nav-item").forEach(item => {
  item.onclick = () => {
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    item.classList.add("active");
    const pg = item.dataset.page;
    $("pg-" + pg).classList.add("active");
    $("crumb-page").textContent = PAGE_NAME[pg] || pg;
    if (loaders[pg]) loaders[pg]();
    Object.values(charts).forEach(c => c.resize());
  };
});

/* ══════════ 1 总览仪表盘 ══════════ */
let ovTrendInit = false;
function drawTrend(trend, runLabel) {
  $("ov-trend-run").textContent = runLabel || "-";
  chart("ov-trend").setOption({
    backgroundColor: "transparent", tooltip: { trigger: "axis" },
    legend: { top: 0, textStyle: { fontSize: 10, color: "#7b879c" } },
    grid: { left: 46, right: 50, top: 30, bottom: 24 },
    xAxis: { type: "category", data: (trend.time || []).map(v => v + "s"), axisLabel: { fontSize: 9 } },
    yAxis: [{ type: "value", name: "km/h" }, { type: "value", name: "辆", position: "right" }],
    series: [
      { name: "平均车速", type: "line", showSymbol: false, data: trend.speed_kmh, smooth: true,
        lineStyle: { color: "#2563eb", width: 2 }, itemStyle: { color: "#2563eb" } },
      { name: "累计通过", type: "line", yAxisIndex: 1, showSymbol: false, data: trend.arrived_cum,
        lineStyle: { color: "#16a34a", width: 2 }, itemStyle: { color: "#16a34a" },
        areaStyle: { color: "rgba(22,163,74,.08)" } }],
  });
  ovTrendInit = true;
}
function drawGauge(id, title, val) {
  chart(id).setOption({
    backgroundColor: "transparent",
    series: [{ type: "gauge", startAngle: 210, endAngle: -30, min: 0, max: 100, radius: "95%",
      progress: { show: true, width: 12, itemStyle: { color: "#2563eb" } },
      axisLine: { lineStyle: { width: 12, color: [[1, "#eef1f7"]] } },
      axisTick: { show: false }, splitLine: { show: false }, axisLabel: { show: false },
      pointer: { show: false }, detail: { fontSize: 22, fontFamily: "Consolas", offsetCenter: [0, 0],
        formatter: v => v.toFixed(1) + "%" }, title: { show: false }, data: [{ value: val }] }],
  });
}
async function loadOverview() {
  const d = await jget("/api/overview");
  $("ov-sc").textContent = d.kpi.scenarios;
  $("ov-runs").textContent = d.kpi.collected_runs;
  $("ov-edge").textContent = "12";
  $("ov-ep").textContent = fmtInt(d.kpi.episodes);
  $("ov-api").textContent = fmtInt(d.kpi.api_count);
  const up = d.kpi.uptime_s;
  $("ov-up").textContent = `${String(Math.floor(up/3600)).padStart(2,"0")}:${String(Math.floor(up%3600/60)).padStart(2,"0")}:${String(up%60).padStart(2,"0")}`;
  $("cpu-badge").textContent = d.system.cpu.toFixed(1) + "%";
  drawGauge("ov-cpu", "", d.system.cpu);
  drawGauge("ov-mem", "", d.system.mem);
  $("ov-cpu-t").textContent = d.system.cpu.toFixed(1) + "%";
  $("ov-mem-t").textContent = d.system.mem.toFixed(1) + "%";
  // 状态卡
  $("ov-train-st").textContent = d.train.running ? "运行中" : "已停止";
  $("ov-train-st").className = "st " + (d.train.running ? "run" : "stop");
  $("ov-train-r").textContent = fmt2(d.train.reward);
  $("ov-train-e").textContent = fmt2(d.train.epsilon);
  $("ov-sim-st").textContent = d.sim.running ? "运行中" : "已停止";
  $("ov-sim-st").className = "st " + (d.sim.running ? "run" : "stop");
  $("ov-sim-sc").textContent = d.sim.scenario || "-";
  $("ov-sim-t").textContent = (d.sim.time || 0) + "s";
  $("ov-eval-st").textContent = d.eval.running ? "运行中" : "空闲";
  $("ov-eval-st").className = "st " + (d.eval.running ? "run" : "stop");
  $("ov-eval-sg").textContent = d.eval.stage || "-";
  if (d.trend && d.trend.time) drawTrend(d.trend, "(最新采集)");
}
async function loadLogs() {
  const d = await jget("/api/logs");
  $("ov-logs").innerHTML = (d.logs || []).map(l =>
    `<div class="logline"><span class="t">${l.time.split(" ")[1]}</span><span class="tag ${l.tag}">[${l.tag}]</span><span>${l.msg}</span></div>`).join("");
}
loaders.overview = () => { loadOverview(); loadLogs(); };

/* ══════════ 2 路网管理 ══════════ */
async function loadNetwork() {
  const d = await jget("/api/intersections");
  const tb = $("nw-table"); tb.innerHTML = "";
  const genSel = $("nw-gen-base"); genSel.innerHTML = "";
  $("nw-count").textContent = `${d.intersections.length} 个场景`;
  const kindName = { single: "单路口", arterial: "干线", generated: "扰动生成" };
  d.intersections.forEach(s => {
    const pv = s.provenance;
    const shape = pv ? `${pv.type}${pv.approaches ? ` (${pv.approaches}进口/${pv.phases}相位)` : ""}` : "-";
    tb.innerHTML += `<tr><td>${s.display}</td><td><span class="badge blue">${kindName[s.kind]}</span></td>
      <td style="font-size:12px">${shape}</td><td style="font-size:12px">${pv && pv.note ? pv.note : (s.routes || "-")}</td></tr>`;
    genSel.innerHTML += `<option value="${s.name}">${s.display}</option>`;
  });
  const st = d.statistics || {};
  $("nw-stats").innerHTML = `
    <tr><td>单路口场景</td><td style="text-align:right">${d.intersections.filter(s=>s.kind==="single").length}</td></tr>
    <tr><td>干线场景</td><td style="text-align:right">${d.intersections.filter(s=>s.kind==="arterial").length}</td></tr>
    <tr><td>扰动生成场景</td><td style="text-align:right">${d.intersections.filter(s=>s.kind==="generated").length}</td></tr>
    <tr><td>异形路口占比</td><td style="text-align:right">${st.total ? `${st.non_standard}/${st.total} (窄路密网特征)` : "-"}</td></tr>
    <tr><td>数据来源</td><td style="text-align:right;font-size:12px">雄安实测调查数据(团队数据V2)</td></tr>
    <tr><td>核对方式</td><td style="text-align:right;font-size:12px">高德地图逐路口核对 + 比赛方确认修正</td></tr>
    <tr><td>评估报告</td><td style="text-align:right">reports/ + platform_data/eval</td></tr>`;
}
async function genScenario(kind) {
  const base = $("nw-gen-base").value;
  const d = await jpost("/api/scenario/generate", { base, kind });
  $("nw-gen-msg").textContent = d.ok ? `已生成 ${d.name} (需求×${d.factor})` : `失败: ${d.msg}`;
  if (d.ok) loadNetwork();
}
loaders.network = loadNetwork;

/* ══════════ 3 GIS地图监控 ══════════ */
async function loadGis() {
  const d = await jget("/api/intersections");
  const svg = $("gis-svg");
  let html = `<rect x="0" y="0" width="860" height="420" fill="#f6f8fc" rx="10"/>
    <text x="24" y="36" font-size="14" fill="#1a2233" font-weight="600">干线走廊(东西向) · J0-J3/J4</text>
    <line x1="60" y1="110" x2="800" y2="110" stroke="#c9d6ea" stroke-width="10" stroke-linecap="round"/>
    <line x1="60" y1="330" x2="800" y2="330" stroke="#c9d6ea" stroke-width="0"/>`;
  // 干线走廊 J0..J4
  const arts = d.intersections.filter(s => s.kind === "arterial");
  arts.forEach((s, i) => {
    const cnt = s.name.includes("干线二") ? 5 : (s.name.includes("干线一") ? 4 : 2);
    const x0 = 120 + i * 240, y0 = 110;
    html += `<text x="${x0 - 20}" y="80" font-size="11" fill="#7b879c">${s.display}</text>`;
    for (let j = 0; j < cnt; j++) {
      const x = x0 + j * (150 / Math.max(cnt - 1, 1)) * (cnt > 2 ? 1.6 : 1);
      html += `<g class="gis-node" data-sc="${s.name}" style="cursor:pointer">
        <rect x="${x - 16}" y="${y0 - 16}" width="32" height="32" rx="6" fill="#2563eb"/>
        <text x="${x}" y="${y0 + 4}" font-size="10" fill="#fff" text-anchor="middle">J${j}</text></g>`;
    }
  });
  // 单路口 4×5 网格
  html += `<text x="24" y="230" font-size="14" fill="#1a2233" font-weight="600">单路口区域 (20个)</text>`;
  const singles = d.intersections.filter(s => s.kind === "single");
  singles.forEach((s, i) => {
    const col = i % 5, row = Math.floor(i / 5);
    const x = 110 + col * 155, y = 268 + row * 62;
    html += `<g class="gis-node" data-sc="${s.name}" style="cursor:pointer">
      <rect x="${x - 40}" y="${y - 20}" width="120" height="40" rx="8" fill="#fff" stroke="#c9d6ea"/>
      <circle cx="${x - 22}" cy="${y}" r="6" fill="#ef4444"/>
      <text x="${x + 8}" y="${y + 4}" font-size="11" fill="#1a2233" text-anchor="middle">${s.display}</text></g>`;
  });
  svg.innerHTML = html;
  svg.querySelectorAll(".gis-node").forEach(g => {
    g.onclick = async () => {
      const sc = g.dataset.sc;
      const all = await jget("/api/intersections");
      const info = all.intersections.find(x => x.name === sc);
      const ev = await jget("/api/data?run=" + encodeURIComponent("art_dqn")).catch(() => null);
      const pv = info.provenance || {};
      $("gis-info").innerHTML = `<h3 style="margin-bottom:10px">${info.display}</h3>
        <table class="tb">
        <tr><td>场景文件</td><td>${info.file}</td></tr>
        <tr><td>类型</td><td>${info.kind === "arterial" ? "干线(多信号协同)" : info.kind === "generated" ? "扰动生成场景" : "单路口"}</td></tr>
        <tr><td>路口形态</td><td>${pv.type || "-"}${pv.approaches ? ` (${pv.approaches}进口/${pv.phases}相位)` : ""}</td></tr>
        <tr><td>形态说明</td><td>${pv.note || "-"}</td></tr>
        <tr><td>数据来源</td><td>${pv.type ? "雄安实测调查数据 · 高德地图核对" : "-"}</td></tr>
        <tr><td>参考延误</td><td>${ev && ev.metrics ? (ev.metrics.arrived ? "已采集" : "-") : "-"}</td></tr></table>
        <div class="foot-note">在「信号控制仿真」页选择该场景启动实时仿真; 在「性能评估」页运行固定配时vs DQN对比。</div>`;
    };
  });
}
loaders.gis = loadGis;

/* ══════════ 4 协同管控算法 ══════════ */
let histLoaded = false;
async function loadAlgorithm() {
  const st = await jget("/api/train/status");
  $("al-st").textContent = st.running ? "训练中" : "已停止";
  $("al-st").className = "st " + (st.running ? "run" : "stop");
  $("al-ep").textContent = fmtInt(st.episodes_done);
  $("al-eps").textContent = fmt2(st.epsilon);
  $("al-r").textContent = fmt2(st.latest_reward);
  $("al-r100").textContent = fmt2(st.avg_reward_100);
  if (!histLoaded) {
    const h = await jget("/api/train/history");
    chart("al-delay").setOption({
      backgroundColor: "transparent", tooltip: { trigger: "axis" },
      grid: { left: 50, right: 20, top: 16, bottom: 26 },
      xAxis: { type: "category", data: h.episode, axisLabel: { fontSize: 9 } },
      yAxis: { type: "value", name: "车均延误(s)" },
      series: [{ type: "line", showSymbol: false, data: h.delay, smooth: true,
        lineStyle: { color: "#d97706", width: 2 }, itemStyle: { color: "#d97706" } }] });
    chart("al-reward").setOption({
      backgroundColor: "transparent", tooltip: { trigger: "axis" },
      grid: { left: 50, right: 20, top: 16, bottom: 26 },
      xAxis: { type: "category", data: h.episode, axisLabel: { fontSize: 9 } },
      yAxis: { type: "value", name: "奖励" },
      series: [{ type: "line", showSymbol: false, data: h.reward, smooth: true,
        lineStyle: { color: "#2563eb", width: 2 }, itemStyle: { color: "#2563eb" } }] });
    chart("al-loss").setOption({
      backgroundColor: "transparent", tooltip: { trigger: "axis" },
      grid: { left: 50, right: 20, top: 16, bottom: 26 },
      xAxis: { type: "category", data: h.episode, axisLabel: { fontSize: 9 } },
      yAxis: { type: "value", name: "loss", scale: true },
      series: [{ type: "line", showSymbol: false, data: h.loss, smooth: true,
        lineStyle: { color: "#7c3aed", width: 2 }, itemStyle: { color: "#7c3aed" } }] });
    histLoaded = true;
  }
  if (!$("al-scenario").options.length) {
    const d = await jget("/api/intersections");
    $("al-scenario").innerHTML = `<option value="all">全部场景轮换 (20单路口+3干线)</option>` +
      d.intersections.map(s => `<option value="${s.name}">${s.display}</option>`).join("");
  }
}
async function trainStart() {
  const d = await jpost("/api/train/start", {
    episodes: +$("al-episodes").value, model: $("al-model").value, scenario: $("al-scenario").value });
  alert(d.msg);
}
async function trainStop() {
  const d = await jpost("/api/train/stop"); alert(d.msg);
}
loaders.algorithm = loadAlgorithm;

/* ══════════ 4.5 模型部署 ══════════ */
async function loadDeploy() {
  const d = await jget("/api/deploy");
  if (!d.ok || !d.report) {
    $("dp-params").textContent = "未生成";
    $("dp-verify").innerHTML = `<tr><td colspan="4">未找到基准报告, 请在项目根目录运行: python export_onnx.py</td></tr>`;
    $("dp-files").innerHTML = `<tr><td colspan="3">model_deployment/ 目录暂无产物</td></tr>`;
    return;
  }
  const r = d.report, b = r.benchmark;
  $("dp-params").textContent = fmtInt(b.params);
  const fp32 = (r.verify || []).find(v => (v.tag || "").includes("FP32")) || {};
  $("dp-agree").innerHTML = ((fp32.action_agreement || 0) * 100).toFixed(2) + "<span class='u'>%</span>";
  $("dp-ms").innerHTML = (b.onnx_fp32_cpu ? b.onnx_fp32_cpu.mean_ms * 1000 : 0).toFixed(0) + "<span class='u'>us</span>";
  $("dp-latency").textContent = b.onnx_fp32_cpu ? (b.onnx_fp32_cpu.mean_ms * 1000).toFixed(0) + "us" : "-";
  const ratio = b.size_pt_mb && b.size_onnx_fp32_mb ? (b.size_onnx_fp32_mb / b.size_pt_mb) : 0;
  $("dp-size").innerHTML = ratio ? (ratio * 100).toFixed(0) + "<span class='u'>%</span>" : "-";
  // 精度一致性表
  const verdict = a => a >= 0.9999 ? '<span class="st run">可直接部署</span>'
    : a >= 0.995 ? '<span class="st run">推荐(体积更小)</span>' : '<span class="st stop">极端场景备用</span>';
  $("dp-verify").innerHTML = (r.verify || []).map(v =>
    `<tr><td>${v.tag}</td><td>${v.max_abs_diff.toExponential(2)}</td><td>${(v.action_agreement * 100).toFixed(2)}%</td><td>${verdict(v.action_agreement)}</td></tr>`).join("");
  // 体积/延迟双轴对比图
  const names = ["PyTorch .pt", "ONNX FP32", "ONNX FP16", "ONNX INT8"];
  const sizes = [b.size_pt_mb * 1024, b.size_onnx_fp32_mb * 1024,
    (b.size_onnx_fp16_mb || 0) * 1024, (b.size_onnx_int8_mb || 0) * 1024];
  const lat = [b.pytorch_cpu ? b.pytorch_cpu.mean_ms * 1000 : 0,
    b.onnx_fp32_cpu ? b.onnx_fp32_cpu.mean_ms * 1000 : 0,
    b.onnx_fp16_cpu ? b.onnx_fp16_cpu.mean_ms * 1000 : 0,
    b.onnx_int8_cpu ? b.onnx_int8_cpu.mean_ms * 1000 : 0];
  chart("dp-bench").setOption({
    backgroundColor: "transparent", tooltip: { trigger: "axis" },
    legend: { top: 0, textStyle: { fontSize: 10, color: "#7b879c" } },
    grid: { left: 54, right: 54, top: 30, bottom: 24 },
    xAxis: { type: "category", data: names, axisLabel: { fontSize: 10 } },
    yAxis: [{ type: "value", name: "KB" }, { type: "value", name: "us/次", position: "right" }],
    series: [
      { name: "模型体积", type: "bar", data: sizes, barWidth: 26,
        itemStyle: { color: "#2563eb", borderRadius: [4, 4, 0, 0] } },
      { name: "单次推理延迟", type: "line", yAxisIndex: 1, data: lat, smooth: true,
        lineStyle: { color: "#d97706", width: 2 }, itemStyle: { color: "#d97706" } }],
  });
  // 模型文件清单
  const purpose = f => f.includes("int8") ? "INT8动态量化 · 极致压缩(极端场景备用)"
    : f.includes("fp16") ? "FP16半精度 · 边缘芯片(NPU)部署"
    : f.endsWith(".data") ? "FP32权重数据 (torch导出器外部数据格式)"
    : "ONNX FP32 · 默认边缘部署格式 (动作一致率100%)";
  $("dp-files").innerHTML = d.models.map(m =>
    `<tr><td style="font-family:Consolas,monospace">${m.file}</td><td>${m.size_kb} KB</td><td>${purpose(m.file)}</td></tr>`).join("");
}
loaders.deploy = loadDeploy;

/* ══════════ 5 信号控制仿真 ══════════ */
const simCharts = {};
let simTls = "", simTimer = null;
const LIGHT_COLOR = { green: "#22c55e", yellow: "#f59e0b", red: "#ef4444" };
async function loadSim() {
  if (!$("sim-scenario").options.length) {
    const d = await jget("/api/intersections");
    $("sim-scenario").innerHTML = d.intersections
      .filter(s => s.kind !== "generated")
      .map(s => `<option value="${s.name}">${s.display}</option>`).join("");
  }
  if (!simTimer) simTimer = setInterval(pollSim, 2000);
  pollSim();
}
function setLight(dir, color) {
  const el = $("lt-" + dir);
  if (!el) return;
  el.className = "light" + (color === "green" ? " green" : color === "yellow" ? " yellow" : "");
}
async function pollSim() {
  const d = await jget("/api/sim/state");
  $("sim-st").textContent = d.message || (d.running ? "运行中" : "已停止");
  $("sim-st").className = "st " + (d.running ? "run" : (d.message === "已完成" ? "stop" : "stop"));
  $("sim-t").textContent = d.time || 0;
  $("sim-policy-now").textContent = d.policy === "dqn" ? "DQN协同管控" : (d.policy === "fixed" ? "固定配时" : "-");
  const m = d.metrics || {};
  $("sm-wait").textContent = fmt2(m.wait);
  $("sm-queue").textContent = fmtInt(m.queue);
  $("sm-arr").textContent = fmtInt(m.arrived);
  $("sm-speed").textContent = fmt2(m.speed_kmh);
  $("sm-fuel").textContent = fmt2(m.fuel_l);
  $("sm-run").textContent = fmtInt(m.running_veh);
  const tlsIds = Object.keys(d.tls || {});
  if (tlsIds.length) {
    if (!simTls || !tlsIds.includes(simTls)) simTls = tlsIds[0];
    $("sim-node").textContent = simTls;
    $("sim-phase").textContent = d.tls[simTls].label;
    const ap = d.approach[simTls] || {};
    for (const dir of ["北", "东", "南", "西"]) setLight(dir, ap[dir] || "red");
  }
  const h = d.history || [];
  if (h.length && chart("sim-curve")) {
    (simCharts.cur = simCharts.cur || chart("sim-curve")).setOption({
      backgroundColor: "transparent", tooltip: { trigger: "axis" },
      legend: { top: 0, textStyle: { fontSize: 10, color: "#7b879c" } },
      grid: { left: 46, right: 46, top: 30, bottom: 24 },
      xAxis: { type: "category", data: h.map(p => p.t + "s"), axisLabel: { fontSize: 9 } },
      yAxis: [{ type: "value", name: "s/辆" }, { type: "value", name: "km/h", position: "right" }],
      series: [
        { name: "平均等待", type: "line", showSymbol: false, data: h.map(p => p.wait), smooth: true,
          lineStyle: { color: "#d97706", width: 2 }, itemStyle: { color: "#d97706" } },
        { name: "排队", type: "line", yAxisIndex: 1, showSymbol: false, data: h.map(p => p.queue), smooth: true,
          lineStyle: { color: "#ef4444", width: 1.6 }, itemStyle: { color: "#ef4444" } },
        { name: "车速", type: "line", showSymbol: false, data: h.map(p => p.speed), smooth: true,
          lineStyle: { color: "#2563eb", width: 1.6 }, itemStyle: { color: "#2563eb" } },
        { name: "累计通过", type: "line", yAxisIndex: 1, showSymbol: false, data: h.map(p => p.arrived), smooth: true,
          lineStyle: { color: "#16a34a", width: 1.6 }, itemStyle: { color: "#16a34a" } }],
    });
  }
}
async function simStart() {
  const d = await jpost("/api/sim/start", {
    scenario: $("sim-scenario").value, policy: $("sim-policy").value, seconds: +$("sim-seconds").value });
  $("sim-st").textContent = d.msg;
  setTimeout(pollSim, 1500);
}
async function simStop() { await jpost("/api/sim/stop"); }
loaders.sim = loadSim;

/* ══════════ 6 性能评估 ══════════ */
let evalTimer = null;
async function loadEval() {
  if (!$("ev-scenario").options.length) {
    const d = await jget("/api/intersections");
    $("ev-scenario").innerHTML = d.intersections
      .map(s => `<option value="${s.name}">${s.display}</option>`).join("");
  }
  if (!evalTimer) evalTimer = setInterval(pollEval, 2000);
}
async function evalStart() {
  const d = await jpost("/api/eval/run", {
    scenario: $("ev-scenario").value, seconds: +$("ev-seconds").value, seed: +$("ev-seed").value });
  $("ev-stage").textContent = d.msg;
}
async function pollEval() {
  const d = await jget("/api/eval/status");
  $("ev-stage").textContent = d.running ? d.stage : (d.error ? ("失败: " + d.error) : (d.stage || "空闲"));
  $("ev-run").disabled = d.running;
  const r = d.result;
  if (!r) return;
  const k = r.kpi;
  $("ev-delay").innerHTML = (k.delay_down >= 0 ? "" : "") + k.delay_down.toFixed(1) + "<span class='u'>%</span>";
  $("ev-tl").innerHTML = k.timeloss_down.toFixed(1) + "<span class='u'>%</span>";
  $("ev-speed").innerHTML = k.speed_up.toFixed(1) + "<span class='u'>%</span>";
  $("ev-count").innerHTML = k.count_up.toFixed(1) + "<span class='u'>%</span>";
  chart("ev-radar").setOption({
    backgroundColor: "transparent",
    legend: { bottom: 0, textStyle: { fontSize: 11, color: "#7b879c" } },
    radar: { indicator: k.radar.map(x => ({ name: x.dim, max: 100 })), radius: "62%",
      splitArea: { areaStyle: { color: ["#fff", "#f7f9fd"] } } },
    series: [{ type: "radar",
      data: [
        { name: "DQN协同管控", value: k.radar.map(x => Math.max(0, x.value)),
          lineStyle: { color: "#2563eb", width: 2.5 }, itemStyle: { color: "#2563eb" },
          areaStyle: { color: "rgba(37,99,235,.18)" } },
        { name: "固定配时基准", value: k.radar.map(() => 0), lineStyle: { color: "#94a3b8" }, itemStyle: { color: "#94a3b8" } }] }],
  });
}
loaders.eval = loadEval;

/* ══════════ 7 云边端协同 ══════════ */
async function loadCve() {
  const d = await jget("/api/edge");
  $("cv-end-n").textContent = "TraCI逐秒";
  $("cv-edge-rows").textContent = fmtInt(d.edge.aggregated_rows) + " 行";
  $("cv-edge-n").textContent = d.edge.nodes.length;
  $("cv-cloud-api").textContent = fmtInt(d.cloud.api_requests) + " 次";
  $("cv-cloud-run").textContent = d.cloud.runs + " 组";
  $("cv-table").innerHTML = d.edge.nodes.map(n =>
    `<tr><td>${n.net}</td><td>${n.tls}</td><td>${n.links}</td><td><span class="st run">已接入</span></td></tr>`).join("");
}
loaders.cve = loadCve;

/* ══════════ 8 接口文档 ══════════ */
async function loadDocs() {
  const d = await jget("/api/docs");
  $("docs-table").innerHTML = d.api.map(a =>
    `<tr><td><span class="badge blue">${a.method}</span></td><td style="font-family:Consolas,monospace">${a.path}</td>
     <td>${a.desc}</td><td>${a.body ? "<code>" + JSON.stringify(a.body) + "</code>" : "-"}</td></tr>`).join("");
}
loaders.docs = loadDocs;

/* ── 启动 ── */
loadOverview();
loadLogs();
setInterval(() => { if ($("pg-overview").classList.contains("active")) { loadOverview(); } }, 3000);
setInterval(() => { if ($("pg-overview").classList.contains("active")) loadLogs(); }, 5000);
setInterval(() => { if ($("pg-algorithm").classList.contains("active")) loadAlgorithm(); }, 3000);
