import { $, S, apiGet, apiPost } from "./core.js";

let busy = false;
let artifactsKey = "";
let commandError = "";

function renderArtifacts(items) {
  const key = JSON.stringify(items);
  if (key === artifactsKey) return;
  artifactsKey = key;
  const host = $("rsi-artifacts");
  host.replaceChildren();
  if (!items.length) host.textContent = "暂无产物";
  for (const item of items) {
    const link = document.createElement("a");
    link.textContent = item.name;
    link.href = "/api/rsi" + item.url;
    link.target = "_blank";
    link.rel = "noopener";
    host.appendChild(link);
    if (item.kind === "image") {
      const image = document.createElement("img");
      image.src = link.href;
      image.alt = item.name;
      host.appendChild(image);
    }
  }
}

function renderFrame(frame) {
  $("rsi-qpos").textContent = "QPOS " + frame.qpos.map((value) => Number(value).toFixed(3)).join(" · ");
  const host = $("rsi-cameras");
  const names = Object.keys(frame.cameras);
  if (host.dataset.names !== JSON.stringify(names)) {
    host.replaceChildren();
    for (const name of names) {
      const cell = document.createElement("div");
      const label = document.createElement("div");
      label.textContent = name;
      const image = document.createElement("img");
      image.alt = name;
      cell.append(label, image);
      host.appendChild(cell);
    }
    if (!names.length) host.textContent = "暂无相机观测";
    host.dataset.names = JSON.stringify(names);
  }
  host.querySelectorAll("img").forEach((image, index) => {
    image.src = "data:image/jpeg;base64," + frame.cameras[names[index]];
  });
  const available = frame.simulation.available && !!frame.simulation.image;
  $("rsi-render").hidden = !available;
  $("rsi-render-empty").hidden = !!available;
  if (available) $("rsi-render").src = "data:image/jpeg;base64," + frame.simulation.image;
  else $("rsi-render").removeAttribute("src");
  $("rsi-render-source").textContent = frame.simulation.source || "";
}

async function pollRsi() {
  if (S.ACTIVE_TAB !== "rsi" || document.hidden || busy) return;
  try {
    const status = await apiGet("/api/rsi/status", { timeoutMs: 6500 });
    if (status.ok === false) throw new Error(status.error);
    $("rsi-service").textContent = status.status;
    $("rsi-stage").textContent = status.stage;
    $("rsi-simulation-status").textContent = status.simulation.status;
    $("rsi-robot").textContent = status.robot.connected ? "已连接" : "未连接";
    $("rsi-source").textContent = status.robot.source;
    $("rsi-cost").textContent = status.api.cost_usd == null ? "未知" : "$" + status.api.cost_usd;
    $("rsi-cost-scope").textContent = (status.api.cost_scope || "计费口径未知") + "；实际账单未知";
    $("rsi-qps").textContent = status.api.qps == null ? "待测量" : status.api.qps;
    $("rsi-model").textContent = status.api.model || "—";
    $("rsi-error").textContent = status.error || commandError;
    $("rsi-missing").textContent = status.missing.length ? "缺少配置：\n" + status.missing.join("\n") : "";
    $("rsi-run").disabled = status.status === "running" || !!status.missing.length || status.session.stopped;
    $("rsi-stop").disabled = status.status !== "running";
    $("rsi-seek").max = Math.max(0, status.episode.length - 1);
    $("rsi-seek").disabled = !status.episode.length || status.status === "running";
    $("rsi-seek").value = status.episode.index;
    $("rsi-frame-index").textContent = `${status.episode.index} / ${status.episode.length}`;
    const [frame, artifacts] = await Promise.all([
      apiGet("/api/rsi/frame", { timeoutMs: 6500 }),
      apiGet("/api/rsi/artifacts", { timeoutMs: 6500 }),
    ]);
    if (frame.ok === false) throw new Error(frame.error);
    if (artifacts.ok === false) throw new Error(artifacts.error);
    renderFrame(frame);
    renderArtifacts(artifacts.items);
  } catch (error) {
    $("rsi-service").textContent = "不可用";
    $("rsi-error").textContent = error.message;
    $("rsi-robot").textContent = "连接状态未知";
    $("rsi-run").disabled = true;
    $("rsi-stop").disabled = true;
    $("rsi-seek").disabled = true;
  }
}

async function command(action, body) {
  busy = true;
  commandError = "";
  $("rsi-run").disabled = true;
  $("rsi-stop").disabled = true;
  $("rsi-seek").disabled = true;
  try {
    const result = await apiPost("/api/rsi/" + action, body);
    if (result.ok === false) throw new Error(result.error);
  } catch (error) {
    commandError = error.message;
    $("rsi-error").textContent = commandError;
  } finally {
    busy = false;
  }
}

function initRsi() {
  $("rsi-run").onclick = () => command("run");
  $("rsi-stop").onclick = () => command("stop");
  $("rsi-seek").onchange = () => command("seek", { index: Number($("rsi-seek").value) });
}

export { initRsi, pollRsi };
