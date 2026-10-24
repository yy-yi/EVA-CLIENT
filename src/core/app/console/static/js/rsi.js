import { $, S, apiGet, apiPost, setCommandMetadata } from "./core.js";

let busy = false;
let polling = false;
let artifactsKey = "";
let runsKey = "";
let commandError = "";
let tasks = [];
let providers = [];
let selectedTaskId = "";
let selectedRunId = "";
let latestStatus = null;
let latestClient = null;
let latestCandidate = null;
let autoArmed = false;
let autoBaseline = "";
let autoReplayId = "";
let autoSendId = "";

function badge(id, text, state) {
  $(id).textContent = text;
  $(id).dataset.state = state;
}

function artifactLink(item) {
  const link = document.createElement("a");
  link.textContent = item.name;
  link.href = "/api/rsi" + item.url;
  link.target = "_blank";
  link.rel = "noopener";
  return link;
}

function renderArtifacts(items) {
  const key = JSON.stringify(items);
  if (key === artifactsKey) return;
  artifactsKey = key;
  const host = $("rsi-artifacts");
  host.replaceChildren();
  $("rsi-artifact-count").textContent = String(items.length);
  if (!items.length) host.textContent = "暂无产物";
  for (const item of items) {
    const cell = document.createElement("div");
    cell.className = "rsi-artifact";
    const link = artifactLink(item);
    cell.appendChild(link);
    if (item.kind === "image") {
      const image = document.createElement("img");
      image.src = link.href;
      image.alt = item.name;
      image.loading = "lazy";
      cell.appendChild(image);
    }
    host.appendChild(cell);
  }
}

function renderFrame(frame) {
  $("rsi-qpos").textContent = "QPOS " + frame.qpos.map((value) => Number(value).toFixed(3)).join(" · ");
  const host = $("rsi-cameras");
  const names = Object.keys(frame.cameras);
  if (host.dataset.names !== JSON.stringify(names)) {
    host.replaceChildren();
    for (const name of names) {
      const cell = document.createElement("figure");
      const label = document.createElement("figcaption");
      label.textContent = name;
      const image = document.createElement("img");
      image.alt = name;
      cell.append(image, label);
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

function renderTasks(payload) {
  tasks = payload.tasks;
  selectedTaskId = payload.selected_task_id;
  const select = $("rsi-task-list");
  const placeholder = new Option("SELECT TASK", "");
  placeholder.disabled = true;
  select.replaceChildren(placeholder);
  for (const task of tasks) {
    select.appendChild(new Option(`${task.task_id} · ${task.prompt_zh || task.prompt_en}`, task.task_id));
  }
  select.value = selectedTaskId;
  select.classList.toggle("has-value", !!select.value);
  $("rsi-task-count").textContent = String(tasks.length);
  const selected = tasks.find((task) => task.task_id === selectedTaskId);
  $("rsi-task-preview").textContent = selected ? selected.prompt_en : "选择任务后开始推理。";
  $("rsi-task-name").value = selected ? selected.task_id : "";
  $("rsi-task-prompt").value = selected ? selected.prompt_en : "";
}

function renderModels(payload) {
  providers = payload.providers;
  const providerSelect = $("rsi-provider-list");
  providerSelect.replaceChildren();
  for (const provider of providers) providerSelect.appendChild(new Option(provider.label, provider.id));
  providerSelect.value = payload.selected.provider_id;
  renderModelChoices(payload.selected.provider_id, payload.selected.model_id);
}

function renderModelChoices(providerId, modelId) {
  const provider = providers.find((item) => item.id === providerId);
  const select = $("rsi-model-list");
  select.replaceChildren();
  for (const model of provider.models) {
    select.appendChild(new Option(`${model.label} · ${model.reasoning_effort}`, model.id));
  }
  select.value = modelId || provider.models[0].id;
}

function renderClient(client) {
  latestClient = client;
  badge("rsi-robot", client.robot_connected ? "CONNECTED" : "DISCONNECTED", client.robot_connected ? "ok" : "off");
  $("rsi-transport").textContent = `${client.robot_type} · ${client.transport_type}`;
  document.querySelectorAll("[data-rsi-dispatch]").forEach((button) => {
    const selected = button.dataset.rsiDispatch === client.mode;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  $("rsi-execution-status").textContent = client.mode === "auto" && autoArmed ? "AUTO · REPLAY → SEND" : "MANUAL REVIEW";
  $("rsi-execution-note").textContent = client.mode === "auto" && autoArmed
    ? "新候选通过仿真回放后自动发送；STOP 随时停止。"
    : "推理不会自动驱动真机。先仿真回放，再点击 SEND TO REAL。";
  updateControls();
}

function candidateMatchesTask(candidate) {
  const selected = tasks.find((task) => task.task_id === selectedTaskId);
  return candidate?.task_id === selectedTaskId && candidate?.task_prompt === selected?.prompt_en;
}

function updateControls() {
  const running = latestStatus?.operation_active || latestStatus?.status === "running";
  const replaying = latestStatus?.replay?.status === "running";
  const candidate = latestCandidate;
  const connected = latestClient?.robot_connected && latestClient?.hardware_enabled;
  const dispatching = latestClient?.pending_steps > 0;
  const sent = latestClient?.sent_candidate_ids.includes(candidate?.candidate_id);
  const replayed = candidate?.replay_completed;
  const selectedCandidate = candidateMatchesTask(candidate);
  $("rsi-run").disabled = busy || dispatching || !latestStatus || running || replaying || !selectedTaskId || !!latestStatus.missing.length;
  $("rsi-stop").disabled = busy || !(running || replaying || connected || autoArmed || latestClient?.pending_steps);
  $("rsi-home").disabled = busy || dispatching || !connected || running || replaying;
  $("rsi-replay").disabled = busy || dispatching || !latestStatus || running || replaying || !selectedCandidate || !candidate?.review_passed;
  $("rsi-send").disabled = busy || dispatching || !latestStatus || running || replaying || !connected || !selectedCandidate || !replayed || sent;
  $("rsi-seek").disabled = busy || !latestStatus?.episode.length || running || replaying;
  for (const id of ["rsi-task-list", "rsi-task-save", "rsi-provider-list", "rsi-model-list"]) {
    $(id).disabled = busy || dispatching || running || replaying;
  }
}

function renderStatus(status) {
  latestStatus = status;
  badge("rsi-service", status.status.toUpperCase(), status.status === "running" ? "run" : status.status === "failed" ? "warn" : "ok");
  $("rsi-stage").textContent = status.stage;
  badge("rsi-simulation-status", status.simulation.status, status.simulation.available ? "ok" : "off");
  $("rsi-source").textContent = `${status.robot.source} · episode ${status.episode.id}`;
  $("rsi-cost").textContent = status.api.cost_usd == null ? "未知" : "$" + status.api.cost_usd;
  $("rsi-cost-scope").textContent = (status.api.cost_scope || "计费口径未知") + "；实际账单未知";
  $("rsi-qps").textContent = status.api.qps == null ? "待测量" : status.api.qps;
  $("rsi-requests").textContent = `${status.api.completed_calls} / ${status.api.calls}`;
  $("rsi-model").textContent = status.api.model || "—";
  $("rsi-error").textContent = status.error || latestClient?.last_error || commandError;
  $("rsi-missing").textContent = status.missing.length ? "缺少配置：\n" + status.missing.join("\n") : "";
  $("rsi-seek").max = Math.max(0, status.episode.length - 1);
  $("rsi-seek").value = status.episode.index;
  $("rsi-frame-index").textContent = `${status.episode.index} / ${status.episode.length} · ${status.episode.fps} Hz`;
  updateControls();
}

function renderRuns(items) {
  const key = JSON.stringify(items);
  if (key === runsKey) return;
  runsKey = key;
  const host = $("rsi-runs");
  host.replaceChildren();
  $("rsi-run-count").textContent = String(items.length);
  if (!items.length) host.textContent = "暂无推理记录";
  for (const item of items) {
    const button = document.createElement("button");
    button.className = "btn rsi-run-item";
    button.dataset.runId = item.id;
    button.classList.toggle("selected", item.id === selectedRunId);
    const label = document.createElement("span");
    label.textContent = `${item.task.prompt || item.task.prompt_en} · ${item.started_at}`;
    const state = document.createElement("small");
    state.textContent = item.status;
    button.append(label, state);
    button.onclick = () => renderRunDetail(item).catch((error) => {
      $("rsi-run-detail").textContent = error.message;
    });
    host.appendChild(button);
    if (item.id === selectedRunId) renderRunDetail(item).catch((error) => {
      $("rsi-run-detail").textContent = error.message;
    });
  }
}

async function renderRunDetail(item) {
  selectedRunId = item.id;
  document.querySelectorAll(".rsi-run-item").forEach((button) => {
    button.classList.toggle("selected", button.dataset.runId === item.id);
  });
  const host = $("rsi-run-detail");
  host.hidden = false;
  host.replaceChildren();
  const link = artifactLink({ name: "DOWNLOAD · 完整推理记录", url: item.download_url });
  const summary = document.createElement("p");
  summary.className = "rsi-note";
  const frames = item.recording.datasets.reduce((total, dataset) => total + dataset.frames, 0);
  summary.textContent = `${item.recording.status} · ${frames} 帧轨迹 · ${item.finished_at || "运行中"}`;
  host.append(link, summary);
  if (item.error) {
    const error = document.createElement("p");
    error.className = "errline";
    error.textContent = item.error;
    host.appendChild(error);
  }
  const manifest = await apiGet(`/api/rsi/runs/${encodeURIComponent(item.id)}/manifest`);
  if (selectedRunId !== item.id) return;
  const details = document.createElement("details");
  const heading = document.createElement("summary");
  heading.textContent = `轨迹、状态、视角与模型过程 · ${manifest.files.length} 个文件`;
  const files = document.createElement("pre");
  files.textContent = manifest.files.join("\n");
  details.append(heading, files);
  host.appendChild(details);
}

async function command(action, body = {}) {
  busy = true;
  commandError = "";
  updateControls();
  try {
    const result = await apiPost("/api/rsi/" + action, body);
    if (result.ok === false) throw new Error(result.error);
    return result;
  } catch (error) {
    commandError = error.message;
    $("rsi-error").textContent = commandError;
    throw error;
  } finally {
    busy = false;
    updateControls();
  }
}

async function automaticDispatch() {
  const candidate = latestCandidate;
  if (!autoArmed || latestClient.mode !== "auto" || !candidate || candidate.candidate_id === autoBaseline) return;
  if (!candidateMatchesTask(candidate) || latestStatus.operation_active || latestStatus.status === "running" || latestStatus.replay?.status === "running") return;
  const id = candidate.candidate_id;
  if (!candidate.replay_completed && candidate.review_passed && autoReplayId !== id) {
    autoReplayId = id;
    await command("replay", { candidate_id: id });
  } else if (candidate.replay_completed && latestClient.hardware_enabled && !latestClient.sent_candidate_ids.includes(id) && autoSendId !== id) {
    autoSendId = id;
    await command("send", { candidate_id: id });
  }
}

async function pollRsi() {
  if (S.ACTIVE_TAB !== "rsi" || document.hidden || busy || polling) return;
  polling = true;
  let clientRead = false;
  try {
    const client = await apiGet("/api/rsi/client-status", { timeoutMs: 6500 });
    if (client.ok === false) throw new Error(client.error);
    renderClient(client);
    clientRead = true;
    if (!providers.length) renderModels(await apiGet("/api/rsi/models"));
    if (!tasks.length) renderTasks(await apiGet("/api/rsi/tasks"));
    const [status, candidate, runs] = await Promise.all([
      apiGet("/api/rsi/status", { timeoutMs: 6500 }),
      apiGet("/api/rsi/candidate", { timeoutMs: 6500 }),
      apiGet("/api/rsi/runs", { timeoutMs: 6500 }),
    ]);
    if (status.ok === false) throw new Error(status.error);
    latestCandidate = candidate.candidate;
    renderStatus(status);
    renderRuns(runs.items);
    const [frame, artifacts] = await Promise.all([
      apiGet("/api/rsi/frame", { timeoutMs: 6500 }),
      apiGet("/api/rsi/artifacts", { timeoutMs: 6500 }),
    ]);
    if (frame.ok === false) throw new Error(frame.error);
    if (artifacts.ok === false) throw new Error(artifacts.error);
    renderFrame(frame);
    renderArtifacts(artifacts.items);
    await automaticDispatch();
  } catch (error) {
    badge("rsi-service", "UNAVAILABLE", "warn");
    $("rsi-error").textContent = error.message;
    latestStatus = null;
    if (!clientRead) {
      latestClient = null;
      badge("rsi-robot", "UNKNOWN", "off");
      $("rsi-transport").textContent = "连接状态暂不可读";
    }
    updateControls();
  } finally {
    polling = false;
  }
}

function cancelAutomaticDispatch() {
  autoArmed = false;
  autoBaseline = latestCandidate?.candidate_id || "";
  autoReplayId = "";
  autoSendId = "";
}

async function selectTask(taskId) {
  cancelAutomaticDispatch();
  renderTasks(await command("task", { task_id: taskId }));
  renderClient(await apiGet("/api/rsi/client-status"));
}

async function saveTask() {
  const id = $("rsi-task-name").value.trim();
  const prompt = $("rsi-task-prompt").value.trim();
  if (!id || !prompt) throw new Error("请填写任务 ID 和任务描述。");
  await command("tasks", { task_id: id, prompt_en: prompt });
  await selectTask(id);
  $("rsi-task-editor").open = false;
}

async function selectDispatch(mode) {
  cancelAutomaticDispatch();
  await command("mode", { mode });
  autoArmed = mode === "auto";
  renderClient(await apiGet("/api/rsi/client-status"));
}

async function selectModel(providerChanged) {
  cancelAutomaticDispatch();
  if (providerChanged) renderModelChoices($("rsi-provider-list").value);
  await command("mode", { mode: "manual" });
  renderClient(await apiGet("/api/rsi/client-status"));
  renderModels(await command("model", {
    provider_id: $("rsi-provider-list").value,
    model_id: $("rsi-model-list").value,
  }));
}

function initRsi() {
  const handle = (operation, errorId = "rsi-error") => () => operation().catch((error) => {
    $(errorId).textContent = error.message;
  });
  $("rsi-run").onclick = handle(() => command("run", {
    task: { ...tasks.find((task) => task.task_id === selectedTaskId), id: selectedTaskId, prompt: tasks.find((task) => task.task_id === selectedTaskId).prompt_en },
  }));
  $("rsi-stop").onclick = handle(async () => {
    cancelAutomaticDispatch();
    const results = await Promise.allSettled([command("stop"), apiPost("/api/rsi/halt")]);
    const failed = results.find((result) => result.status === "rejected");
    if (failed) throw failed.reason;
    const rejected = results.find((result) => result.status === "fulfilled" && result.value.ok === false);
    if (rejected) throw new Error(rejected.value.error);
  });
  $("rsi-home").onclick = handle(() => command("home"));
  $("rsi-replay").onclick = handle(() => command("replay", { candidate_id: latestCandidate.candidate_id }));
  $("rsi-send").onclick = handle(() => command("send", { candidate_id: latestCandidate.candidate_id }));
  $("rsi-seek").onchange = handle(() => command("seek", { index: Number($("rsi-seek").value) }));
  $("rsi-task-list").onchange = handle(() => selectTask($("rsi-task-list").value), "rsi-task-error");
  $("rsi-task-save").onclick = handle(saveTask, "rsi-task-error");
  $("rsi-provider-list").onchange = handle(() => selectModel(true));
  $("rsi-model-list").onchange = handle(() => selectModel(false));
  document.querySelectorAll("[data-rsi-dispatch]").forEach((button) => {
    button.onclick = handle(() => selectDispatch(button.dataset.rsiDispatch));
  });
  $("rsi-device").onclick = () => document.querySelector('.tab[data-tab="manual"]').click();
  $("rsi-dashboard").onclick = () => document.querySelector('.tab[data-tab="dashboard"]').click();
  setCommandMetadata($("rsi-home"), "web:halt|web:rsi_home:{generation}", true);
  setCommandMetadata($("rsi-stop"), "web:halt");
}

export { initRsi, pollRsi };
