"use strict";

const $ = (sel) => document.querySelector(sel);
const TARGETS = {
  "1080p": { "16:9": [1920, 1080], "9:16": [1080, 1920], "1:1": [1080, 1080] },
  "720p": { "16:9": [1280, 720], "9:16": [720, 1280], "1:1": [720, 720] },
};
const ACTIVE = new Set(["queued", "running"]);

const state = {
  info: null,
  engines: {},
  engine: null,
  mode: "t2v",
  image: null, // Blob
  imageSize: null,
  jobs: [],
  selected: null,
  cards: new Map(),
  pollTimer: null,
};

// ---------- utilities ----------

const store = {
  get(key, fallback) {
    try {
      const raw = localStorage.getItem("videogen:" + key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem("videogen:" + key, JSON.stringify(value));
    } catch {
      /* private mode or storage disabled: preferences just won't persist */
    }
  },
};

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : body.detail?.[0]?.msg || JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.status === 204 ? null : res.json();
}

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

function fmtSeconds(s) {
  if (s == null) return "";
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${Math.round(s % 60)} s`;
}

function segValue(group) {
  return $(`#${group} [aria-checked="true"]`)?.dataset.value;
}

function setSeg(group, value) {
  const buttons = document.querySelectorAll(`#${group} [role="radio"]`);
  let found = false;
  buttons.forEach((b) => {
    const on = b.dataset.value === value && !b.hidden;
    b.setAttribute("aria-checked", String(on));
    found ||= on;
  });
  if (!found && buttons.length) buttons[0].setAttribute("aria-checked", "true");
}

function showError(message) {
  const box = $("#form-error");
  box.textContent = message || "";
  box.hidden = !message;
}

// ---------- hardware + engines ----------

function renderHardware(hw) {
  const chip = $("#hw");
  if (hw.device === "cuda") {
    chip.textContent = `${hw.gpu_name} · ${hw.vram_gb.toFixed(0)} GB VRAM · ${hw.ram_gb.toFixed(0)} GB RAM`;
    chip.className = "hw ok";
  } else if (hw.device === "mps") {
    chip.textContent = `Apple Silicon · ${hw.ram_gb.toFixed(0)} GB unified memory`;
    chip.className = "hw ok";
  } else {
    chip.textContent = "No GPU detected";
    chip.className = "hw warn";
  }
}

function renderEngines() {
  const box = $("#engines");
  box.textContent = "";
  const tpl = $("#engine-template");
  for (const engine of state.info.engines) {
    const node = tpl.content.firstElementChild.cloneNode(true);
    const input = node.querySelector("input");
    input.value = engine.id;
    input.id = `engine-${engine.id}`;
    input.addEventListener("change", () => selectEngine(engine.id));
    node.querySelector(".engine-title").textContent = engine.label;
    node.querySelector(".engine-tagline").textContent = engine.tagline;
    const badges = node.querySelector(".engine-badges");
    const add = (text, cls = "") => badges.append(el("span", { class: `badge ${cls}` }, text));
    add(engine.native_resolution);
    add(`${engine.fps} fps`);
    add(`up to ${engine.max_duration} s`);
    if (engine.supports_audio) add("sound", "good");
    if (engine.id !== "mock") {
      if (engine.downloaded) add("downloaded", "good");
      else add(`~${Math.round(engine.download_gb)} GB one-time download`, "note");
    }
    node.title = `${engine.description}\n\nLicense: ${engine.license}${engine.memory_plan ? `\nMemory: ${engine.memory_plan}` : ""}`;
    node.querySelector(".engine-warn").textContent = engine.warnings.join(" ");
    box.append(node);
  }
}

function selectEngine(id) {
  const engine = state.engines[id] || state.info.engines[0];
  state.engine = engine;
  const radio = $(`#engine-${engine.id}`);
  if (radio) radio.checked = true;
  store.set("engine", engine.id);

  const slider = $("#duration");
  slider.max = String(engine.max_duration);
  $("#duration-max-label").textContent = `${engine.max_duration} s`;
  if (Number(slider.value) > engine.max_duration) slider.value = String(engine.max_duration);
  updateDuration();

  $("#audio-toggle").hidden = !engine.supports_audio;
  $("#enhance-toggle").hidden = !engine.supports_prompt_enhancement;
  $("#tuning").hidden = !engine.tunable;
  $("#negative-field").hidden = !engine.tunable;
  $("#steps").placeholder = engine.default_steps ? `${engine.default_steps} (default)` : "default";
  $("#guidance").placeholder = engine.default_guidance != null ? `${engine.default_guidance} (default)` : "default";
  updateEstimate();
}

// ---------- form state ----------

function setMode(mode) {
  state.mode = mode;
  for (const tab of document.querySelectorAll(".mode [role=tab]")) {
    tab.setAttribute("aria-selected", String(tab.dataset.mode === mode));
  }
  $("#image-field").hidden = mode !== "i2v";
  $("#aspect-auto").hidden = mode !== "i2v";
  if (mode === "i2v" && state.image) setSeg("aspect", "auto");
  if (mode === "t2v" && segValue("aspect") === "auto") setSeg("aspect", "16:9");
  updateEstimate();
}

function setImage(blob) {
  state.image = blob;
  const preview = $("#dz-preview");
  if (preview.src) URL.revokeObjectURL(preview.src);
  if (!blob) {
    preview.removeAttribute("src");
    preview.hidden = true;
    $("#dz-empty").hidden = false;
    $("#dz-clear").hidden = true;
    state.imageSize = null;
    if (segValue("aspect") === "auto") setSeg("aspect", "16:9");
    updateEstimate();
    return;
  }
  preview.onload = () => {
    state.imageSize = [preview.naturalWidth, preview.naturalHeight];
    updateEstimate();
  };
  preview.src = URL.createObjectURL(blob);
  preview.hidden = false;
  $("#dz-empty").hidden = true;
  $("#dz-clear").hidden = false;
  setSeg("aspect", "auto");
  showError("");
}

function updateDuration() {
  $("#duration-out").textContent = `${$("#duration").value} s`;
  updateEstimate();
}

function resolvedAspect() {
  const aspect = segValue("aspect");
  if (aspect !== "auto") return aspect;
  if (!state.imageSize) return "16:9";
  const r = state.imageSize[0] / state.imageSize[1];
  const options = { "16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1 };
  return Object.keys(options).reduce((a, b) => (Math.abs(Math.log(r / options[a])) <= Math.abs(Math.log(r / options[b])) ? a : b));
}

function updateEstimate() {
  const engine = state.engine;
  if (!engine) return;
  const [w, h] = TARGETS[segValue("resolution")][resolvedAspect()];
  const seconds = Number($("#duration").value);
  const smooth = Number($("#smooth").value);
  const fps = smooth > engine.fps ? smooth : engine.fps;
  const parts = [`${w}×${h}`, `${fps} fps`, `${seconds} s`];
  if (engine.supports_audio && $("#audio").checked) parts.push("with sound");
  const segments = engine.id.startsWith("wan") && seconds > 5 ? Math.ceil((seconds - 5) / 5) + 1 : 1;
  if (segments > 1) parts.push(`${segments} chained segments`);
  $("#estimate").textContent = `Output: ${parts.join(" · ")}`;
}

function savePrefs() {
  store.set("prefs", {
    resolution: segValue("resolution"),
    aspect: segValue("aspect") === "auto" ? "16:9" : segValue("aspect"),
    duration: $("#duration").value,
    audio: $("#audio").checked,
    enhance: $("#enhance").checked,
    upscaler: $("#upscaler").value,
    smooth: $("#smooth").value,
  });
}

function loadPrefs() {
  const p = store.get("prefs", null);
  if (!p) return;
  if (p.resolution) setSeg("resolution", p.resolution);
  if (p.aspect) setSeg("aspect", p.aspect);
  if (p.duration) $("#duration").value = p.duration;
  if (typeof p.audio === "boolean") $("#audio").checked = p.audio;
  if (typeof p.enhance === "boolean") $("#enhance").checked = p.enhance;
  if (p.upscaler) $("#upscaler").value = p.upscaler;
  if (p.smooth) $("#smooth").value = p.smooth;
}

// ---------- submit ----------

async function submit(event) {
  event?.preventDefault();
  showError("");
  const prompt = $("#prompt").value.trim();
  if (!prompt) {
    showError("Please describe the video you want.");
    $("#prompt").focus();
    return;
  }
  if (state.mode === "i2v" && !state.image) {
    showError("Add a reference image, or switch to Text → Video.");
    return;
  }
  const form = new FormData();
  form.append("prompt", prompt);
  form.append("engine", state.engine.id);
  form.append("negative_prompt", $("#negative").value);
  form.append("duration", $("#duration").value);
  form.append("resolution", segValue("resolution"));
  form.append("aspect", segValue("aspect"));
  form.append("seed", $("#seed").value.trim() || "-1");
  form.append("steps", state.engine.tunable ? $("#steps").value.trim() : "");
  form.append("guidance", state.engine.tunable ? $("#guidance").value.trim() : "");
  form.append("audio", String($("#audio").checked));
  form.append("enhance_prompt", String($("#enhance").checked));
  form.append("upscaler", $("#upscaler").value);
  form.append("smooth_fps", $("#smooth").value);
  if (state.mode === "i2v") form.append("image", state.image, "input.png");

  const button = $("#generate");
  button.disabled = true;
  try {
    const job = await api("/api/jobs", { method: "POST", body: form });
    savePrefs();
    state.selected = job.id;
    await refreshJobs();
    if (window.matchMedia("(max-width: 960px)").matches) $("#viewer").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    showError(err.message);
  } finally {
    button.disabled = false;
  }
}

// ---------- jobs ----------

async function refreshJobs() {
  try {
    state.jobs = await api("/api/jobs");
  } catch (err) {
    console.warn("Could not refresh jobs", err);
  }
  renderHistory();
  renderViewer();
  schedulePoll();
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  const busy = state.jobs.some((j) => ACTIVE.has(j.status));
  state.pollTimer = setTimeout(refreshJobs, busy ? 1200 : 6000);
}

function statusLabel(job) {
  if (job.status === "running") return `${Math.round(job.progress * 100)}%`;
  return { queued: "Queued", failed: "Failed", cancelled: "Cancelled", done: "" }[job.status] || job.status;
}

function renderHistory() {
  const box = $("#history");
  const ids = new Set(state.jobs.map((j) => j.id));
  for (const [id, node] of state.cards) {
    if (!ids.has(id)) {
      node.remove();
      state.cards.delete(id);
    }
  }
  box.querySelector(".empty")?.remove();
  if (!state.jobs.length) box.append(el("p", { class: "empty" }, "Nothing here yet. Generate your first video!"));
  $("#history-count").textContent = state.jobs.length ? `${state.jobs.length} video${state.jobs.length > 1 ? "s" : ""}` : "";

  const tpl = $("#card-template");
  state.jobs.forEach((job, index) => {
    let card = state.cards.get(job.id);
    if (!card) {
      card = tpl.content.firstElementChild.cloneNode(true);
      card.addEventListener("click", () => {
        state.selected = job.id;
        renderHistory();
        renderViewer();
      });
      state.cards.set(job.id, card);
    }
    if (box.children[index] !== card) box.insertBefore(card, box.children[index] || null);
    const img = card.querySelector("img");
    const poster = job.poster_url || job.input_url;
    if (poster && img.getAttribute("src") !== poster) img.src = poster;
    const status = card.querySelector(".status");
    status.className = `status ${job.status}`;
    status.textContent = statusLabel(job);
    card.querySelector(".mini-bar").hidden = !ACTIVE.has(job.status);
    card.querySelector(".mini-bar > span").style.width = `${Math.round(job.progress * 100)}%`;
    card.querySelector(".card-text").textContent = job.request.prompt;
    card.classList.toggle("selected", job.id === state.selected);
    card.setAttribute("aria-label", `${job.request.prompt} (${job.status})`);
  });
}

function renderViewer() {
  const job = state.jobs.find((j) => j.id === state.selected) || state.jobs[0];
  const player = $("#player");
  const empty = $("#viewer-empty");
  const progress = $("#viewer-progress");
  const details = $("#details");
  if (!job) {
    empty.hidden = false;
    player.hidden = progress.hidden = details.hidden = true;
    return;
  }
  if (!state.selected) state.selected = job.id;
  empty.hidden = true;
  details.hidden = false;

  const [w, h] = TARGETS[job.request.resolution]?.[job.request.aspect] || [16, 9];
  $("#viewer").style.aspectRatio = `${w} / ${h}`;

  if (job.status === "done") {
    progress.hidden = true;
    player.hidden = false;
    if (player.dataset.src !== job.video_url) {
      player.dataset.src = job.video_url;
      player.src = job.video_url;
      player.poster = job.poster_url || "";
    }
  } else {
    if (!player.paused) player.pause();
    player.hidden = true;
    player.dataset.src = "";
    progress.hidden = false;
    const active = ACTIVE.has(job.status);
    progress.querySelector(".spinner").hidden = !active;
    $("#viewer-cancel").hidden = !active;
    $("#viewer-message").textContent = active ? job.message : job.status === "failed" ? "Generation failed" : "Cancelled";
    $("#viewer-bar").style.width = `${Math.round(job.progress * 100)}%`;
    const sub = [];
    if (job.status === "running") sub.push(`${Math.round(job.progress * 100)}%`);
    if (job.elapsed && active) sub.push(`${fmtSeconds(job.elapsed)} elapsed`);
    $("#viewer-sub").textContent = sub.join(" · ");
  }

  $("#detail-prompt").textContent = job.request.prompt;
  const chips = $("#detail-chips");
  chips.textContent = "";
  const r = job.result || {};
  const engine = state.engines[job.request.engine];
  const add = (text) => text && chips.append(el("span", { class: "chip" }, text));
  add(r.engine_label || engine?.label || job.request.engine);
  add(job.request.mode === "i2v" ? "image → video" : "text → video");
  add(r.width ? `${r.width}×${r.height}` : `${job.request.resolution} ${job.request.aspect}`);
  add(r.fps ? `${r.fps} fps` : "");
  add(`${r.duration || job.request.duration} s`);
  if (r.has_audio) add("sound");
  add(`seed ${job.request.seed}`);
  if (r.generate_seconds) add(`generated in ${fmtSeconds(r.generate_seconds)}`);
  if (r.size_bytes) add(`${(r.size_bytes / 1e6).toFixed(1)} MB`);

  const enhanced = $("#detail-enhanced");
  enhanced.hidden = !r.enhanced_prompt;
  $("#detail-enhanced-text").textContent = r.enhanced_prompt || "";
  const error = $("#detail-error");
  error.hidden = job.status !== "failed";
  error.textContent = job.error || "";

  const download = $("#btn-download");
  download.hidden = job.status !== "done";
  if (job.video_url) {
    download.href = job.video_url;
    download.download = `videogen-${job.id}.mp4`;
  }
  $("#btn-extend").hidden = job.status !== "done";
  $("#btn-delete").hidden = ACTIVE.has(job.status);
}

function currentJob() {
  return state.jobs.find((j) => j.id === state.selected);
}

async function reuse(job) {
  const q = job.request;
  $("#prompt").value = q.prompt;
  updateCount();
  if (state.engines[q.engine]) selectEngine(q.engine);
  setSeg("resolution", q.resolution);
  $("#duration").value = String(Math.round(q.duration));
  updateDuration();
  $("#seed").value = q.seed >= 0 ? String(q.seed) : "";
  $("#negative").value = q.negative_prompt || "";
  $("#steps").value = q.steps ?? "";
  $("#guidance").value = q.guidance ?? "";
  $("#audio").checked = q.audio;
  $("#enhance").checked = q.enhance_prompt;
  $("#upscaler").value = q.upscaler;
  $("#smooth").value = String(q.smooth_fps || 0);
  if (job.input_url) {
    const blob = await (await fetch(job.input_url)).blob();
    setMode("i2v");
    setImage(blob);
  } else {
    setMode("t2v");
    setImage(null);
  }
  setSeg("aspect", q.aspect);
  updateEstimate();
  $("#prompt").focus();
}

function lastFrame(url) {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.muted = true;
    video.preload = "auto";
    video.src = url;
    video.addEventListener("error", () => reject(new Error("Could not read the video")));
    video.addEventListener("loadedmetadata", () => {
      video.currentTime = Math.max(0, video.duration - 0.05);
    });
    video.addEventListener("seeked", () => {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0);
      canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Could not capture frame"))), "image/png");
    }, { once: true });
  });
}

async function extend(job) {
  try {
    const blob = await lastFrame(job.video_url);
    await reuse({ ...job, input_url: null });
    setMode("i2v");
    setImage(blob);
    $("#seed").value = "";
    $("#prompt").focus();
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (err) {
    showError(err.message);
  }
}

function updateCount() {
  $("#prompt-count").textContent = `${$("#prompt").value.length} / 4000`;
}

// ---------- wiring ----------

function bind() {
  for (const tab of document.querySelectorAll(".mode [role=tab]")) {
    tab.addEventListener("click", () => setMode(tab.dataset.mode));
  }
  for (const group of ["resolution", "aspect"]) {
    $(`#${group}`).addEventListener("click", (e) => {
      const button = e.target.closest("[role=radio]");
      if (!button) return;
      setSeg(group, button.dataset.value);
      updateEstimate();
    });
  }
  $("#duration").addEventListener("input", updateDuration);
  for (const id of ["#audio", "#smooth"]) $(id).addEventListener("change", updateEstimate);
  $("#prompt").addEventListener("input", updateCount);
  $("#form").addEventListener("submit", submit);
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") submit(e);
  });
  $("#seed-dice").addEventListener("click", () => {
    $("#seed").value = String(Math.floor(Math.random() * 2 ** 31));
  });

  const dz = $("#dropzone");
  const input = $("#image-input");
  dz.addEventListener("click", (e) => {
    if (e.target.closest("#dz-clear")) return;
    input.click();
  });
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => input.files[0] && setImage(input.files[0]));
  $("#dz-clear").addEventListener("click", (e) => {
    e.stopPropagation();
    input.value = "";
    setImage(null);
  });
  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("drag");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("drag");
    const file = [...e.dataTransfer.files].find((f) => f.type.startsWith("image/"));
    if (file) setImage(file);
  });
  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (!item) return;
    setMode("i2v");
    setImage(item.getAsFile());
  });

  $("#viewer-cancel").addEventListener("click", async () => {
    const job = currentJob();
    if (job) await api(`/api/jobs/${job.id}/cancel`, { method: "POST" }).catch((e) => showError(e.message));
    refreshJobs();
  });
  $("#btn-reuse").addEventListener("click", () => currentJob() && reuse(currentJob()));
  $("#btn-extend").addEventListener("click", () => currentJob() && extend(currentJob()));
  $("#btn-delete").addEventListener("click", async () => {
    const job = currentJob();
    if (!job || !confirm("Delete this video permanently?")) return;
    try {
      await api(`/api/jobs/${job.id}`, { method: "DELETE" });
      state.selected = null;
    } catch (err) {
      showError(err.message);
    }
    refreshJobs();
  });
}

async function init() {
  bind();
  loadPrefs();
  updateCount();
  try {
    state.info = await api("/api/info");
  } catch (err) {
    showError(`Could not reach the VideoGen server: ${err.message}`);
    return;
  }
  for (const engine of state.info.engines) state.engines[engine.id] = engine;
  renderHardware(state.info.hardware);
  renderEngines();
  const saved = store.get("engine", null);
  selectEngine(state.engines[saved] ? saved : state.info.default_engine);
  setMode("t2v");
  await refreshJobs();
}

init();
