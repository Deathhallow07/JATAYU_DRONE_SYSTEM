// =============================================================================
//  Launching a run from the console.
//
//  The relay is the one process that is always up and that the browser can
//  reach, so it is the one that owns starting a run: the operator picks a clip
//  and a flight log in the GUI and this spawns the three processes that turn
//  those two files into a populated console.
//
//      run_pipeline.py     the REAL Pipeline_yoloe_logs_robust.py, with the
//                          chosen video and CSV patched into its config
//      mission_watcher.py  tails the run folder the pipeline creates, so
//                          casualties appear as they are found
//      telem_csv.py        replays the CSV as `telemetry`, in step with the
//                          frame the pipeline is on
//
//  Three processes rather than one because the imports do not fit in one
//  interpreter on either of the machines this runs on: the pipeline needs
//  ultralytics + onnxruntime + sklearn, the producers need python-socketio,
//  and no env here has all of them. PIPELINE_PY and GUI_PY are separate for
//  exactly that reason.
//
//  Everything is spawned with `detached` so it gets its own process group and
//  Stop can signal the whole tree. The pipeline holds a CUDA context and an
//  MJPEG server; a SIGINT to the leader alone leaves both behind, and the next
//  run then fails to bind STREAM_PORT for reasons that look like nothing to do
//  with it.
// =============================================================================

const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const REPO = path.join(__dirname, "..");
const PY_DIR = path.join(REPO, "Python");
// The bundled pipeline: script, weights and the modules it imports all ship
// inside this checkout, so the console runs on a fresh machine with nothing
// to install beside it. Python/.env still overrides any of them.
const PIPELINE_DIR = path.join(REPO, "pipeline");
// The bundled demo footage — see sampleRun().
const SAMPLE_DIR = process.env.SAMPLE_DIR || path.join(PIPELINE_DIR, "1xzoom1.5ms20m");
const SAMPLE_START_FRAME = Number(process.env.SAMPLE_START_FRAME || 3600);

const firstExisting = (...candidates) =>
  candidates.find((p) => p && fs.existsSync(p)) || candidates[candidates.length - 1];

const HOME = process.env.HOME || "";

// ── Paths ────────────────────────────────────────────────────────────────────
//
// Every one of these is overridable in Python/.env. The defaults are this
// machine's layout, so a fresh checkout on it runs with no configuration —
// and a machine where they are wrong says which one is missing (see probe())
// instead of failing inside a Python traceback thirty seconds later.

const config = {
  // The pipeline being visualised. Not modified — run_pipeline.py rewrites a
  // copy of its source in memory. See that file for why.
  pipelineScript: process.env.PIPELINE_SCRIPT ||
    path.join(PIPELINE_DIR, "Pipeline_yoloe_logs_robust.py"),

  // Interpreter with ultralytics + torch + onnxruntime + sklearn.
  pipelinePy: process.env.PIPELINE_PY ||
    firstExisting(path.join(HOME, "miniconda3/envs/work_env/bin/python"), "python3"),

  // Interpreter with python-socketio + opencv, for the watcher and telemetry.
  guiPy: process.env.GUI_PY ||
    firstExisting(path.join(HOME, "miniconda3/envs/gcs_gui/bin/python"), "python3"),

  yoloeWeights: process.env.YOLOE_WEIGHTS ||
    firstExisting(path.join(PIPELINE_DIR, "yoloe-26s-seg.pt"), ""),
  reidCkpt: process.env.REID_CKPT ||
    firstExisting(path.join(PIPELINE_DIR, "epoch_003.pth"), ""),

  // DISPLAY_MODE="stream" is what puts the annotated, geolocalised frames in
  // the Geolocalisation raw feed: the browser reads this MJPEG server directly
  // rather than having every frame base64'd through the relay.
  streamPort: Number(process.env.PIPELINE_STREAM_PORT || 8080),

  // Where the BROWSER should look for that stream. Normally derived from the
  // port above, so the port is configured once and the two cannot drift apart
  // — a console pointed at 8080 while the pipeline serves 8081 shows an empty
  // feed and no error, because an <img> that never loads says nothing.
  //
  // Set PIPELINE_STREAM_URL only when the pipeline runs on a different host
  // from the browser, where a port alone is not enough to find it.
  streamUrl: process.env.PIPELINE_STREAM_URL || null,

  // Directories the file picker may browse. A colon-separated list, like PATH.
  // The browser sends absolute paths, so this is a jail, not a convenience:
  // resolveMedia() refuses anything that resolves outside it.
  //
  // Defaults to "/" — the whole filesystem, as the relay user can see it —
  // because footage lives wherever it lives and a picker that cannot reach it
  // is not a picker. Narrow it to the directories that actually hold
  // recordings if this is ever exposed beyond a trusted network.
  mediaRoots: (process.env.MEDIA_ROOTS || "/")
    .split(":").map((p) => p.trim()).filter(Boolean).map((p) => path.resolve(p)),

  // Extra import directories for the pipeline, colon-separated. It hardcodes
  // sys.path.append() for the machine it was written on, so a module it
  // imports from outside its own folder (coordinate_transformer, typically)
  // needs pointing at here.
  pipelinePythonPath: process.env.PIPELINE_PYTHONPATH || PIPELINE_DIR,

  // The monitoring feed's encoding. The pipeline ships it as full-resolution
  // JPEG at OpenCV's default quality 95 — measured at 779 KB a frame, 113 Mbps
  // — which only a LAN can carry. Over anything slower the browser falls
  // behind and the video pane shows a different minute of the flight from the
  // map beside it. The DETECTOR is untouched; this is the copy people watch.
  streamQuality: Number(process.env.PIPELINE_STREAM_QUALITY || 70),
  streamWidth: Number(process.env.PIPELINE_STREAM_WIDTH || 1280),

  depthBackend: process.env.DEPTH_BACKEND || "stub",
  segBackend: process.env.SEG_BACKEND || "stub",
};

const VIDEO_EXT = new Set([".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm", ".mpg", ".mpeg"]);
const DATA_EXT = new Set([".csv"]);

// Lines of child stdout/stderr kept for the launcher dialog. Enough to hold a
// Python traceback, which is the thing anyone actually needs to read here.
const OUTPUT_LIMIT = 300;

// ── File browser ─────────────────────────────────────────────────────────────

/**
 * Resolve a browser-supplied path, refusing anything outside MEDIA_ROOTS.
 *
 * Same rule as the artifact server: resolve first, then test containment with
 * path.relative. A prefix string test would accept "/home/me/scp_evil" for a
 * root of "/home/me/scp".
 */
const resolveMedia = (p) => {
  if (!p) return null;
  const abs = path.resolve(p);
  const ok = config.mediaRoots.some((root) => {
    const rel = path.relative(root, abs);
    return rel === "" || (!rel.startsWith("..") && !path.isAbsolute(rel));
  });
  return ok ? abs : null;
};

const entryKind = (name) => {
  const ext = path.extname(name).toLowerCase();
  if (VIDEO_EXT.has(ext)) return "video";
  if (DATA_EXT.has(ext)) return "csv";
  return null;
};

/**
 * One directory listing for the picker.
 *
 * Only videos and CSVs are returned — a flight recording directory also holds
 * bags and calibration dumps, and a picker that lists them invites choosing
 * one. `parent` is omitted at a root so the dialog cannot walk out of the jail
 * by clicking "..".
 */
function browse(target) {
  const abs = target ? resolveMedia(target) : null;

  // No path (or one outside the jail): start at home rather than at the top of
  // the tree. With MEDIA_ROOTS at "/" the roots listing is a single "/" entry,
  // and making someone walk down from there to their footage is worse than the
  // jail this replaced.
  if (!abs) {
    const home = resolveMedia(HOME);
    if (home && fs.existsSync(home)) return { ...browse(home), shortcuts: shortcuts() };
    return {
      path: null,
      roots: config.mediaRoots,
      shortcuts: shortcuts(),
      dirs: config.mediaRoots
        .filter((r) => fs.existsSync(r))
        .map((r) => ({ name: r, path: r })),
      files: [],
    };
  }

  let stat;
  try {
    stat = fs.statSync(abs);
  } catch {
    return { path: abs, error: "not found", roots: config.mediaRoots, dirs: [], files: [] };
  }

  // A file was passed: list the directory it is in, so "open next to this one"
  // works without the caller having to strip the basename.
  const dir = stat.isDirectory() ? abs : path.dirname(abs);

  let names;
  try {
    names = fs.readdirSync(dir, { withFileTypes: true });
  } catch (err) {
    return { path: dir, error: err.code || "unreadable", roots: config.mediaRoots, dirs: [], files: [] };
  }

  const dirs = [];
  const files = [];
  for (const d of names) {
    if (d.name.startsWith(".")) continue;
    const full = path.join(dir, d.name);
    if (d.isDirectory()) {
      dirs.push({ name: d.name, path: full });
      continue;
    }
    const kind = entryKind(d.name);
    if (!kind) continue;
    let size = null;
    let mtime = null;
    try {
      const s = fs.statSync(full);
      size = s.size;
      mtime = s.mtimeMs;
    } catch { /* a file that vanished between readdir and stat is just gone */ }
    files.push({ name: d.name, path: full, kind, size, mtime });
  }

  const byName = (a, b) => a.name.localeCompare(b.name, undefined, { numeric: true });
  dirs.sort(byName);
  files.sort(byName);

  const parent = resolveMedia(path.dirname(dir));
  return {
    path: dir,
    parent: parent && parent !== dir ? parent : null,
    roots: config.mediaRoots,
    shortcuts: shortcuts(),
    dirs,
    files,
  };
}

/**
 * One-click jumps, so browsing a whole filesystem is not a walk down from "/".
 *
 * Only places that exist and are inside the jail are offered — an entry that
 * 403s on click is worse than no entry.
 */
function shortcuts() {
  const candidates = [
    { name: "Home", path: HOME },
    { name: "Media", path: "/media" },
    { name: "Mounts", path: "/mnt" },
    { name: "Data", path: "/data" },
    ...config.mediaRoots.map((r) => ({ name: r === "/" ? "Filesystem" : path.basename(r) || r, path: r })),
  ];
  const seen = new Set();
  return candidates.filter((c) => {
    if (!c.path || seen.has(c.path)) return false;
    seen.add(c.path);
    return resolveMedia(c.path) !== null && fs.existsSync(c.path);
  });
}

/**
 * The CSV that goes with a clip, if an obvious one is sitting next to it.
 *
 * These recordings are written as a pair — `1xzoom1.5ms25m.mp4` beside
 * `1xzoom1.5ms25m.csv` — so the dialog can preselect the right log instead of
 * making the operator find it again in the same folder. Same-stem first, then
 * a lone CSV in the directory; anything more ambiguous is left to the operator.
 */
function suggestCsv(videoPath) {
  const abs = resolveMedia(videoPath);
  if (!abs) return null;
  const dir = path.dirname(abs);
  const stem = path.basename(abs, path.extname(abs));

  const sameStem = path.join(dir, `${stem}.csv`);
  if (fs.existsSync(sameStem)) return sameStem;

  let csvs = [];
  try {
    csvs = fs.readdirSync(dir).filter((n) => n.toLowerCase().endsWith(".csv"));
  } catch { return null; }
  return csvs.length === 1 ? path.join(dir, csvs[0]) : null;
}

// ── Orphans ──────────────────────────────────────────────────────────────────

/**
 * Kill producers left behind by a previous relay.
 *
 * Children are spawned `detached` so Stop can signal the whole process group.
 * The cost of that is they do NOT die with the relay when it goes away
 * abruptly — `tmux kill-session`, a SIGKILL, a crash — and `Bus` reconnects
 * forever by design, because a producer must never stop just because the GUI
 * blinked. So an orphaned telem_csv rejoins the NEXT relay and keeps
 * publishing telemetry from the run it was started for.
 *
 * With two producers on the socket the console does not see a conflict, it
 * sees one drone: positions from the finished run and the live one arrive
 * interleaved at 4 Hz each, and the marker alternates between the two places
 * on successive samples. An orphaned mission_watcher does the same to GIDs and
 * log lines.
 *
 * Matched on the absolute path of THIS checkout's scripts, so a second copy of
 * the console running from another directory is left alone.
 */
function reapOrphans(log, keepPids = new Set()) {
  const ours = ["run_pipeline.py", "mission_watcher.py", "telem_csv.py",
                "depth_worker.py", "seg_worker.py"]
    .map((f) => path.join(PY_DIR, f.includes("worker") ? path.join("workers", f) : f));

  let pids;
  try {
    pids = fs.readdirSync("/proc").filter((d) => /^\d+$/.test(d));
  } catch {
    return 0;   // not Linux, or no procfs — nothing to do
  }

  let killed = 0;
  for (const pid of pids) {
    const n = Number(pid);
    if (n === process.pid || keepPids.has(n)) continue;
    let cmd;
    try {
      cmd = fs.readFileSync(`/proc/${pid}/cmdline`, "utf8").split("\0").join(" ");
    } catch {
      continue;   // exited between readdir and read
    }
    if (!ours.some((script) => cmd.includes(script))) continue;

    log.warn({ pid: n, cmd: cmd.slice(0, 120) }, "reaping orphaned producer");
    try { process.kill(-n, "SIGKILL"); } catch {
      try { process.kill(n, "SIGKILL"); } catch { /* already gone */ }
    }
    killed++;
  }
  return killed;
}

// ── Readiness ────────────────────────────────────────────────────────────────

/**
 * What is and is not in place, for the launcher dialog to show before the
 * operator commits to a run.
 *
 * Checked here rather than at startup because these paths live on a different
 * machine from the one this is usually developed on, and a relay that refused
 * to boot over a missing checkpoint would take the whole console down with it.
 */
/**
 * The footage that ships with the checkout.
 *
 * A clip and its flight log live in pipeline/1xzoom1.5ms20m, so a machine that
 * has this folder and nothing else can still run the console end to end. The
 * directory is read rather than the filenames hardcoded: the pair is swapped
 * for other footage from time to time, and a stale constant would leave the
 * Sample button pointing at a file that is no longer there.
 *
 * `startFrame` skips the first minute. The recording opens on the ground
 * before the drone is up, and a demo that shows an empty field for a minute
 * reads as a console that is not working.
 */
function sampleRun() {
  let names;
  try { names = fs.readdirSync(SAMPLE_DIR); } catch { return null; }
  const pick = (exts) => {
    const hit = names.filter((n) => exts.has(path.extname(n).toLowerCase())).sort()[0];
    return hit ? path.join(SAMPLE_DIR, hit) : null;
  };
  const video = pick(VIDEO_EXT);
  if (!video) return null;
  return { video, csv: pick(DATA_EXT), startFrame: SAMPLE_START_FRAME };
}

function probe() {
  const isFile = (p) => {
    try { return Boolean(p) && fs.statSync(p).isFile(); } catch { return false; }
  };

  /**
   * An interpreter has to be an executable FILE.
   *
   * `fs.existsSync` is true for a directory, so pointing PIPELINE_PY at a conda
   * ENV rather than at the python inside it passed this check and then failed
   * at spawn with `EACCES` — a permission error for what is really a wrong
   * path, reported a minute later in a different place. `reason` carries the
   * fix rather than just the verdict.
   */
  const interpreter = (p) => {
    if (!p) return { path: p, ok: false, reason: "not set" };
    if (p === "python3") return { path: p, ok: true };
    let st;
    try { st = fs.statSync(p); } catch { return { path: p, ok: false, reason: "not found" }; }
    if (st.isDirectory()) {
      const inside = path.join(p, "bin", "python");
      return {
        path: p,
        ok: false,
        reason: isFile(inside)
          ? `that is a directory — use the interpreter inside it: ${inside}`
          : "that is a directory, not an interpreter",
      };
    }
    try {
      fs.accessSync(p, fs.constants.X_OK);
    } catch {
      return { path: p, ok: false, reason: "not executable" };
    }
    return { path: p, ok: true };
  };

  const file = (p) => (p ? { path: p, ok: isFile(p), ...(isFile(p) ? {} : { reason: "not found" }) }
                         : { path: p, ok: false, reason: "not set" });

  return {
    pipelineScript: file(config.pipelineScript),
    pipelinePy: interpreter(config.pipelinePy),
    guiPy: interpreter(config.guiPy),
    yoloeWeights: file(config.yoloeWeights),
    reidCkpt: file(config.reidCkpt),
    streamPort: config.streamPort,
    // Absolute when pinned to another host, otherwise a port the browser
    // resolves against whatever host it reached the console on.
    streamUrl: config.streamUrl,
    mediaRoots: config.mediaRoots,
    // null when the checkout has no bundled footage; the Sample button hides.
    sample: sampleRun(),
  };
}

// ── The run ──────────────────────────────────────────────────────────────────

class Launcher {
  /**
   * @param {object} opts
   * @param {string} opts.missionRoot  where run folders are created and served from
   * @param {string} opts.guiUrl       relay URL the producers connect back to
   * @param {Function} opts.onStatus   called with the full status on every change
   * @param {object} opts.log          pino child logger
   */
  constructor({ missionRoot, guiUrl, onStatus, log }) {
    this.missionRoot = missionRoot;
    this.guiUrl = guiUrl;
    this.onStatus = onStatus;
    this.log = log;

    this.state = "idle";      // idle | starting | running | stopping | stopped | error
    this.run = null;          // { video, csv, startFrame, workers, startedAt }
    this.error = null;
    this.procs = new Map();   // name -> { child, state, exitCode }
    this.output = [];         // [{ proc, line, ts }]
  }

  status() {
    return {
      state: this.state,
      run: this.run,
      error: this.error,
      config: probe(),
      procs: [...this.procs.entries()].map(([name, p]) => ({
        name, pid: p.child?.pid ?? null, state: p.state, exitCode: p.exitCode ?? null,
      })),
      output: this.output,
    };
  }

  _emit() {
    try {
      this.onStatus(this.status());
    } catch (err) {
      this.log.error({ err }, "run status listener failed");
    }
  }

  _say(proc, line) {
    this.output.push({ proc, line, ts: Date.now() });
    if (this.output.length > OUTPUT_LIMIT) {
      this.output.splice(0, this.output.length - OUTPUT_LIMIT);
    }
  }

  /**
   * Spawn one child and wire its output into the run's transcript.
   *
   * stdout and stderr both go to the same buffer: a Python traceback is on
   * stderr, the pipeline's banner is on stdout, and keeping them apart only
   * makes the dialog show half the story.
   */
  _spawn(name, cmd, args, env) {
    const child = spawn(cmd, args, {
      cwd: PY_DIR,
      detached: true,
      env: { ...process.env, ...env },
      stdio: ["ignore", "pipe", "pipe"],
    });

    const entry = { child, state: "running", exitCode: null };
    this.procs.set(name, entry);
    this.log.info({ name, cmd, args, pid: child.pid }, "spawned");

    const pump = (stream) => {
      let buf = "";
      stream.setEncoding("utf8");
      stream.on("data", (chunk) => {
        buf += chunk;
        const lines = buf.split("\n");
        buf = lines.pop();
        for (const line of lines) {
          if (line.trim()) this._say(name, line.trimEnd());
        }
        this._emit();
      });
    };
    pump(child.stdout);
    pump(child.stderr);

    child.on("error", (err) => {
      entry.state = "error";
      this._say(name, `failed to start: ${err.message}`);
      // The pipeline is the run. If it cannot start, nothing else has a job.
      if (name === "pipeline") {
        this.state = "error";
        this.error = `${name}: ${err.message}`;
        this.stop({ reason: "pipeline failed to start" });
      }
      this._emit();
    });

    child.on("exit", (code, signal) => {
      entry.state = signal ? "killed" : code === 0 ? "finished" : "error";
      entry.exitCode = code;
      this._say(name, signal ? `exited on ${signal}` : `exited with code ${code}`);
      this.log.info({ name, code, signal }, "child exited");

      // The pipeline finishing IS the run finishing — the watcher would
      // otherwise sit tailing a log nobody is writing, and the telemetry
      // producer would hold the drone on its last frame forever.
      if (name === "pipeline" && this.state !== "stopping") {
        this.state = code === 0 ? "stopped" : "error";
        if (code !== 0) this.error = `pipeline exited with code ${code}`;
        this.stop({ reason: "pipeline finished", keepState: true });
      }
      this._emit();
    });

    return child;
  }

  /**
   * Start a run.
   *
   * @param {string}  video       clip to run the pipeline on
   * @param {?string} csv         flight log; without it there is no geolocalisation
   * @param {number}  startFrame  enter the clip part-way through
   * @param {boolean} workers     also run the depth + traversability feeds
   * @param {?number} fps         video fps, for a CSV with no frame column
   */
  start({ video, csv, startFrame = 0, workers = false, fps = 30 }) {
    if (this.state === "starting" || this.state === "running") {
      return { ok: false, error: "a run is already in progress" };
    }

    const videoPath = resolveMedia(video);
    if (!videoPath || !fs.existsSync(videoPath)) {
      return { ok: false, error: `video not found or outside MEDIA_ROOTS: ${video}` };
    }
    if (!VIDEO_EXT.has(path.extname(videoPath).toLowerCase())) {
      return { ok: false, error: `not a video file: ${path.basename(videoPath)}` };
    }

    let csvPath = null;
    if (csv) {
      csvPath = resolveMedia(csv);
      if (!csvPath || !fs.existsSync(csvPath)) {
        return { ok: false, error: `CSV not found or outside MEDIA_ROOTS: ${csv}` };
      }
    }

    // Checked before spawning anything: otherwise three children all fail with
    // the same error and the dialog shows it three times without saying that
    // one setting is behind all of them.
    const cfg = probe();
    for (const [key, env] of [["pipelineScript", "PIPELINE_SCRIPT"],
                              ["pipelinePy", "PIPELINE_PY"],
                              ["guiPy", "GUI_PY"]]) {
      if (!cfg[key].ok) {
        return {
          ok: false,
          error: `${env}=${cfg[key].path || "<unset>"} — ${cfg[key].reason}. ` +
                 `Fix it in Python/.env and restart the relay.`,
        };
      }
    }

    fs.mkdirSync(this.missionRoot, { recursive: true });

    // Anything still alive from a previous relay would publish alongside the
    // run about to start. See reapOrphans().
    const reaped = reapOrphans(this.log);
    if (reaped) this.log.warn({ reaped }, "reaped orphaned producers before starting");

    this.state = "starting";
    this.error = null;
    this.procs.clear();
    this.output = [];
    this.run = {
      video: videoPath,
      csv: csvPath,
      startFrame: Number(startFrame) || 0,
      workers: Boolean(workers),
      fps: Number(fps) || 30,
      missionRoot: this.missionRoot,
      streamPort: config.streamPort,
      startedAt: new Date().toISOString(),
    };

    const sharedEnv = { GUI_URL: this.guiUrl, MISSION_ROOT: this.missionRoot };

    // 1. The pipeline. Everything else is an observer of what it writes.
    const pipeArgs = [
      path.join(PY_DIR, "run_pipeline.py"),
      "--pipeline", config.pipelineScript,
      "--video", videoPath,
      "--out", this.missionRoot,
      "--start-frame", String(this.run.startFrame),
      "--display", "stream",
      "--stream-port", String(config.streamPort),
    ];
    if (csvPath) pipeArgs.push("--telem", csvPath);
    if (config.yoloeWeights) pipeArgs.push("--weights", config.yoloeWeights);
    if (config.reidCkpt) pipeArgs.push("--reid", config.reidCkpt);
    if (config.pipelinePythonPath) pipeArgs.push("--pythonpath", config.pipelinePythonPath);
    pipeArgs.push("--stream-quality", String(config.streamQuality));
    pipeArgs.push("--stream-width", String(config.streamWidth));

    this._spawn("pipeline", config.pipelinePy, pipeArgs, {
      ...sharedEnv,
      // Unbuffered, or the dialog shows nothing at all until the run ends and
      // the first thing anyone sees of a crash is the exit code.
      PYTHONUNBUFFERED: "1",
    });

    // 2. The watcher. It waits for a run folder to appear, so starting it now
    //    rather than after the pipeline's slow model load is not a race.
    this._spawn("watcher", config.guiPy, [
      path.join(PY_DIR, "mission_watcher.py"),
      "--root", this.missionRoot,
      // Same reason as telem_csv's --since: until the pipeline creates its
      // folder, the newest one under the root is the LAST run's, complete with
      // all of its casualties. Attaching there fills the console with a
      // finished run while the live one is never read.
      "--since", String(Date.now() / 1000 - 5),
    ], { ...sharedEnv, PYTHONUNBUFFERED: "1" });

    // 3. Telemetry, only with a CSV. Without one there is nothing to fly the
    //    drone marker with, and a producer emitting nothing would still claim
    //    the System pane's "telemetry live" light.
    if (csvPath) {
      this._spawn("telem", config.guiPy, [
        path.join(PY_DIR, "telem_csv.py"),
        "--csv", csvPath,
        "--root", this.missionRoot,
        "--start-frame", String(this.run.startFrame),
        "--fps", String(this.run.fps),
        // Only follow a run folder created from here on. Without it the
        // telemetry clock can latch onto the PREVIOUS run's finished log,
        // which is the newest one under the root until the pipeline gets
        // around to making its own — and then the drone sits on that run's
        // last frame for the whole mission.
        "--since", String(Date.now() / 1000 - 5),
      ], { ...sharedEnv, PYTHONUNBUFFERED: "1" });
    }

    // 4. Depth + traversability, off the pipeline's own MJPEG server so all
    //    three raw feeds show the same frame. Opt-in: on a CPU box they are
    //    two more model loads competing with the pipeline for the same cores.
    if (workers) {
      const source = `http://127.0.0.1:${config.streamPort}/`;
      this._spawn("depth", config.guiPy, [
        path.join(PY_DIR, "workers", "depth_worker.py"),
        "--source", source, "--backend", config.depthBackend,
      ], { ...sharedEnv, PYTHONUNBUFFERED: "1" });
      this._spawn("seg", config.guiPy, [
        path.join(PY_DIR, "workers", "seg_worker.py"),
        "--source", source, "--backend", config.segBackend,
      ], { ...sharedEnv, PYTHONUNBUFFERED: "1" });
    }

    this.state = "running";
    this._emit();
    return { ok: true, run: this.run };
  }

  /**
   * Stop everything this run started.
   *
   * SIGINT first, because the pipeline installs a handler that writes the
   * mission summary and closes the run folder cleanly — a SIGKILL there loses
   * the final GPS block for every casualty found. SIGKILL only for what is
   * still alive after the grace period.
   */
  stop({ reason = "stopped by the operator", keepState = false } = {}) {
    if (this.procs.size === 0) {
      if (!keepState) this.state = "idle";
      this._emit();
      return { ok: true };
    }

    if (!keepState) this.state = "stopping";
    this._say("relay", reason);
    this.log.info({ reason }, "stopping run");

    const groups = [];
    for (const [name, p] of this.procs) {
      const pid = p.child?.pid;
      if (!pid || p.state !== "running") continue;
      groups.push({ name, pid });
      // Negative pid signals the whole process group. `detached: true` at
      // spawn is what makes the child a group leader and this possible.
      try { process.kill(-pid, "SIGINT"); } catch { /* already gone */ }
    }

    setTimeout(() => {
      for (const { name, pid } of groups) {
        const p = this.procs.get(name);
        if (!p || p.state !== "running") continue;
        this.log.warn({ name, pid }, "did not exit on SIGINT — SIGKILL");
        try { process.kill(-pid, "SIGKILL"); } catch { /* already gone */ }
      }
      if (this.state === "stopping") this.state = "stopped";
      this._emit();
    }, 6000).unref();

    this._emit();
    return { ok: true };
  }
}

module.exports = { Launcher, browse, suggestCsv, probe, resolveMedia, reapOrphans, config };
