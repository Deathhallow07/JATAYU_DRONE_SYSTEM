// =============================================================================
//  SAR-GUI relay
//
//  Same shape as the existing GCS backend: Python publishes on one namespace,
//  browsers listen on another, and the server does not interpret the payloads.
//
//  Two deliberate differences from that backend:
//
//  1. It keeps a BOUNDED SNAPSHOT (recent log lines, the GID map, the latest
//     frame per worker). The existing relay is stateless because telemetry
//     re-arrives at 10 Hz — a browser that reloads is current within 100 ms.
//     Here a GID may be emitted once, an hour into the mission, so a reload
//     with no snapshot shows an empty screen for the rest of the run. The
//     cache is capped (see SNAPSHOT_LIMITS) and is not persisted.
//
//  2. It serves the pipeline's RUN_FOLDER over HTTP so the browser can show
//     representative.jpg / crop_frame_*.jpg without the images being base64'd
//     through the socket. Confined to MISSION_ROOT — see resolveArtifact().
// =============================================================================

const express = require("express");
const http = require("http");
const https = require("https");
const cors = require("cors");
const socketIo = require("socket.io");
const fs = require("fs");
const path = require("path");
const pino = require("pino");

// Before ./pipeline: that module reads its paths out of process.env at require
// time, so loading the .env after it would leave every one of them on its
// built-in default and say nothing about it.
require("dotenv").config({ path: path.join(__dirname, "..", "Python", ".env"), quiet: true });

const { Launcher, browse, suggestCsv, probe, reapOrphans } = require("./pipeline");

const logger = pino({
  transport: {
    target: "pino-pretty",
    options: { translateTime: "SYS:HH:MM:ss.l", ignore: "pid,hostname" },
  },
  level: process.env.LOG_LEVEL || "info",
});

// ── Config ───────────────────────────────────────────────────────────────────

const PORT = Number(process.env.GUI_PORT || 7100);

// Every artifact path the browser asks for must resolve inside this directory.
// A run launched from the console writes here too: run_pipeline.py redirects
// the pipeline's RUN_FOLDER into this directory so the crops the browser asks
// for are inside the jail. Point it anywhere with MISSION_ROOT in Python/.env.
const MISSION_ROOT = path.resolve(
  process.env.MISSION_ROOT || path.join(__dirname, "..", "mission_logs")
);

const allowedOrigins = process.env.ALLOWED_ORIGINS
  ? process.env.ALLOWED_ORIGINS.split(",").map((o) => o.trim())
  : "*";

const SNAPSHOT_LIMITS = {
  logs: 2000,   // ring buffer of mission.log lines
  gids: 512,    // hard ceiling; a real mission is O(10)
};

// ── Snapshot ─────────────────────────────────────────────────────────────────

const snapshot = {
  mission: null,        // { runFolder, startedAt, replay, source }
  logs: [],             // [{ seq, ts, level, msg }]
  gids: new Map(),      // gid -> latest gid payload
  stats: null,          // MISSION_STATS mirror
  frames: {},           // worker -> latest frame payload (depth, seg, detect)
  telemetry: null,      // latest UAV telemetry (telem_worker.py / telem_csv.py)
  run: null,            // launcher status; see Back/pipeline.js
  pythonConnected: false,
};

let logSeq = 0;

const pushLog = (entry) => {
  entry.seq = ++logSeq;
  snapshot.logs.push(entry);
  if (snapshot.logs.length > SNAPSHOT_LIMITS.logs) {
    snapshot.logs.splice(0, snapshot.logs.length - SNAPSHOT_LIMITS.logs);
  }
  return entry;
};

const serializeSnapshot = () => ({
  mission: snapshot.mission,
  logs: snapshot.logs,
  gids: [...snapshot.gids.values()],
  stats: snapshot.stats,
  frames: snapshot.frames,
  telemetry: snapshot.telemetry,
  // Asked of the launcher rather than read from the mirror above: that mirror
  // is only written when a run CHANGES state, so on a relay that has not
  // launched anything yet it is null — and the dialog, which reads `config`
  // off this, then comes up with no readiness chips and no Sample button
  // until the first run. The launcher always knows its own state.
  run: snapshot.run ?? launcher.status(),
  pythonConnected: snapshot.pythonConnected,
});

const resetSnapshot = (mission) => {
  snapshot.mission = mission;
  snapshot.logs = [];
  snapshot.gids.clear();
  snapshot.stats = null;
  snapshot.frames = {};
  // Deliberately NOT cleared: telemetry describes the aircraft, not the run.
  // Blanking it on mission_start would drop the map's drone marker every time
  // a producer re-announces, for up to a second until the next 4 Hz sample.
  logSeq = 0;
};

// ── The run launcher ─────────────────────────────────────────────────────────
//
// Starting a run is a relay responsibility because the relay is the only
// process that is always up and that the browser can talk to. It owns the
// pipeline, the watcher and the telemetry producer for the duration of a run;
// see Back/pipeline.js.

const launcher = new Launcher({
  missionRoot: MISSION_ROOT,
  guiUrl: process.env.GUI_URL || `http://127.0.0.1:${PORT}`,
  log: logger.child({ ns: "launch" }),
  onStatus: (status) => {
    snapshot.run = status;
    reactNS.emit("run_status", status);
  },
});

// ── HTTP ─────────────────────────────────────────────────────────────────────

const app = express();
app.use(cors({ origin: allowedOrigins }));

/**
 * Map a browser-supplied artifact path onto a real file.
 *
 * The browser only ever sends paths relative to the run folder
 * ("gid_3/representative.jpg"), but it is the browser — nothing stops a
 * crafted "../../../etc/passwd". Resolve first, then require the result to be
 * inside MISSION_ROOT; a prefix test alone would accept a sibling directory
 * whose name merely starts with the root ("/missions_evil" vs "/missions").
 */
const resolveArtifact = (relPath) => {
  const base = snapshot.mission?.runFolder;
  if (!base) return null;

  const abs = path.resolve(base, relPath);
  const rel = path.relative(MISSION_ROOT, abs);

  if (rel.startsWith("..") || path.isAbsolute(rel)) return null;
  return abs;
};

app.get("/artifacts/*splat", (req, res) => {
  const abs = resolveArtifact(req.params.splat.join("/"));
  if (!abs) return res.status(403).end();

  res.sendFile(abs, (err) => {
    if (err && !res.headersSent) res.status(404).end();
  });
});

/**
 * Directory listing for the launcher's file picker.
 *
 * The browser cannot give a real path for a file the operator picked through
 * <input type="file"> — it only ever hands over a sandboxed blob — and the clip
 * has to be opened by the PIPELINE, on this machine, not uploaded. So the
 * picker browses the server's own filesystem, confined to MEDIA_ROOTS.
 */
app.get("/media", (req, res) => {
  const listing = browse(req.query.path ? String(req.query.path) : null);
  if (req.query.suggestCsvFor) {
    listing.suggestedCsv = suggestCsv(String(req.query.suggestCsvFor));
  }
  res.json(listing);
});

/** What the launcher needs in place, so the dialog can say what is missing. */
app.get("/launcher", (_req, res) => {
  res.json({ ...launcher.status(), missionRoot: MISSION_ROOT });
});

app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    python: snapshot.pythonConnected,
    mission: snapshot.mission,
    gids: snapshot.gids.size,
    missionRoot: MISSION_ROOT,
    run: snapshot.run?.state ?? "idle",
  });
});

// HTTPS when certs are present (matches the existing backend so the two can be
// reverse-proxied the same way), plain HTTP otherwise so a fresh clone runs.
let server;
try {
  server = https.createServer(
    {
      key: fs.readFileSync(path.join(__dirname, "server.key")),
      cert: fs.readFileSync(path.join(__dirname, "server.cert")),
    },
    app
  );
  logger.info("TLS certs found — serving HTTPS");
} catch {
  server = http.createServer(app);
  logger.warn("No server.key/server.cert — serving plain HTTP");
}

const io = socketIo(server, {
  maxHttpBufferSize: 2e7, // 20 MB: depth/segmentation frames arrive as base64 JPEG
  pingTimeout: 60000,
  pingInterval: 10000,
  cors: { origin: allowedOrigins, methods: ["GET", "POST"] },
});

// ── Namespaces ───────────────────────────────────────────────────────────────
//   /py     — mission watcher + model workers (producers)
//   /react  — browser panes (consumers + commands)

const pyNS = io.of("/py");
const reactNS = io.of("/react");

const pyLog = logger.child({ ns: "py" });

pyNS.on("connection", (socket) => {
  const role = socket.handshake.auth?.role || "unknown";
  pyLog.info({ id: socket.id, role }, "producer connected");

  snapshot.pythonConnected = true;
  reactNS.emit("producer_status", { connected: true, role });

  // A new mission run clears everything the panes are showing.
  //
  // Producers re-announce their mission on every reconnect (Bus.set_hello), so
  // this fires again for a run already in progress. Resetting on those would
  // throw away every GID and log line received so far — which is exactly what
  // happens when a producer starts before the relay is listening, drops its
  // first connect, and reconnects mid-run. Reset only when the run identity
  // actually changes.
  socket.on("mission_start", (m) => {
    const isSameRun =
      snapshot.mission &&
      snapshot.mission.runFolder === m?.runFolder &&
      snapshot.mission.startedAt === m?.startedAt;

    if (isSameRun) {
      snapshot.mission = m; // refresh speed/replay flags without dropping state
      pyLog.debug({ runFolder: m?.runFolder }, "mission_start (re-announce)");
    } else {
      resetSnapshot(m);
      pyLog.info({ runFolder: m?.runFolder, replay: m?.replay }, "mission_start");
    }
    reactNS.emit("mission_start", m);
  });

  // Logs arrive batched (the watcher coalesces a tail read into one emit)
  // so a pipeline dumping hundreds of lines a second costs one frame, not one
  // socket write per line.
  socket.on("logs", (batch) => {
    if (!Array.isArray(batch) || batch.length === 0) return;
    const stamped = batch.map(pushLog);
    reactNS.emit("logs", stamped);
  });

  socket.on("gid", (g) => {
    if (g?.gid == null) return;
    if (!snapshot.gids.has(g.gid) && snapshot.gids.size >= SNAPSHOT_LIMITS.gids) {
      pyLog.warn({ gid: g.gid }, "GID cache full — forwarding without caching");
    } else {
      snapshot.gids.set(g.gid, g);
    }
    pyLog.info({ gid: g.gid, event: g.event }, "gid");
    reactNS.emit("gid", g);
  });

  socket.on("stats", (s) => {
    snapshot.stats = s;
    reactNS.emit("stats", s);
  });

  // worker: "depth" | "seg" | "detect". One slot per worker — only the newest
  // frame is worth keeping, and holding a history of JPEGs would grow without
  // bound.
  socket.on("frame", (f) => {
    if (!f?.worker) return;
    snapshot.frames[f.worker] = f;
    reactNS.emit("frame", f);
  });

  // 4 Hz from telem_worker.py. Only the newest fix matters, so the snapshot
  // keeps one — a reload shows the drone where it is, not where it started.
  socket.on("telemetry", (t) => {
    if (t?.latitude == null || t?.longitude == null) return;
    snapshot.telemetry = t;
    reactNS.volatile.emit("telemetry", t);
  });

  socket.on("worker_status", (s) => {
    reactNS.emit("worker_status", s);
  });

  socket.on("disconnect", (reason) => {
    pyLog.warn({ id: socket.id, role, reason }, "producer disconnected");
    if (pyNS.sockets.size === 0) {
      snapshot.pythonConnected = false;
      reactNS.emit("producer_status", { connected: false, role });
    }
  });
});

const reactLog = logger.child({ ns: "react" });

reactNS.on("connection", (socket) => {
  reactLog.info({ id: socket.id }, "browser connected");

  // Sent unprompted so a pane is populated on mount without a round trip;
  // also available as an explicit request for a manual resync.
  socket.emit("snapshot", serializeSnapshot());
  socket.on("snapshot", (ack) => {
    if (typeof ack === "function") ack(serializeSnapshot());
  });

  // Commands are forwarded verbatim; the producers own their semantics.
  //   replay_control  { action: "play"|"pause"|"restart"|"seek", speed?, seq? }
  //   run_worker      { worker: "depth"|"seg", ...worker args }
  //   plan_path       { gid, start: {row, col} }
  for (const cmd of ["replay_control", "run_worker", "plan_path", "select_run"]) {
    socket.on(cmd, (payload, ack) => {
      if (pyNS.sockets.size === 0) {
        reactLog.warn({ cmd }, "no producer connected — command dropped");
        if (typeof ack === "function") ack({ ok: false, error: "no producer connected" });
        return;
      }
      reactLog.info({ cmd, payload }, "command");
      pyNS.emit(cmd, payload);
      if (typeof ack === "function") ack({ ok: true });
    });
  }

  // Launching a run is not a "forward it to whoever is listening" command like
  // the ones above: there is no producer to forward it to yet — starting the
  // producers is the whole point. So the relay handles these itself.
  socket.on("run_start", (payload, ack) => {
    reactLog.info({ payload }, "run_start");
    const result = launcher.start({
      video: payload?.video,
      csv: payload?.csv || null,
      startFrame: payload?.startFrame ?? 0,
      workers: Boolean(payload?.workers),
      fps: payload?.fps ?? 30,
    });
    if (!result.ok) reactLog.warn({ error: result.error }, "run_start refused");
    if (typeof ack === "function") ack(result);
  });

  socket.on("run_stop", (_payload, ack) => {
    reactLog.info("run_stop");
    const result = launcher.stop();
    if (typeof ack === "function") ack(result);
  });

  socket.on("run_status", (ack) => {
    if (typeof ack === "function") ack(launcher.status());
  });

  socket.on("disconnect", (reason) => reactLog.info({ id: socket.id, reason }, "browser disconnected"));
});

// ── Listen ───────────────────────────────────────────────────────────────────

server.listen(PORT, "0.0.0.0", () => {
  if (allowedOrigins === "*") {
    logger.warn("CORS is open (origin: *) — set ALLOWED_ORIGINS for anything routable");
  }
  logger.info(`SAR-GUI relay on :${PORT}`);
  logger.info(`Mission root: ${MISSION_ROOT}`);

  // Said once, at boot, rather than only when a launch fails: these paths live
  // on a different machine from the one this is usually developed on, and the
  // time to find out a checkpoint is missing is before the demo.
  const cfg = probe();
  for (const key of ["pipelineScript", "pipelinePy", "guiPy", "yoloeWeights", "reidCkpt"]) {
    const { path: p, ok, reason } = cfg[key];
    if (ok) logger.info(`  ${key}: ${p}`);
    else logger.warn(`  ${key}: ${p || "<unset>"} — ${reason || "NOT FOUND"}`);
  }
  logger.info(`  media roots: ${cfg.mediaRoots.join(", ")}`);

  // A relay that was killed rather than shut down leaves its producers running,
  // and they reconnect to this one. Clear them before anyone presses Start.
  const reaped = reapOrphans(logger.child({ ns: "launch" }));
  if (reaped) logger.warn(`reaped ${reaped} orphaned producer(s) from a previous relay`);
});

// A relay that exits leaves the pipeline holding a GPU and an MJPEG port with
// nothing reading either, and the next launch then fails to bind STREAM_PORT.
// The run belongs to the relay for its lifetime, so it ends with it.
for (const sig of ["SIGINT", "SIGTERM"]) {
  process.on(sig, () => {
    logger.info({ sig }, "shutting down — stopping any run");
    launcher.stop({ reason: `relay received ${sig}` });
    setTimeout(() => process.exit(0), 500).unref();
  });
}

process.on("uncaughtException", (err) => {
  logger.fatal({ err }, "uncaught exception");
  process.exit(1);
});
process.on("unhandledRejection", (reason) => logger.error({ reason }, "unhandled rejection"));
