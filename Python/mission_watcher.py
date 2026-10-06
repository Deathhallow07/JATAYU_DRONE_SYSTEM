#!/usr/bin/env python3
"""
Mission watcher — turns the ReID pipeline's RUN_FOLDER into a live feed.

The pipeline (ReID-Pipeline/YOLOE/.../Pipeline_yoloe_logs_robust.py) already
writes everything the GUI needs:

    mission_logs/<YYYYmmdd_HHMMSS>/
        mission.log                  "[OK     ] GID=3 Folder ready -> ..."
        gid_3/
            metadata.txt             the identity record, appended per buffer
            representative.jpg       sharpest crop
            crop_frame_000142.jpg    every crop in the buffer

so this watcher reads those artifacts rather than asking the pipeline to
publish anything. Nothing in the pipeline changes, and a mission that ran last
week replays exactly like one running now.

mission.log is the clock for both modes. Every GID folder is announced in it by

    GID=<n> Folder ready -> <path>

which the pipeline writes at the end of save_new_gid_artifacts — i.e. after
metadata.txt and the crops are on disk. Keying GID emission off that line means
we never read a half-written record, and in replay the GIDs appear at the same
point in the log they appeared at live.

    LIVE     tail mission.log as the pipeline appends to it
    REPLAY   read a finished mission.log at a controllable rate

Usage
-----
    python mission_watcher.py                       # newest run, live
    python mission_watcher.py --replay              # newest run, replayed
    python mission_watcher.py --run <folder> --replay --speed 4
"""

import argparse
import json
import logging
import os
import re
import threading
import time
from datetime import datetime

from bus import Bus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("watcher")

DEFAULT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..",
        "ReID-Pipeline", "YOLOE", "testing", "mission_logs",
    )
)

# "[OK     ] GID=3 Folder ready -> /path/to/gid_3"
LOG_LINE_RE = re.compile(r"^\[(\w+)\s*\]\s?(.*)$")
GID_READY_RE = re.compile(r"GID=(\d+)\s+Folder ready")
FPS_RE = re.compile(r"FPS=([0-9.]+)")

# Batch window for log emission. Coalescing a burst into one socket frame keeps
# a pipeline logging a few hundred lines a second from turning into a few
# hundred websocket writes a second.
BATCH_INTERVAL = 0.10
BATCH_MAX = 400

# How long to wait for the relay before reading the source anyway.
CONNECT_WAIT = 15.0


# ---------------------------------------------------------------------------
# metadata.txt parsing
# ---------------------------------------------------------------------------

# Values are written with their unit attached ("420.800 sec"). Strip a trailing
# unit before coercing, or the field reaches the browser as a string and every
# numeric format on it silently turns into NaN.
_UNIT_RE = re.compile(r"^(-?[0-9.eE+]+)\s*(sec|s|m|ms|deg|px)$")


def _num(value):
    """Best-effort numeric coercion; metadata values are all strings on disk."""
    if isinstance(value, str):
        unit = _UNIT_RE.match(value.strip())
        if unit:
            value = unit.group(1)
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def parse_metadata(path):
    """
    Parse a GID's metadata.txt into

        {
          header: {...},                 # GID, Created, Mission Folder
          final_gps: {lat, lon, samples},
          buffers: [ {event, tid, wall_time, mission_timestamp, ...,
                      entries: [{frame, conf, sharpness, lat, lon}]} ]
        }

    The file is written by three different code paths (header once, FINAL GPS
    block rewritten in place, buffer records appended), so this parses by
    section marker rather than by line number.
    """
    if not os.path.exists(path):
        return None

    with open(path, "r", errors="replace") as f:
        lines = f.read().splitlines()

    out = {"header": {}, "final_gps": {}, "buffers": []}

    section = "header"
    buf = None
    in_entries = False

    for raw in lines:
        line = raw.rstrip()
        if not line or set(line) <= {"=", "-", " "}:
            # A rule line closes the entry table; everything else is decoration.
            if in_entries:
                in_entries = False
            continue

        if line.startswith("BUFFER "):
            m = re.match(r"BUFFER\s+(\w+)\s+TID=(-?\d+)", line)
            buf = {
                "event": (m.group(1).lower() if m else "unknown"),
                "tid": (int(m.group(2)) if m else None),
                "entries": [],
            }
            out["buffers"].append(buf)
            section = "buffer"
            in_entries = False
            continue

        if line.startswith("FINAL GPS") or "FINAL GPS (" in line:
            section = "final_gps"
            continue

        if line.startswith("Frame") and "Conf" in line and "Sharpness" in line:
            in_entries = True
            continue

        if in_entries and buf is not None:
            parts = line.split()
            if len(parts) >= 5:
                buf["entries"].append({
                    "frame": _num(parts[0]),
                    "conf": _num(parts[1]),
                    "sharpness": _num(parts[2]),
                    "lat": _num(parts[3]),
                    "lon": _num(parts[4]),
                })
            continue

        if ":" not in line:
            continue

        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()

        if key.startswith("Final GPS"):
            target = out["final_gps"]
            key = key.replace("Final GPS", "").strip() or "value"
        elif section == "buffer" and buf is not None:
            target = buf
        else:
            target = out["header"]

        target[key.lower().replace(" ", "_").replace(".", "")] = _num(value)

    return out


def _latest(seq, *keys, default=None):
    """Last non-None value of any of `keys` across a list of dicts."""
    for item in reversed(seq):
        for k in keys:
            if item.get(k) is not None:
                return item[k]
    return default


def _reduce(seq, key, fn, default=None):
    """
    Apply `fn` (min/max) across every buffer's value for `key`.

    Not the same as _latest. The pipeline computes First/Last Frame and the
    Best Confidence/Sharpness over the CUMULATIVE gallery, so the newest
    buffer's value is usually already the answer — but the gallery is capped
    (the M/N/L cap prunes old entries), so an early frame can drop out of the
    cumulative view and the "first frame" of a GID would appear to move
    forwards in time as the mission runs. Reducing over every record that was
    ever written keeps the span honest.
    """
    values = [b[key] for b in seq if isinstance(b.get(key), (int, float))]
    return fn(values) if values else default


def build_gid_payload(run_folder, gid):
    """
    Collapse gid_<n>/ into the flat record the GID pane renders.

    Crops are sent as URLs relative to the run folder, not as bytes — the relay
    serves the run folder over HTTP, so a GID with 200 crops costs 200 lazy
    <img> requests instead of a multi-megabyte socket payload.
    """
    folder = os.path.join(run_folder, f"gid_{gid}")
    meta = parse_metadata(os.path.join(folder, "metadata.txt"))
    if meta is None:
        return None

    buffers = meta["buffers"]
    fg = meta["final_gps"]

    try:
        crops = sorted(
            f for f in os.listdir(folder)
            if f.startswith("crop_frame_") and f.endswith(".jpg")
        )
    except OSError:
        crops = []

    rel = f"gid_{gid}"
    has_rep = os.path.exists(os.path.join(folder, "representative.jpg"))

    return {
        "gid": int(gid),
        "created": meta["header"].get("created"),
        "event": (buffers[-1]["event"] if buffers else "new"),
        "updates": len(buffers),
        "tid": _latest(buffers, "tid"),

        "lat": fg.get("latitude"),
        "lon": fg.get("longitude"),
        "gpsSamples": fg.get("samples"),

        # Cumulative and monotonic — the newest record is the current size.
        "gallerySize": _latest(buffers, "cumulative_gallery", default=0),
        # Spans and bests, reduced over every buffer ever written (see _reduce).
        "bestConf": _reduce(buffers, "best_confidence", max),
        "bestSharpness": _reduce(buffers, "best_sharpness", max),
        "firstFrame": _reduce(buffers, "first_frame", min),
        "lastFrame": _reduce(buffers, "last_frame", max),
        "missionTimestamp": _latest(buffers, "mission_timestamp"),
        "rosTime": _latest(buffers, "ros_assignment_time"),

        # Cache-bust on update count: a GID's representative.jpg is overwritten
        # in place when a better crop arrives, and the browser would otherwise
        # keep showing the first one for the rest of the mission.
        "representative": f"{rel}/representative.jpg?v={len(buffers)}" if has_rep else None,
        "crops": [f"{rel}/{c}" for c in crops],

        "buffers": [
            {k: v for k, v in b.items() if k != "entries"} | {"entryCount": len(b["entries"])}
            for b in buffers
        ],
        "track": [e for b in buffers for e in b["entries"]],
    }


# ---------------------------------------------------------------------------
# Run folder discovery
# ---------------------------------------------------------------------------

def list_runs(root, since=0.0):
    """
    Run folders newest first. Names are %Y%m%d_%H%M%S, so name sort == time sort.

    `since` drops anything whose mission.log was last written before that
    moment. It is what stops a launcher-started watcher latching onto the
    PREVIOUS run: the pipeline takes a moment to create its folder, and for
    that moment the newest folder under the root is the last run's — complete,
    with all of its GIDs in it. Attaching there means the console fills with a
    finished run's casualties while the live one is never read at all.

    A live run's mission.log is appended to continuously so its mtime stays
    fresh; a finished one's does not. Left at 0 every run is listed, which is
    what `--list` and the run picker want.
    """
    if not os.path.isdir(root):
        return []
    runs = []
    for name in sorted(os.listdir(root), reverse=True):
        path = os.path.join(root, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "mission.log")):
            if since and os.path.getmtime(os.path.join(path, "mission.log")) < since:
                continue
            gids = len([d for d in os.listdir(path) if d.startswith("gid_")])
            runs.append({
                "name": name,
                "path": path,
                "gids": gids,
                "mtime": os.path.getmtime(path),
                "size": os.path.getsize(os.path.join(path, "mission.log")),
            })
    return runs


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------

class MissionWatcher:
    def __init__(self, bus, root, run=None, replay=False, speed=1.0, rate=60.0,
                 keep_frame_lines=False, since=0.0):
        self.bus = bus
        self.root = root
        self.run_folder = run
        self.replay = replay
        self.keep_frame_lines = keep_frame_lines
        # Only attach to a run folder at least this fresh. See list_runs().
        self.since = since

        # Replay pacing. mission.log carries no per-line timestamps, so replay
        # runs at a fixed line rate rather than reconstructing wall time. GIDs
        # still land at the right point in the log because they are keyed off
        # the "Folder ready" line (see module docstring).
        self.rate = rate
        self.speed = speed
        self.paused = False

        self._stop = threading.Event()
        self._pending = []
        self._pending_lock = threading.Lock()
        self._seen_gid_events = {}      # gid -> times the ready line was seen
        self._last_gid_payload = {}     # gid -> the payload as last emitted

        self.stats = {
            "frames": 0, "gids": 0, "logLines": 0,
            "fpsLast": None, "fpsMin": None, "fpsMax": None,
            "levels": {}, "startedAt": time.time(),
        }

    # -- commands from the browser -----------------------------------------

    def attach_commands(self):
        def replay_control(payload=None):
            payload = payload or {}
            action = payload.get("action")
            if action == "pause":
                self.paused = True
            elif action == "play":
                self.paused = False
            elif action == "restart":
                self.restart()
            if "speed" in payload:
                try:
                    self.speed = max(0.1, min(50.0, float(payload["speed"])))
                except (TypeError, ValueError):
                    pass
            log.info("replay_control %s -> paused=%s speed=%.1fx",
                     action, self.paused, self.speed)

        self.bus.on("replay_control", replay_control)

        def select_run(payload=None):
            payload = payload or {}
            path = payload.get("path")
            if path and os.path.isdir(path):
                self.run_folder = path
                self.replay = bool(payload.get("replay", True))
                self.restart()

        self.bus.on("select_run", select_run)

    def restart(self):
        self._stop.set()

    def resync(self):
        """
        Republish every GID after a reconnect.

        Log lines are a stream and losing some while the socket was down is
        survivable — the console shows a gap. GIDs are not: each is announced
        exactly once, so one emitted into a dead socket is gone for the rest of
        the run and a located casualty never appears on screen.
        """
        if not self._last_gid_payload:
            return
        for gid in sorted(self._last_gid_payload):
            self.bus.emit("gid", self._last_gid_payload[gid])
        log.info("resynced %d GIDs after reconnect", len(self._last_gid_payload))

    # -- emission ----------------------------------------------------------

    def _queue_log(self, level, msg):
        # FRAME is one line per processed frame ("FPS=21.3") and nothing else.
        # At 20-30 fps that is ~70% of everything the pipeline writes, it is
        # muted by default in the log pane, and the number it carries is
        # already in the status bar's FPS readout — which comes from `stats`,
        # counted below, not from these lines.
        #
        # Dropping it here rather than in the browser is the point: otherwise
        # every one of them still crosses the socket and still lands in the
        # browser's 5000-line array, so every real line arrives behind a queue
        # of frame counters. That is what makes the mission log appear to stop
        # updating on a fast run.
        if level == "FRAME" and not self.keep_frame_lines:
            return
        with self._pending_lock:
            self._pending.append({
                "ts": time.time(),
                "level": level,
                "msg": msg,
            })

    def _flush_loop(self):
        while not self._stop.is_set():
            time.sleep(BATCH_INTERVAL)
            with self._pending_lock:
                if not self._pending:
                    continue
                batch, self._pending = self._pending[:BATCH_MAX], self._pending[BATCH_MAX:]
            self.bus.emit("logs", batch)

    def _stats_loop(self):
        while not self._stop.is_set():
            time.sleep(1.0)
            self.stats["elapsed"] = time.time() - self.stats["startedAt"]
            self.stats["paused"] = self.paused
            self.stats["speed"] = self.speed
            self.bus.emit("stats", dict(self.stats))

    # -- line handling -----------------------------------------------------

    def handle_line(self, line):
        line = line.rstrip("\n")
        if not line.strip():
            return

        m = LOG_LINE_RE.match(line)
        level, msg = (m.group(1), m.group(2)) if m else ("RAW", line)

        self.stats["logLines"] += 1
        self.stats["levels"][level] = self.stats["levels"].get(level, 0) + 1

        fps = FPS_RE.search(msg)
        if fps:
            v = float(fps.group(1))
            self.stats["frames"] += 1
            self.stats["fpsLast"] = v
            self.stats["fpsMin"] = v if self.stats["fpsMin"] is None else min(self.stats["fpsMin"], v)
            self.stats["fpsMax"] = v if self.stats["fpsMax"] is None else max(self.stats["fpsMax"], v)

        self._queue_log(level, msg)

        ready = GID_READY_RE.search(msg)
        if ready:
            self._emit_gid(int(ready.group(1)))

    def _emit_gid(self, gid):
        payload = build_gid_payload(self.run_folder, gid)
        if payload is None:
            log.warning("GID %s announced but metadata.txt is missing", gid)
            return

        seen = self._seen_gid_events.get(gid, 0) + 1
        self._seen_gid_events[gid] = seen

        if self.replay:
            # The folder on disk is the FINAL state. Replaying it as-is would
            # show every GID fully-formed the moment it first appears, which
            # is exactly the history the demo is meant to show unfolding. Trim
            # the record to the buffers that had been written by this point.
            payload = self._truncate_to(payload, seen)

        if seen == 1:
            self.stats["gids"] += 1

        self._last_gid_payload[gid] = payload
        self.bus.emit("gid", payload)
        log.info("GID %s emitted (update %s, event=%s)", gid, seen, payload["event"])

    @staticmethod
    def _truncate_to(payload, n_buffers):
        buffers = payload["buffers"][:n_buffers]
        if not buffers:
            return payload
        last = buffers[-1]
        return payload | {
            "buffers": buffers,
            "updates": len(buffers),
            "event": last.get("event", payload["event"]),
            "tid": last.get("tid", payload["tid"]),
            "gallerySize": last.get("cumulative_gallery", payload["gallerySize"]),
            "bestConf": _reduce(buffers, "best_confidence", max),
            "bestSharpness": _reduce(buffers, "best_sharpness", max),
            "firstFrame": _reduce(buffers, "first_frame", min),
            "lastFrame": _reduce(buffers, "last_frame", max),
            "missionTimestamp": last.get("mission_timestamp"),
            # Keep the cache-buster in step with the truncated update count, so
            # a replayed GID's image URL is the one the live run would have had.
            "representative": (
                re.sub(r"\?v=\d+$", f"?v={len(buffers)}", payload["representative"])
                if payload["representative"] else None
            ),
            # Crops carry their source frame in the filename, so the set visible
            # at this point is the set whose frame <= the newest buffer's last.
            "crops": [
                c for c in payload["crops"]
                if _crop_frame(c) is None
                or last.get("last_frame") is None
                or _crop_frame(c) <= last["last_frame"]
            ],
        }

    # -- sources -----------------------------------------------------------

    def _tail(self):
        """Live: follow mission.log, including the lines already in it."""
        path = os.path.join(self.run_folder, "mission.log")
        log.info("tailing %s", path)

        with open(path, "r", errors="replace") as f:
            while not self._stop.is_set():
                line = f.readline()
                if line:
                    self.handle_line(line)
                    continue
                # Truncation/rotation: the pipeline opens mission.log with "w"
                # at startup, so a restart into the same folder rewinds it.
                if os.path.getsize(path) < f.tell():
                    log.info("mission.log truncated — rewinding")
                    f.seek(0)
                else:
                    time.sleep(0.15)

    def _replay(self):
        path = os.path.join(self.run_folder, "mission.log")
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()

        log.info("replaying %s (%d lines) at %.1f lines/s", path, len(lines), self.rate)

        for line in lines:
            if self._stop.is_set():
                return
            while self.paused and not self._stop.is_set():
                time.sleep(0.05)
            self.handle_line(line)
            time.sleep(1.0 / max(1e-3, self.rate * self.speed))

        log.info("replay complete — %d GIDs", self.stats["gids"])
        self._queue_log("INFO", "=== REPLAY COMPLETE ===")
        # Hold the process open so the panes keep their data and the operator
        # can restart the replay from the UI.
        while not self._stop.is_set():
            time.sleep(0.2)

    # -- main --------------------------------------------------------------

    def run_once(self):
        self._stop.clear()

        if not self.run_folder:
            runs = list_runs(self.root, self.since)
            if not runs:
                log.warning("no run folder under %s yet — waiting%s", self.root,
                            " for one newer than this watcher" if self.since else "")
                while not runs and not self._stop.is_set():
                    time.sleep(1.0)
                    runs = list_runs(self.root, self.since)
                if not runs:
                    return
            self.run_folder = runs[0]["path"]

        self.stats = {
            "frames": 0, "gids": 0, "logLines": 0,
            "fpsLast": None, "fpsMin": None, "fpsMax": None,
            "levels": {}, "startedAt": time.time(),
        }
        self._seen_gid_events.clear()
        self._last_gid_payload.clear()

        # set_hello, not emit: connect() is asynchronous and a run started
        # before the socket is up would otherwise never announce itself.
        self.bus.set_hello("mission_start", {
            "runFolder": self.run_folder,
            "name": os.path.basename(self.run_folder),
            "startedAt": datetime.now().isoformat(),
            "replay": self.replay,
            "speed": self.speed,
            "runs": [{k: r[k] for k in ("name", "path", "gids")} for r in list_runs(self.root)],  # all of them, for the picker
        })

        # Give the relay a moment to come up before reading the source. The
        # launcher starts both at once, and a watcher that wins that race emits
        # its opening GIDs into a socket that is not connected yet — emit() is a
        # no-op then, so they are lost. Bounded, because the pipeline's
        # artifacts are the record of truth and the watcher must never block on
        # the GUI being present.
        if not self.bus.wait_connected(timeout=CONNECT_WAIT):
            log.warning("relay not up after %.0fs — starting anyway; "
                        "GIDs will be resynced when it connects", CONNECT_WAIT)

        threading.Thread(target=self._flush_loop, daemon=True).start()
        threading.Thread(target=self._stats_loop, daemon=True).start()

        try:
            (self._replay if self.replay else self._tail)()
        except FileNotFoundError:
            log.error("mission.log vanished under %s", self.run_folder)
        except KeyboardInterrupt:
            raise
        finally:
            self._stop.set()

    def run_forever(self):
        while True:
            self.run_once()
            time.sleep(0.5)      # a restart command lands here


def _crop_frame(name):
    m = re.search(r"crop_frame_(\d+)\.jpg", name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="mission_logs directory")
    ap.add_argument("--run", default=None, help="specific run folder (default: newest)")
    ap.add_argument("--replay", action="store_true", help="replay a finished run instead of tailing")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    ap.add_argument("--rate", type=float, default=60.0, help="replay base rate, log lines/sec")
    ap.add_argument("--url", default=os.environ.get("GUI_URL", "http://127.0.0.1:7100"))
    ap.add_argument("--since", type=float, default=0.0,
                    help="only attach to a run folder written at or after this "
                         "epoch time, so a previous run is never mistaken for "
                         "the one starting")
    ap.add_argument("--keep-frame-lines", action="store_true",
                    help="forward the pipeline's per-frame FPS lines too; they "
                         "are ~70%% of the log and the status bar already "
                         "carries the number")
    ap.add_argument("--list", action="store_true", help="list run folders and exit")
    args = ap.parse_args()

    if args.list:
        print(json.dumps(list_runs(args.root), indent=2))
        return

    bus = Bus("watcher", args.url)
    watcher = MissionWatcher(bus, args.root, args.run, args.replay, args.speed, args.rate,
                             keep_frame_lines=args.keep_frame_lines, since=args.since)
    watcher.attach_commands()
    bus.on_connect(watcher.resync)
    bus.connect()

    try:
        watcher.run_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
