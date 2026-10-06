#!/usr/bin/env python3
"""
Telemetry for the map and the System pane, replayed from a flight-controller
CSV while the real pipeline processes the matching video.

`telem_worker.py` is the same producer against a live MAVLink link. This is the
recorded equivalent: it emits the identical `telemetry` payload, so the map's
drone marker, its breadcrumb trail and every row in the System pane light up
without changing a line of the frontend.

    python telem_csv.py --csv fcb.csv --root mission_logs/        # follow the pipeline
    python telem_csv.py --csv fcb.csv --realtime                  # no pipeline, wall clock

## Which row is "now"

Not the wall clock. `Pipeline_yoloe_logs_robust.py` on a CPU box runs at a
fraction of real time, so a drone flown at wall-clock speed would be a minute
ahead of the frame the pipeline is actually looking at — and every casualty pin
would drop far behind a marker that had already flown past it.

So the pipeline's own progress is the clock. It writes exactly one

    [FRAME  ] FPS=12.3

line per frame it finishes, and its `frame_id` starts at the frame the capture
seeked to and increments once per such line. Counting them in mission.log gives
the frame being processed, with no change to the pipeline and nothing to keep
in sync but a file it is already writing.

That frame then indexes the CSV directly when the recording carries a `frame`
column (the FCB logger's does), which is an exact mapping rather than two
clocks that were never synchronised.

`--realtime` is the fallback for showing a flight with no pipeline attached.
"""

import argparse
import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bus import Bus                       # noqa: E402
from telem_track import TelemetryTrack    # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("telem-csv")

# Matches telem_worker.py. The map interpolates between fixes, so emitting
# faster only costs socket writes.
EMIT_HZ = 4.0

# One per frame the pipeline completes. `log_frame(f"FPS={fps:.1f}")`.
FRAME_RE = re.compile(r"^\[FRAME\s*\]\s*FPS=")

# How often to look for a run folder that has not been created yet.
POLL_S = 1.0


def newest_run(root, since=0.0):
    """
    Newest run folder under `root`, ignoring anything older than `since`.

    `since` is what stops this latching onto the PREVIOUS run. The pipeline
    needs a moment to create its folder, and for that moment the newest
    mission.log under the root is the one the last run left behind — complete,
    with its whole frame count already in it. Latching there makes the clock
    read that run's final frame and sit on it forever, so the drone parks
    mid-flight and never moves again while telemetry streams perfectly.

    A live run's mission.log is being appended to continuously, so its mtime is
    always fresh; a finished one's is not. Comparing against the moment this
    producer started therefore separates them cleanly, and also works when this
    is attached by hand to a pipeline that is already running.
    """
    try:
        entries = [os.path.join(root, d) for d in os.listdir(root)]
    except OSError:
        return None

    runs = []
    for d in entries:
        log = os.path.join(d, "mission.log")
        try:
            if os.path.isfile(log) and os.path.getmtime(log) >= since:
                runs.append((os.path.getmtime(log), d))
        except OSError:
            continue
    return max(runs)[1] if runs else None


class FrameClock:
    """
    The pipeline's current video frame, read from the run it is writing.

    Tails mission.log and counts FRAME lines. The file is opened in binary and
    decoded per line because the pipeline appends while this reads: a partial
    UTF-8 sequence at the tail must not take the clock down.
    """

    def __init__(self, root, start_frame=0, run=None, since=0.0):
        self.root = root
        self.start_frame = start_frame
        self.run = run
        self.since = since
        self._fh = None
        self._pending = b""
        self.count = 0

    def _open(self):
        if self.run is None:
            self.run = newest_run(self.root, self.since)
            if self.run is None:
                return False
            log.info("following %s", self.run)
        try:
            self._fh = open(os.path.join(self.run, "mission.log"), "rb")
        except OSError:
            self.run = None
            return False
        return True

    def frame(self):
        """Frame index the pipeline is on, or None while nothing is running."""
        if self._fh is None and not self._open():
            return None

        chunk = self._fh.read()
        if chunk:
            self._pending += chunk
            *lines, self._pending = self._pending.split(b"\n")
            for raw in lines:
                if FRAME_RE.match(raw.decode("utf-8", "replace")):
                    self.count += 1

        # The pipeline rewrites mission.log from scratch at the start of a run,
        # so a file that shrank is a NEW run in the same folder, not a glitch.
        try:
            if os.fstat(self._fh.fileno()).st_size < self._fh.tell():
                log.info("mission.log truncated — restarting the clock")
                self._fh.close()
                self._fh = None
                self._pending = b""
                self.count = 0
                return None
        except OSError:
            pass

        return self.start_frame + self.count if self.count else None


def payload(state, frame, source):
    """
    The `telemetry` wire shape, field-for-field what telem_worker.py emits.

    Fields the flight log does not carry are None rather than invented, so the
    System pane shows "—" for them: a fabricated battery percentage on a wall
    display is the kind of number someone makes a decision on.
    """
    return {
        "uavId": 0,
        "latitude": state["lat"],
        "longitude": state["lon"],
        "altitude": state.get("alt"),
        "groundspeed": state.get("speed"),
        "battery": None,
        "batteryPct": state.get("battery"),
        "current": None,
        "armed": True,
        "mode": "AUTO",
        "heading": state.get("heading"),
        "Status": "ACTIVE",
        "Last_Heartbeat": round(1.0 / EMIT_HZ, 3),
        # Wall-clock, so the console's 3 s staleness check reads this as live
        # even while video time crawls behind it on a CPU box.
        "ts": time.time(),
        # Extras the MAVLink worker has no equivalent for. The frontend shows
        # `source` in the System pane so nobody mistakes a replayed flight for
        # an aircraft that is actually up.
        "source": source,
        "frame": frame,
        "roll": state.get("roll"),
        "pitch": state.get("pitch"),
        "yaw": state.get("yaw"),
        "zoom": state.get("zoom"),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="flight-controller CSV")
    ap.add_argument("--root", default=os.environ.get("MISSION_ROOT"),
                    help="mission_logs root to follow the pipeline in")
    ap.add_argument("--run", help="a specific run folder (default: the newest)")
    ap.add_argument("--start-frame", type=int, default=0,
                    help="the pipeline's START_FRAME, so the two agree on frame 0")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="video fps, used only when the CSV has no frame column")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="shift the CSV against the video, seconds")
    ap.add_argument("--since", type=float, default=None,
                    help="ignore run folders older than this epoch time; "
                         "defaults to now, so a previous run is never mistaken "
                         "for the one starting")
    ap.add_argument("--realtime", action="store_true",
                    help="play at wall-clock speed instead of following the pipeline")
    ap.add_argument("--speed", type=float, default=1.0, help="with --realtime")
    ap.add_argument("--url", default=os.environ.get("GUI_URL", "http://127.0.0.1:7100"))
    args = ap.parse_args()

    track = TelemetryTrack(args.csv, args.offset)

    bus = Bus(role="telem-csv", url=args.url)
    bus.connect()

    source = "csv-realtime" if args.realtime else "csv-pipeline"
    bus.set_hello("worker_status", {
        "worker": "telem", "state": "running",
        "source": f"{os.path.basename(args.csv)} ({source})",
    })

    clock = None
    if not args.realtime:
        if not args.root:
            raise SystemExit("telem_csv: --root is required without --realtime")
        # Default to now, less a little slack for a pipeline that created its
        # folder in the moment between the relay spawning the two of us.
        since = args.since if args.since is not None else time.time() - 10.0
        clock = FrameClock(args.root, args.start_frame, args.run, since)
        log.info("following the pipeline's frame count under %s", args.root)
    else:
        log.info("wall-clock playback at %.2fx", args.speed)

    period = 1.0 / EMIT_HZ
    t_start = time.time()
    live = False

    try:
        while True:
            if clock is not None:
                frame = clock.frame()
                if frame is None:
                    # Nothing to be in step with yet. Emitting the first row
                    # anyway would park the drone at the start of the flight
                    # and have the System pane claim a live link before the
                    # pipeline has read a single frame.
                    if live:
                        log.info("pipeline stopped writing frames")
                        live = False
                    time.sleep(POLL_S)
                    continue
                if not live:
                    log.info("pipeline is live — frame %d", frame)
                    live = True
                state = track.at_frame(frame)
                if state is None:
                    state = track.at(frame / args.fps)
            else:
                elapsed = (time.time() - t_start) * args.speed
                if elapsed > track.duration:
                    log.info("end of the flight log")
                    break
                frame = int(elapsed * args.fps)
                state = track.at(elapsed)

            bus.emit("telemetry", payload(state, frame, source))
            time.sleep(period)
    except KeyboardInterrupt:
        log.info("stopped")
    finally:
        bus.emit("worker_status", {"worker": "telem", "state": "stopped", "source": args.csv})
        bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
