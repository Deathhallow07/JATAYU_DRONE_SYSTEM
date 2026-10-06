"""
The flight path, read from a flight-controller CSV and sampled by video time.

Split out of mission_run.py so the two things that need it can share one copy:

  * mission_run.py  — the GUI's own detector, which reads the CSV to place
    detections on the ground and to fly the drone marker.
  * telem_csv.py    — the CSV telemetry producer, which republishes the same
    rows as `telemetry` while the REAL pipeline
    (Pipeline_yoloe_logs_robust.py) does the detecting.

Two copies of the column-alias table would drift, and a map that disagrees with
where the pipeline thinks it is looking is worse than no map.
"""

import logging
import math
import os

log = logging.getLogger("telem_track")


def haversine_m(lat1, lon1, lat2, lon2):
    r1, r2 = math.radians(lat1), math.radians(lat2)
    dlat = r2 - r1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(r1) * math.cos(r2) * math.sin(dlon / 2) ** 2
    return 6371000.0 * 2 * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    r1, r2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(r2)
    x = math.cos(r1) * math.sin(r2) - math.sin(r1) * math.cos(r2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0



class TelemetryTrack:
    """
    The flight path, read from the CSV and sampled by video timestamp.

    Column names vary between the recordings in this repo, so the usual aliases
    are all accepted. Time is taken relative to the first row: a flight log
    normally starts at a Unix timestamp or a ROS clock, and the video starts at
    zero, so the two are aligned at their starts unless --telem-offset says
    otherwise.
    """

    # Ordered by preference, not alphabetically: a recording from the FCB
    # logger carries BOTH yaw_deg and heading_deg, and heading is the one that
    # points where the airframe is going.
    T_KEYS = ("t_mono_s", "timestamp_sec", "timestamp", "t", "time", "time_s")
    LAT_KEYS = ("lat_deg", "drone_lat", "lat", "latitude")
    LON_KEYS = ("lon_deg", "drone_lon", "lon", "longitude")
    ALT_KEYS = ("alt_rel_m", "drone_altitude_agl", "rel_alt", "altitude", "alt", "alt_m")
    HDG_KEYS = ("heading_deg", "heading", "yaw_deg", "yaw", "hdg")
    SPD_KEYS = ("groundspeed_ms", "groundspeed", "gs", "speed")
    ROLL_KEYS = ("roll_deg", "roll")
    PITCH_KEYS = ("pitch_deg", "pitch")
    YAW_KEYS = ("yaw_deg", "yaw")
    ZOOM_KEYS = ("zoom_ratio", "zoom")
    BAT_KEYS = ("battery_pct", "battery_remaining", "batt_pct", "battery")
    FRAME_KEYS = ("frame", "frame_idx", "frame_id")

    def __init__(self, path, offset=0.0):
        import csv

        self.rows = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []

            def pick(keys):
                lowered = {c.lower().strip(): c for c in fields}
                return next((lowered[k] for k in keys if k in lowered), None)

            tk, lak, lok = pick(self.T_KEYS), pick(self.LAT_KEYS), pick(self.LON_KEYS)
            altk, hdgk, batk = pick(self.ALT_KEYS), pick(self.HDG_KEYS), pick(self.BAT_KEYS)
            spdk, frk = pick(self.SPD_KEYS), pick(self.FRAME_KEYS)
            rollk, pitchk = pick(self.ROLL_KEYS), pick(self.PITCH_KEYS)
            yawk, zoomk = pick(self.YAW_KEYS), pick(self.ZOOM_KEYS)
            if not (tk and lak and lok):
                raise ValueError(
                    f"{path}: need time + lat + lon columns; found {fields}"
                )

            for r in reader:
                try:
                    row = {
                        "t": float(r[tk]),
                        "lat": float(r[lak]),
                        "lon": float(r[lok]),
                    }
                except (TypeError, ValueError, KeyError):
                    continue
                # A row with no fix is a row with no position, not a row at the
                # origin — 0,0 would otherwise teleport the drone to the ocean.
                if not (math.isfinite(row["lat"]) and math.isfinite(row["lon"])):
                    continue
                if row["lat"] == 0.0 and row["lon"] == 0.0:
                    continue
                for key, col in (("alt", altk), ("heading", hdgk),
                                 ("speed", spdk), ("battery", batk),
                                 ("roll", rollk), ("pitch", pitchk),
                                 ("yaw", yawk), ("zoom", zoomk)):
                    if col:
                        try:
                            row[key] = float(r[col])
                        except (TypeError, ValueError, KeyError):
                            pass
                if frk:
                    try:
                        row["frame"] = int(float(r[frk]))
                    except (TypeError, ValueError, KeyError):
                        pass
                self.rows.append(row)

        if not self.rows:
            raise ValueError(f"{path}: no usable rows")

        self.rows.sort(key=lambda r: r["t"])
        self.t0 = self.rows[0]["t"] - offset
        self.duration = self.rows[-1]["t"] - self.rows[0]["t"]
        self._cursor = 0

        # A recording from the FCB logger carries the video frame number each
        # fix belongs to. That is an exact mapping and beats aligning two clocks
        # that were never synchronised — so prefer it whenever it is present.
        self.by_frame = {}
        if offset == 0.0:
            for r in self.rows:
                if "frame" in r:
                    self.by_frame.setdefault(r["frame"], r)

        log.info("telemetry: %d fixes spanning %.1f s%s (%s)",
                 len(self.rows), self.duration,
                 f", frame-indexed ({len(self.by_frame)} frames)" if self.by_frame else "",
                 os.path.basename(path))

    def at_frame(self, frame_idx):
        """Exact fix for a video frame, when the CSV names frames. Else None."""
        r = self.by_frame.get(frame_idx)
        if r is None:
            return None
        nxt = self.by_frame.get(frame_idx + 1, r)
        return self._state(r, nxt, 0.0)

    def at(self, video_t):
        """
        Interpolated state at a video timestamp.

        Interpolated, not nearest-sample: a 1 Hz log against a 30 fps video
        would otherwise make the drone jump once a second instead of flying.
        The cursor only ever moves forward, so this stays O(1) per frame over a
        long flight rather than rescanning the log every time.
        """
        t = self.t0 + video_t
        rows = self.rows

        if t <= rows[0]["t"]:
            return self._state(rows[0], rows[min(1, len(rows) - 1)], 0.0)
        if t >= rows[-1]["t"]:
            return self._state(rows[-1], rows[-1], 0.0)

        i = self._cursor
        if i >= len(rows) - 1 or rows[i]["t"] > t:
            i = 0  # a seek backwards (looped video); rescan from the start
        while i < len(rows) - 2 and rows[i + 1]["t"] <= t:
            i += 1
        self._cursor = i

        a, b = rows[i], rows[i + 1]
        span = b["t"] - a["t"]
        f = 0.0 if span <= 0 else (t - a["t"]) / span
        return self._state(a, b, f)

    @staticmethod
    def _state(a, b, f):
        lat = a["lat"] + (b["lat"] - a["lat"]) * f
        lon = a["lon"] + (b["lon"] - a["lon"]) * f

        # Heading and speed from the two bracketing fixes when the log does not
        # carry them, which most of these CSVs do not.
        # Prefer what the flight controller logged; derive only what is missing.
        heading = a.get("heading")
        if heading is None:
            heading = (bearing_deg(a["lat"], a["lon"], b["lat"], b["lon"])
                       if (a["lat"], a["lon"]) != (b["lat"], b["lon"]) else 0.0)

        speed = a.get("speed")
        if speed is None:
            dt = b["t"] - a["t"]
            speed = (haversine_m(a["lat"], a["lon"], b["lat"], b["lon"]) / dt) if dt > 0 else 0.0

        alt = a.get("alt")
        if alt is not None and b.get("alt") is not None:
            alt = alt + (b["alt"] - alt) * f

        return {
            "lat": lat, "lon": lon, "alt": alt,
            "heading": heading, "speed": speed,
            "battery": a.get("battery"),
            # Attitude is NOT interpolated: it is only used to project a
            # detection seen in one specific frame, and the nearest logged
            # attitude is the one that frame was actually taken at.
            "roll": a.get("roll"), "pitch": a.get("pitch"), "yaw": a.get("yaw"),
            "zoom": a.get("zoom"),
        }
