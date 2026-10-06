#!/usr/bin/env python3
"""
MAVLink telemetry producer for the console's map and system panel.

The rest of this app observes the pipeline's artifacts; this is the one producer
that talks to the aircraft. It pumps HEARTBEAT / GLOBAL_POSITION_INT / VFR_HUD /
SYS_STATUS off a MAVLink endpoint and republishes them on the relay as
`telemetry`, at a rate the browser can actually paint.

The payload deliberately mirrors the existing GCS's UAVs/Telem.py field-for-field
(latitude, longitude, altitude, groundspeed, battery, armed, mode, heading,
Status, Last_Heartbeat) so the two consoles' system panels stay interchangeable,
plus `batteryPct` — a medic reads a percentage, not a pack voltage.

Environment (same names as the existing GCS's Python/.env, read from Python/.env):

    ENABLE_UAV0=true        connect this vehicle at all
    UAV0_PORT=14550         udp port on 127.0.0.1; UAVn defaults to 14550+n*10
    UAV0_ENDPOINT=...       full pymavlink connection string, wins over the port

    python telem_worker.py              # UAV0 from .env
    python telem_worker.py --uav 1
    python telem_worker.py --endpoint udpin:0.0.0.0:14550

If pymavlink is not installed or the vehicle never heartbeats, this exits
quietly rather than noisily: the console falls back to the mission log's GPS
track and shows NO TELEM in the health panel, which is the honest display for
a replay with no aircraft attached.
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bus import Bus  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("telem")

# Emit rate. The map interpolates between fixes, so pushing faster than this
# only costs socket writes — the existing GCS's bridge settled on the same 4 Hz.
EMIT_HZ = 4.0

# Consecutive missed heartbeats before we tear the link down and reconnect.
HB_MISSES_MAX = 3

MAV_STATE = {
    0: "UNINIT", 1: "BOOT", 2: "CALIBRATING", 3: "STANDBY", 4: "ACTIVE",
    5: "CRITICAL", 6: "EMERGENCY", 7: "POWEROFF", 8: "FLIGHT_TERMINATION",
}


def _env_flag(key, default="false"):
    return os.environ.get(key, default).strip().lower() in ("true", "1", "yes")


def _load_dotenv():
    """Read Python/.env without adding a python-dotenv dependency.

    Existing environment wins, so `UAV0_PORT=14560 python telem_worker.py`
    behaves the way anyone would expect from a shell override.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


def endpoint_for(uav_id):
    """Connection string for a vehicle, following the existing .env convention."""
    explicit = os.environ.get(f"UAV{uav_id}_ENDPOINT")
    if explicit:
        return explicit
    port = os.environ.get(f"UAV{uav_id}_PORT", str(14550 + uav_id * 10))
    return f"127.0.0.1:{port}"


class TelemetryWorker:
    def __init__(self, bus, uav_id, endpoint):
        self.bus = bus
        self.uav_id = uav_id
        self.endpoint = endpoint
        self.conn = None
        self.hb_misses = 0
        self.last_hb = time.time()
        self.hb_gap = 0.0

    # ── link ────────────────────────────────────────────────────────────────
    def connect(self):
        from pymavlink import mavutil

        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

        log.info("connecting to %s", self.endpoint)
        self.status("loading")
        try:
            conn = mavutil.mavlink_connection(self.endpoint)
            if conn.wait_heartbeat(timeout=10) is None:
                conn.close()
                raise TimeoutError("no heartbeat within 10 s")
        except Exception as e:
            log.warning("connect failed: %s", e)
            self.status("error", str(e))
            return False

        self.conn = conn
        self.hb_misses = 0
        self.last_hb = time.time()
        log.info("connected — sysid %s", conn.target_system)
        self.status("running")
        return True

    def status(self, state, error=None):
        payload = {"worker": "telem", "state": state, "source": self.endpoint}
        if error:
            payload["error"] = error
        self.bus.emit("worker_status", payload)

    # ── pump ────────────────────────────────────────────────────────────────
    def sample(self):
        """One telemetry dict, or None while the streams are still coming up."""
        c = self.conn

        hb = c.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
        if hb is None:
            self.hb_misses += 1
            if self.hb_misses >= HB_MISSES_MAX:
                log.warning("heartbeat lost — reconnecting")
                self.conn = None
            return None

        # Ignore heartbeats from companion computers / GCS on the same link;
        # only the autopilot (component 1) describes the airframe.
        if hb.get_srcComponent() != 1:
            return None

        self.hb_misses = 0
        now = time.time()
        self.hb_gap = now - self.last_hb
        self.last_hb = now

        msgs = c.messages
        gpint = msgs.get("GLOBAL_POSITION_INT")
        vfr = msgs.get("VFR_HUD")
        sysstat = msgs.get("SYS_STATUS")

        # EKF still aligning: a heartbeat but no position yet. Not an error, and
        # emitting lat=0/lon=0 would fling the map into the Gulf of Guinea.
        if gpint is None:
            return None

        # battery_remaining is -1 when the firmware has no capacity configured.
        pct = getattr(sysstat, "battery_remaining", None) if sysstat else None
        if pct is not None and pct < 0:
            pct = None

        return {
            "uavId": self.uav_id,
            "latitude": gpint.lat / 1e7,
            "longitude": gpint.lon / 1e7,
            "altitude": gpint.relative_alt / 1000.0,
            "groundspeed": vfr.groundspeed if vfr else None,
            "battery": sysstat.voltage_battery / 1000.0 if sysstat else None,
            "batteryPct": pct,
            "current": (sysstat.current_battery / 100.0
                        if sysstat and sysstat.current_battery >= 0 else None),
            "armed": bool(hb.base_mode & 0b10000000),
            "mode": c.flightmode,
            "heading": vfr.heading if vfr else None,
            "Status": MAV_STATE.get(hb.system_status, "UNKNOWN"),
            "Last_Heartbeat": round(self.hb_gap, 3),
            "ts": now,
        }

    def run(self):
        period = 1.0 / EMIT_HZ
        while True:
            if self.conn is None:
                if not self.connect():
                    time.sleep(5)
                continue
            try:
                sample = self.sample()
            except Exception:
                log.exception("telemetry read failed — reconnecting")
                self.conn = None
                self.status("error", "read failed")
                time.sleep(2)
                continue

            if sample is not None:
                self.bus.emit("telemetry", sample)
            time.sleep(period)


def main():
    _load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uav", type=int, default=0, help="vehicle index (default 0)")
    ap.add_argument("--endpoint", help="pymavlink connection string; overrides .env")
    args = ap.parse_args()

    if args.endpoint is None and not _env_flag(f"ENABLE_UAV{args.uav}", "true"):
        log.info("ENABLE_UAV%d is false — nothing to do", args.uav)
        return 0

    try:
        import pymavlink  # noqa: F401
    except ImportError:
        log.error("pymavlink is not installed — pip install pymavlink")
        log.error("the console will fall back to the mission log's GPS track")
        return 1

    endpoint = args.endpoint or endpoint_for(args.uav)

    bus = Bus(role=f"telem-uav{args.uav}")
    bus.connect()

    worker = TelemetryWorker(bus, args.uav, endpoint)
    try:
        worker.run()
    except KeyboardInterrupt:
        log.info("stopped")
        worker.status("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
