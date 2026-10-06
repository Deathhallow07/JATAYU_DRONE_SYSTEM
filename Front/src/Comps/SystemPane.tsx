import Pane from "./Pane";
import { useMission } from "../context/MissionContext";
import { theme } from "../theme";

const fmt = (v: number | null | undefined, digits = 1, unit = "") =>
  v == null || !Number.isFinite(v) ? "—" : `${v.toFixed(digits)}${unit}`;

const fmtCoord = (v: number | null | undefined) =>
  v == null || !Number.isFinite(v) ? "—" : v.toFixed(6);

/** ArduPilot mode colouring, carried over from the existing GCS. */
const modeColor = (mode: string | null | undefined) => {
  switch ((mode ?? "").toUpperCase()) {
    case "AUTO":
    case "GUIDED":
      return theme.success;
    case "RTL":
    case "LAND":
      return theme.warning;
    case "STABILIZE":
    case "ALT_HOLD":
    case "LOITER":
      return theme.accent;
    default:
      return theme.textDim;
  }
};

// Below this the medic should be told, not left to read a number.
const BATTERY_LOW_PCT = 25;

const batteryColor = (pct: number | null) => {
  if (pct == null) return theme.textDim;
  if (pct <= 15) return theme.danger;
  if (pct <= BATTERY_LOW_PCT) return theme.warning;
  return theme.success;
};

function Row({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="sys-row">
      <span className="sys-k">{label}</span>
      <span className="sys-v" style={color ? { color, fontWeight: 700 } : undefined}>
        {value}
      </span>
    </div>
  );
}

/**
 * Aircraft state, for the person deciding whether the drone can keep flying.
 *
 * Every field reads "—" rather than a stale last-known value when the link is
 * down: a battery percentage frozen at 80% from four minutes ago is worse than
 * no number at all.
 */
export default function SystemPane({ area }: { area: string }) {
  const { telemetry, telemetryLive, dronePos, droneFromLog, stats, gids, mission } = useMission();

  const t = telemetryLive ? telemetry : null;
  const pct = t?.batteryPct ?? null;

  // A replayed flight log and an aircraft that is actually airborne must not
  // read the same on a wall display. telem_csv.py stamps every sample with
  // where it came from; telem_worker.py (a real MAVLink link) sends nothing,
  // so an absent `source` is the live case.
  const replay = t?.source?.startsWith("csv") ?? false;
  const sourceLabel = !telemetryLive ? "no telemetry"
    : t?.source === "csv-pipeline" ? "flight log · in step with the pipeline"
      : t?.source === "csv-realtime" ? "flight log · wall clock"
        : `UAV ${telemetry?.uavId ?? 0}`;

  return (
    <Pane
      area={area}
      title="System"
      subtitle={sourceLabel}
      status={{
        color: !telemetryLive ? theme.danger : replay ? theme.accent : theme.success,
        label: !telemetryLive ? "No telemetry"
          : replay ? "Replaying a recorded flight log" : "Telemetry live",
      }}
    >
      <div className="sys">
        {!telemetryLive && (
          <div className="sys-banner">
            NO TELEM
            <span>
              {droneFromLog && dronePos
                ? "Drone position is the newest GPS fix in the mission log."
                : "Start a run with a telemetry CSV, or connect the aircraft."}
            </span>
          </div>
        )}

        {replay && (
          <div className="sys-banner replay">
            FLIGHT LOG
            <span>
              Recorded telemetry, not a live aircraft. Battery and mode are not
              in the log and read &quot;—&quot;.
            </span>
          </div>
        )}

        <div className="sys-group">UAV position</div>
        <Row label="Latitude" value={fmtCoord(t ? t.latitude : dronePos?.[1])} />
        <Row label="Longitude" value={fmtCoord(t ? t.longitude : dronePos?.[0])} />
        <Row label="Altitude" value={fmt(t?.altitude, 1, " m")} />
        <Row label="Heading" value={fmt(t?.heading, 0, "°")} />
        <Row label="Ground speed" value={fmt(t?.groundspeed, 1, " m/s")} />
        {replay && (
          <>
            {/* The attitude every casualty pin is projected through, and the
                frame it came from — the one readout that says whether the map
                and the pipeline are looking at the same moment. */}
            <Row label="Roll / Pitch" value={`${fmt(t?.roll, 1, "°")} / ${fmt(t?.pitch, 1, "°")}`} />
            <Row label="Yaw" value={fmt(t?.yaw, 1, "°")} />
            <Row label="Zoom" value={t?.zoom == null ? "—" : `${t.zoom.toFixed(1)}x`} />
            <Row label="Video frame" value={t?.frame == null ? "—" : String(t.frame)} />
          </>
        )}

        <div className="sys-group">Health</div>
        <Row
          label="Battery"
          value={pct != null ? `${pct.toFixed(0)}%` : fmt(t?.battery, 1, " V")}
          color={batteryColor(pct)}
        />
        {pct != null && t?.battery != null && (
          <div className="sys-bar" aria-hidden>
            <span style={{ width: `${Math.max(0, Math.min(100, pct))}%`, background: batteryColor(pct) }} />
          </div>
        )}
        <Row label="Voltage" value={fmt(t?.battery, 2, " V")} />
        <Row label="Current" value={fmt(t?.current, 1, " A")} />
        <Row
          label="Armed"
          value={t == null ? "—" : t.armed ? "ARMED" : "DISARMED"}
          color={t == null ? undefined : t.armed ? theme.danger : theme.textDim}
        />
        <Row label="Mode" value={t?.mode || "—"} color={modeColor(t?.mode)} />
        <Row label="State" value={t?.Status ?? "—"} />
        <Row
          label="Link"
          value={t == null ? "NO TELEM" : `${(t.Last_Heartbeat * 1000).toFixed(0)} ms`}
          color={t == null ? theme.danger : theme.success}
        />

        <div className="sys-group">Search</div>
        <Row label="Casualties" value={String(gids.length)} color={gids.length ? theme.accent : undefined} />
        <Row label="Frames" value={String(stats?.frames ?? 0)} />
        <Row label="Detection rate" value={fmt(stats?.fpsLast, 1, " fps")} />
        <Row label="Run" value={mission?.name ?? "—"} />
      </div>
    </Pane>
  );
}
