import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import Pane from "./Pane";
import { useMission } from "../context/MissionContext";
import { levelColor, theme } from "../theme";

// How many rows are actually put in the DOM. The context keeps up to 5000 lines
// so scrollback and filtering work across the whole run, but mounting 5000 rows
// makes every append reflow the lot and the pane visibly stutters on a busy
// pipeline. The newest slice is what an operator watches; the header says when
// older lines are being withheld.
const RENDER_LIMIT = 1200;

// The prefixes the pipeline's log helpers emit, in the order an operator scans
// them: outcomes first, then the per-stage chatter.
const LEVELS = ["OK", "INFO", "WARN", "ERROR", "DET", "REID", "TRACK", "GPS", "MATCH", "FRAME"];

// FRAME is one line per frame ("FPS=12.4") — thousands of them, and they drown
// everything an operator is actually looking for. Off by default; the FPS
// readout in the status bar carries the same information.
const DEFAULT_MUTED = new Set(["FRAME"]);

export default function LogPane({ area, onClose }: {
  area: string; onClose?: () => void;
}) {
  const { logs, mission, send } = useMission();
  const [muted, setMuted] = useState<Set<string>>(new Set(DEFAULT_MUTED));
  const [query, setQuery] = useState("");
  const [follow, setFollow] = useState(true);

  const bodyRef = useRef<HTMLDivElement | null>(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return logs.filter(
      (l) => !muted.has(l.level) && (!q || l.msg.toLowerCase().includes(q)),
    );
  }, [logs, muted, query]);

  const shown = filtered.length > RENDER_LIMIT
    ? filtered.slice(filtered.length - RENDER_LIMIT)
    : filtered;

  // Pin to the bottom after the new rows are laid out but before paint, so
  // following the log never shows a frame scrolled to the wrong place.
  useLayoutEffect(() => {
    if (!follow) return;
    const el = bodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [shown.length, follow]);

  // Scrolling up drops follow; returning to the bottom resumes it. Without
  // this, reading back through the log fights the autoscroll on every batch.
  useEffect(() => {
    const el = bodyRef.current;
    if (!el) return;
    const onScroll = () => {
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
      setFollow(atBottom);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  const toggle = (lvl: string) =>
    setMuted((prev) => {
      const next = new Set(prev);
      if (next.has(lvl)) next.delete(lvl);
      else next.add(lvl);
      return next;
    });

  const replayBtn = (label: string, title: string, action: string, extra?: object) => (
    <button
      className="btn"
      onClick={() => send("replay_control", { action, ...extra })}
      title={title}
    >
      {label}
    </button>
  );

  return (
    <Pane
      area={area}
      title="Mission Log"
      subtitle={
        filtered.length > shown.length
          ? `last ${shown.length} of ${filtered.length}`
          : `${filtered.length} lines`
      }
      column
      actions={
        <>
          {mission?.replay && (
            <>
              {replayBtn("▶ Play", "Resume the replay", "play")}
              {replayBtn("❚❚ Pause", "Hold the replay where it is", "pause")}
              {replayBtn("↻ Restart", "Replay this run from the beginning", "restart")}
              <select
                className="btn"
                defaultValue="1"
                onChange={(e) => send("replay_control", { speed: Number(e.target.value) })}
                title="How fast the finished run is replayed"
              >
                {[0.5, 1, 2, 4, 8, 16].map((s) => (
                  <option key={s} value={s}>{s}×</option>
                ))}
              </select>
            </>
          )}
          {/* "Follow" alone read as "follow the drone", which is what the
              map's button does. This one follows the BOTTOM of the log. */}
          <button
            className={`btn${follow ? " active" : ""}`}
            onClick={() => setFollow((f) => !f)}
            title={follow
              ? "Scrolling to each new line as it arrives. Click to stop and read back."
              : "Jump to the newest line and keep following it"}
          >
            {follow ? "Auto-scrolling" : "Auto-scroll"}
          </button>
          {onClose && (
            <button className="btn" onClick={onClose} title="Close the mission log (L or Esc)">
              Close
            </button>
          )}
        </>
      }
    >
      <div
        className="fixed"
        style={{
          display: "flex", flexWrap: "wrap", gap: 4, padding: "6px 8px",
          borderBottom: `1px solid ${theme.border}`, background: theme.surface1,
        }}
      >
        <input
          className="btn"
          style={{ flex: 1, minWidth: 110, textTransform: "none", letterSpacing: 0, color: theme.text }}
          placeholder="filter lines containing…"
          title="Show only log lines containing this text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        {LEVELS.map((lvl) => (
          <button
            key={lvl}
            className="btn"
            onClick={() => toggle(lvl)}
            style={{
              color: muted.has(lvl) ? theme.textFaint : levelColor(lvl),
              opacity: muted.has(lvl) ? 0.45 : 1,
              borderColor: muted.has(lvl) ? theme.border : levelColor(lvl),
            }}
            title={lvl === "FRAME"
              ? "FRAME is one line per processed frame and is dropped at the "
                + "watcher, so it never reaches this pane — the FPS readout in "
                + "the status bar carries the same number. Restart the watcher "
                + "with --keep-frame-lines to see them."
              : muted.has(lvl)
                ? `${lvl} lines are hidden — click to show them`
                : `Hide ${lvl} lines`}
            disabled={lvl === "FRAME"}
          >
            {lvl}
          </button>
        ))}
      </div>

      <div className="log grow" ref={bodyRef} style={{ overflowY: "auto" }}>
        {shown.length === 0 ? (
          <div className="empty-state">
            No log lines yet.
            <code>python Python/mission_watcher.py --replay</code>
          </div>
        ) : (
          shown.map((l) => (
            <div className="log-row" key={l.seq}>
              <span className="log-level" style={{ color: levelColor(l.level) }}>{l.level}</span>
              <span className="log-msg">{l.msg}</span>
            </div>
          ))
        )}
      </div>
    </Pane>
  );
}
