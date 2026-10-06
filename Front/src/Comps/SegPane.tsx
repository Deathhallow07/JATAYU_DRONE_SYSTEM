import { useEffect, useRef, useState } from "react";
import Pane from "./Pane";
import FrameView from "./FrameView";
import { useMission } from "../context/MissionContext";
import { theme } from "../theme";

const STATE_COLOR: Record<string, string> = {
  running: theme.success,
  loading: theme.warning,
  error: theme.danger,
  stopped: theme.textFaint,
};

type PickMode = "goal" | "start";

/**
 * Traversability pane: the segmented frame with the planned route drawn on it,
 * plus a click target for re-planning.
 *
 * The route itself is rendered into the JPEG by the worker — it owns the grid,
 * so it draws the authoritative overlay. This canvas adds only what depends on
 * the pointer: the hovered cell and its coordinates. Drawing the route twice
 * would guarantee the two disagree the moment a frame is a beat behind.
 */
export default function SegPane({ area, expanded, onExpand }: {
  area: string; expanded?: boolean; onExpand?: () => void;
}) {
  const { frames, workers, send } = useMission();
  const frame = frames.seg;
  const status = workers.seg;
  const state = status?.state ?? (frame ? "running" : "stopped");

  const [mode, setMode] = useState<PickMode>("goal");
  const [hover, setHover] = useState<{ row: number; col: number } | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const cols = frame?.gridSize?.[0];
  const rows = frame?.gridSize?.[1];

  // Repaint the hover highlight. The canvas is a fixed grid-sized bitmap scaled
  // by CSS to the image box, so one cell is exactly one unit here and nothing
  // has to know the pane's pixel size.
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !cols || !rows) return;

    if (cv.width !== cols || cv.height !== rows) {
      cv.width = cols;
      cv.height = rows;
    }

    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, cols, rows);

    if (hover) {
      const blocked = frame?.grid?.[hover.row]?.[hover.col] === 1;
      ctx.fillStyle = blocked ? "rgba(255,40,40,0.55)" : "rgba(120,240,255,0.55)";
      ctx.fillRect(hover.col, hover.row, 1, 1);
    }
  }, [hover, cols, rows, frame?.grid]);

  const pick = (fx: number, fy: number) => {
    if (!cols || !rows) return;
    const col = Math.floor(fx * cols);
    const row = Math.floor(fy * rows);
    send("plan_path", mode === "goal" ? { goal: [row, col] } : { start: [row, col] });
    // Advance to picking the goal after a start, which is the order an
    // operator works in: "responder is here, casualty is there".
    if (mode === "start") setMode("goal");
  };

  const move = (ev: React.MouseEvent<HTMLDivElement>) => {
    if (!cols || !rows) return;
    const r = (ev.currentTarget as HTMLElement).getBoundingClientRect();
    if (!r.width || !r.height) return;
    setHover({
      col: Math.min(cols - 1, Math.floor(((ev.clientX - r.left) / r.width) * cols)),
      row: Math.min(rows - 1, Math.floor(((ev.clientY - r.top) / r.height) * rows)),
    });
  };

  const pct = (v: number | undefined) => (v == null ? "—" : `${(v * 100).toFixed(0)}%`);

  return (
    <Pane
      area={area}
      title="Traversability & Route"
      subtitle={frame?.backend === "stub" ? "stub backend" : frame?.backend ? `${frame.backend} · ${frame.device}` : undefined}
      expanded={expanded}
      onExpand={onExpand}
      status={{
        color: frame && frame.reachable === false ? theme.warning : (STATE_COLOR[state] ?? theme.textFaint),
        label: frame?.reachable === false ? "No route to goal" : (status?.error ?? state),
      }}
      actions={
        frame && (
          <>
            <button
              className={`btn${mode === "start" ? " active" : ""}`}
              onClick={() => setMode("start")}
              title="Then click the frame to place where the responder is starting from"
            >
              Set start
            </button>
            <button
              className={`btn${mode === "goal" ? " active" : ""}`}
              onClick={() => setMode("goal")}
              title="Then click the frame to place the casualty the route should reach"
            >
              Set goal
            </button>
            <button
              className="btn"
              onClick={() => send("plan_path", { action: "clear_path" })}
              title="Forget the points you placed and go back to the automatically planned route"
            >
              Auto route
            </button>
          </>
        )
      }
      center={!frame}
      column={!!frame}
    >
      {!frame ? (
        <div className="empty-state">
          {status?.state === "error" ? (
            <>Segmentation worker error:<code>{status.error}</code></>
          ) : (
            <>
              Traversability worker not running.
              <code>python Python/workers/seg_worker.py \
  --source &lt;video|image|0&gt; --backend stub</code>
            </>
          )}
        </div>
      ) : (
        <>
          <div
            className="grow"
            onMouseMove={move}
            onMouseLeave={() => setHover(null)}
          >
            <FrameView
              jpeg={frame.jpeg}
              width={frame.width}
              height={frame.height}
              alt="Traversability map with planned route"
              onPick={pick}
            >
              <canvas
                className="frame-overlay"
                ref={canvasRef}
                style={{ imageRendering: "pixelated", pointerEvents: "none" }}
              />
            </FrameView>
          </div>

          <div className="readouts fixed">
            <span>
              route{" "}
              <b style={{ color: frame.reachable ? theme.success : theme.danger }}>
                {frame.reachable ? `${frame.pathCells} cells` : "unreachable"}
              </b>
            </span>
            <span>length <b>{frame.pathLengthCells ?? "—"}</b></span>
            <span>traversable <b>{pct(frame.traversableFraction)}</b></span>
            <span>grid <b>{cols}×{rows}</b></span>
            <span>
              start <b>{frame.start ? `${frame.start[0]},${frame.start[1]}` : "—"}</b>
              {" → "}
              goal <b>{frame.goal ? `${frame.goal[0]},${frame.goal[1]}` : "—"}</b>
            </span>
            {hover && (
              <span style={{ color: theme.accent }}>
                cursor {hover.row},{hover.col}{" "}
                {frame.grid?.[hover.row]?.[hover.col] === 1 ? "· blocked" : "· free"}
              </span>
            )}
            <span>latency <b>{frame.latencyMs.toFixed(0)} ms</b></span>
            {frame.backend === "stub" && (
              <span style={{ color: theme.warning }}>placeholder segmentation</span>
            )}
          </div>
        </>
      )}
    </Pane>
  );
}
