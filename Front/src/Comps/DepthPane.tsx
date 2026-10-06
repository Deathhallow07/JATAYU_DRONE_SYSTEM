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

const n = (v: number | undefined, d = 2) => (v == null ? "—" : v.toFixed(d));

export default function DepthPane({ area, expanded, onExpand }: {
  area: string; expanded?: boolean; onExpand?: () => void;
}) {
  const { frames, workers, send } = useMission();
  const frame = frames.depth;
  const status = workers.depth;

  const state = status?.state ?? (frame ? "running" : "stopped");

  return (
    <Pane
      area={area}
      title="Depth"
      subtitle={frame?.backend === "stub" ? "stub backend" : frame?.backend ? `${frame.backend} · ${frame.device}` : undefined}
      expanded={expanded}
      onExpand={onExpand}
      status={{ color: STATE_COLOR[state] ?? theme.textFaint, label: status?.error ?? state }}
      actions={
        frame && (
          <>
            <button
              className="btn"
              onClick={() => send("run_worker", { worker: "depth", action: "pause" })}
              title="Stop this feed updating, so the current frame can be studied"
            >
              ❚❚ Pause
            </button>
            <button
              className="btn"
              onClick={() => send("run_worker", { worker: "depth", action: "play" })}
              title="Resume this feed"
            >
              ▶ Resume
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
            <>Depth worker error:<code>{status.error}</code></>
          ) : (
            <>
              Depth worker not running.
              <code>python Python/workers/depth_worker.py \
  --source &lt;video|image|0&gt; --backend stub</code>
            </>
          )}
        </div>
      ) : (
        <>
          <div className="grow">
            <FrameView jpeg={frame.jpeg} width={frame.width} height={frame.height} alt="Depth estimate" />
          </div>
          <div className="readouts fixed">
            <span>min <b>{n(frame.depthMin, 3)}</b></span>
            <span>max <b>{n(frame.depthMax, 3)}</b></span>
            <span>mean <b>{n(frame.depthMean, 3)}</b></span>
            <span>latency <b>{n(frame.latencyMs, 0)} ms</b></span>
            <span>frame <b>#{frame.seq}</b></span>
            {frame.backend === "stub" && (
              <span style={{ color: theme.warning }}>placeholder — not real depth</span>
            )}
          </div>
        </>
      )}
    </Pane>
  );
}
