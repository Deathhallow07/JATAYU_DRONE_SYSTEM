import { useEffect, useRef, useState } from "react";
import Pane, { EmptyState } from "./Pane";
import FrameView from "./FrameView";
import { useMission } from "../context/MissionContext";
import { pipelineStreamUrl } from "../config";
import { theme } from "../theme";

/**
 * The pipeline's annotated output — detection boxes, track ids and the
 * geolocalisation overlay.
 *
 * Two producers can fill this pane, and it prefers whichever is actually
 * running:
 *
 *  1. `mission_run.py` publishes annotated frames over the relay as the
 *     `detect` worker. This is the live path, and it is the one that is in step
 *     with the casualties appearing and the drone moving, because the same
 *     process emits all three.
 *
 *  2. `Pipeline_yoloe_logs_robust.py` serves MJPEG on STREAM_PORT when
 *     DISPLAY_MODE is "stream". Used when the real pipeline is driving the
 *     console. An <img> pulls it straight from the pipeline host at full frame
 *     rate rather than base64'ing it through a socket that is also carrying
 *     every log line.
 *
 * The socket feed wins when present: if both are up, the one carrying this
 * run's GIDs is the one to show.
 *
 * The MJPEG path cannot tell a stalled stream from a slow one — an <img> that
 * stops receiving parts fires no event. Hence the explicit Reconnect, which
 * appends a cache-buster to force a fresh GET rather than a revalidation of a
 * connection the browser still believes is open.
 */
export default function PipelinePane({ area, expanded, onExpand }: {
  area: string; expanded?: boolean; onExpand?: () => void;
}) {
  const { frames, workers, producerConnected, run } = useMission();
  const frame = frames.detect;
  const status = workers.detect;

  const [nonce, setNonce] = useState(() => Date.now());
  const [failed, setFailed] = useState(false);
  const [live, setLive] = useState(false);
  const imgRef = useRef<HTMLImageElement | null>(null);

  // A run that just started is a stream server that just came up, on a port
  // that was refusing connections a moment ago. An <img> that already failed
  // never retries on its own, so the pane would sit on "No annotated feed" for
  // the whole run unless someone thought to press Reconnect.
  //
  // The pipeline loads its models first, so the port is not listening the
  // instant the run starts: retry while the run is up and the image has not
  // loaded, rather than once.
  const runStartedAt = run?.run?.startedAt ?? null;
  const runActive = run?.state === "running" || run?.state === "starting";
  useEffect(() => {
    if (!runStartedAt) return;
    setFailed(false);
    setLive(false);
    setNonce(Date.now());
  }, [runStartedAt]);

  useEffect(() => {
    if (!runActive || live) return;
    const id = setInterval(() => setNonce(Date.now()), 4000);
    return () => clearInterval(id);
  }, [runActive, live]);

  // A frame that stopped arriving means the run ended or the producer died;
  // either way the pane should stop claiming to be live.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  const socketFresh = frame != null && now - frame.ts * 1000 < 5000;

  // Resolved from the relay's own config, so changing PIPELINE_STREAM_PORT in
  // Python/.env moves both the pipeline and this pane together.
  const base = pipelineStreamUrl(run?.config);
  const src = `${base}${base.includes("?") ? "&" : "?"}t=${nonce}`;

  // Unmounting must actually stop the transfer. An MJPEG response never ends,
  // so a detached <img> keeps its connection open and the pipeline keeps
  // encoding for a viewer that is gone — clearing src closes it.
  useEffect(() => {
    const el = imgRef.current;
    return () => { if (el) el.src = ""; };
  }, []);

  const reconnect = () => {
    setFailed(false);
    setLive(false);
    setNonce(Date.now());
  };

  // ── Live socket feed from mission_run.py ────────────────────────────────
  if (frame) {
    const vt = typeof frame.videoTime === "number" ? frame.videoTime : null;
    const stamp = vt == null
      ? undefined
      : `t+${String(Math.floor(vt / 60)).padStart(2, "0")}:${(vt % 60).toFixed(1).padStart(4, "0")}`;

    return (
      <Pane
        area={area}
        title="Geolocalisation"
        subtitle={[status?.state === "error" ? "error" : socketFresh ? "live" : "ended", stamp]
          .filter(Boolean).join(" · ")}
        expanded={expanded}
        onExpand={onExpand}
        status={{
          color: socketFresh ? theme.success : theme.textFaint,
          label: socketFresh ? "Detections streaming" : "Feed ended",
        }}
        column
      >
        <div className="grow">
          <FrameView
            jpeg={frame.jpeg}
            alt="Pipeline detections and geolocalisation overlay"
            width={frame.width}
            height={frame.height}
          />
        </div>
        <div className="readouts fixed">
          <span>detector <b>{frame.backend ?? "—"}</b></span>
          <span>device <b>{frame.device ?? "—"}</b></span>
          <span>latency <b>{frame.latencyMs} ms</b></span>
          <span>frame <b>{frame.seq}</b></span>
        </div>
      </Pane>
    );
  }

  // ── MJPEG fallback: the real pipeline's own stream server ───────────────
  return (
    <Pane
      area={area}
      title="Geolocalisation"
      subtitle={failed ? "no feed" : live ? "live · mjpeg" : "connecting…"}
      expanded={expanded}
      onExpand={onExpand}
      status={{
        color: failed ? theme.danger : live ? theme.success : theme.warning,
        label: failed ? "No annotated feed" : live ? "Streaming" : "Connecting",
      }}
      actions={
        <button className="btn" onClick={reconnect} title="Re-open the MJPEG stream">
          Reconnect
        </button>
      }
      center={!failed}
    >
      {failed ? (
        <EmptyState>
          No annotated feed.
          <br />
          {producerConnected
            ? "A producer is connected but not publishing detections."
            : "Nothing is running."}
          <code>
            {`# live, one script — this fills the pane over the relay\npython Python/mission_run.py \\\n  --video clip.mp4 --telem fcb.csv\n\n# or the real pipeline's own MJPEG server\n# Pipeline_yoloe_logs_robust.py\nDISPLAY_MODE = "stream"\nSTREAM_PORT  = 8080`}
          </code>
        </EmptyState>
      ) : (
        <div className="frame-wrap">
          <img
            ref={imgRef}
            className="frame-img"
            src={src}
            alt="Pipeline detections and geolocalisation overlay"
            onLoad={() => { setLive(true); setFailed(false); }}
            onError={() => { setLive(false); setFailed(true); }}
          />
        </div>
      )}
    </Pane>
  );
}
