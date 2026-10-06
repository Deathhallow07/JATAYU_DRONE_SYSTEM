// In dev, Vite proxies /socket.io and /artifacts to the relay (vite.config.ts),
// so everything stays same-origin. A built bundle served from somewhere else
// needs VITE_RELAY_URL pointing at the relay.
export const RELAY_URL: string = import.meta.env.VITE_RELAY_URL ?? window.location.origin;

/**
 * Resolve a run-folder-relative artifact path to a URL the browser can fetch.
 *
 * `version` is a cache-buster, and it is load-bearing rather than defensive.
 * The pipeline OVERWRITES a casualty's representative.jpg in place every time
 * a sharper crop of that person arrives — same run folder, same filename, new
 * picture. Without a changing query string the browser keeps serving the first
 * one it fetched, so a casualty's photo would freeze at the blurriest crop the
 * pipeline ever had of them and never update again for the rest of the run.
 *
 * Pass a GID's `updates` counter: it is the pipeline's own revision number for
 * that record, so the URL changes exactly when the file behind it does and not
 * on every re-render.
 */
export const artifactUrl = (
  rel: string | null | undefined,
  version?: number | string | null,
): string | undefined =>
  rel
    ? `${RELAY_URL}/artifacts/${rel}${version == null ? "" : `?v=${version}`}`
    : undefined;

/**
 * Mapbox token for the mission map.
 *
 * Same token and same override the existing GCS uses (Front/src/config.tsx there),
 * so one account covers both consoles. Set VITE_MAPBOX_TOKEN in Front/.env
 * (see Front/.env.example); it is deliberately not committed.
 */
export const MAPBOX_TOKEN: string =
  import.meta.env.VITE_MAPBOX_TOKEN ?? "";

/**
 * The pipeline's own annotated feed — detections, track ids and the
 * geolocalisation overlay, exactly as the pipeline draws them.
 *
 * Pipeline_yoloe_logs_robust.py serves this as MJPEG when DISPLAY_MODE is set
 * to "stream" (STREAM_PORT, default 8080). It is an <img> src, not a socket
 * stream: MJPEG over HTTP goes straight from the pipeline host to the browser
 * without being base64'd through the relay, which matters at full frame rate.
 */
export const PIPELINE_STREAM_URL: string =
  import.meta.env.VITE_PIPELINE_STREAM ?? `http://${window.location.hostname}:8080/`;

/**
 * The same thing, but resolved at RUNTIME from what the relay reports.
 *
 * The port is configured once, as PIPELINE_STREAM_PORT in Python/.env, and the
 * relay passes it to the pipeline it launches AND reports it here — so the two
 * cannot drift apart. They silently could otherwise: a console built against
 * 8080 while the pipeline serves 8081 shows an empty pane and no error, because
 * an <img> whose source never loads has nothing to report.
 *
 * `streamUrl` wins when set (PIPELINE_STREAM_URL in the .env), for a pipeline
 * on a different host from the browser — a port alone cannot find that.
 * Falling back to the build-time constant covers a console open before any run
 * has been launched.
 */
export const pipelineStreamUrl = (
  cfg?: { streamPort?: number; streamUrl?: string | null } | null,
): string => {
  if (cfg?.streamUrl) return cfg.streamUrl;
  if (cfg?.streamPort) return `http://${window.location.hostname}:${cfg.streamPort}/`;
  return PIPELINE_STREAM_URL;
};

/** Fallback map centre before any fix arrives — DTU, matching the existing GCS. */
export const DEFAULT_CENTER: [number, number] = [77.115481, 28.753619];
