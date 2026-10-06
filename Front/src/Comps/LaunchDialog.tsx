import { useCallback, useEffect, useRef, useState } from "react";
import { useMission } from "../context/MissionContext";
import { RELAY_URL } from "../config";
import { theme } from "../theme";
import type { MediaEntry, MediaListing } from "../types";

/**
 * Pick a clip and a flight log, and start the pipeline on them.
 *
 * ## Why this browses the server's filesystem
 *
 * The obvious control is `<input type="file">`, and it cannot work here. A
 * browser hands back a sandboxed Blob and a bare filename — never a path — and
 * the process that has to open the clip is `Pipeline_yoloe_logs_robust.py`, on
 * the machine with the GPU. Uploading a 450 MB recording through the relay to
 * put it back on the disk it is already sitting on would be the only way to
 * make a file input mean anything.
 *
 * So the picker lists the relay's own filesystem (`GET /media`), confined to
 * MEDIA_ROOTS, and what crosses the wire is a path.
 *
 * ## Why the CSV is part of choosing the video
 *
 * The two are one recording. Without the flight log the pipeline has no
 * attitude and no fix for the frame it is looking at, so no detection can be
 * projected to the ground: there is no casualty pin, no geo gate on
 * re-identification, and the map has nothing to fly. The console would come up
 * looking like it was working.
 *
 * The CSV is therefore preselected when one is sitting next to the clip — these
 * recordings are written as a pair — and choosing to run without one is an
 * explicit act with the consequence spelled out.
 */

type Step = "video" | "csv";

const RELAY_DOWN =
  "Cannot reach the relay, so there is nothing to browse and nothing to " +
  "launch a run.\n\nStart it:    cd Back && npm run start:prod\n\n" +
  "If it is already running, the dev server was started before the /media " +
  "proxy was added — restart it:    cd Front && npm run dev";

const fmtSize = (bytes: number | null) => {
  if (bytes == null) return "";
  const mb = bytes / 1e6;
  return mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb.toFixed(0)} MB`;
};

/** Shorten a long absolute path from the left, keeping the end that identifies it. */
const tail = (p: string | null | undefined, keep = 3) => {
  if (!p) return "—";
  const parts = p.split("/").filter(Boolean);
  return parts.length <= keep ? p : `…/${parts.slice(-keep).join("/")}`;
};

export default function LaunchDialog({ onClose }: { onClose: () => void }) {
  const { run, send, connected } = useMission();

  const [step, setStep] = useState<Step>("video");
  const [listing, setListing] = useState<MediaListing | null>(null);
  const [loading, setLoading] = useState(true);
  const [video, setVideo] = useState<string | null>(null);
  const [csv, setCsv] = useState<string | null>(null);
  const [startFrame, setStartFrame] = useState(0);
  const [workers, setWorkers] = useState(false);
  const [refused, setRefused] = useState<string | null>(null);
  // What is typed in the path box. Kept separate from `listing.path` so typing
  // does not fight the listing that arrives while you are still typing.
  const [typed, setTyped] = useState("");

  const cfg = run?.config;
  const active = run?.state === "running" || run?.state === "starting";

  // `connected` is the socket to the relay. Without it "Start run" emits into
  // nothing and the dialog would sit looking like it had worked.
  const relayDown = !connected;

  /**
   * Read a directory from the relay.
   *
   * The failure that matters here is the relay not running at all, and it does
   * not arrive as a network error: the Vite dev server answers /media itself
   * with its own HTML when it cannot reach the proxy target, so `res.json()`
   * throws "unexpected end of data" and the dialog would report a JSON parsing
   * problem for a relay that is simply not started. Every non-JSON answer is
   * therefore reported as what it actually is, with the command that fixes it.
   */
  const load = useCallback(async (path: string | null, suggestFor?: string) => {
    setLoading(true);
    try {
      const qs = new URLSearchParams();
      if (path) qs.set("path", path);
      if (suggestFor) qs.set("suggestCsvFor", suggestFor);

      const res = await fetch(`${RELAY_URL}/media?${qs}`);
      const type = res.headers.get("content-type") ?? "";

      if (!res.ok || !type.includes("application/json")) {
        setListing({
          path, roots: [], dirs: [], files: [],
          error: RELAY_DOWN,
        });
        return null;
      }

      const data: MediaListing = await res.json();
      setListing(data);
      return data;
    } catch {
      // A thrown fetch is the relay refusing the connection outright.
      setListing({ path, roots: [], dirs: [], files: [], error: RELAY_DOWN });
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Open where the last run's clip came from, so launching a second run over
  // the same footage is two clicks rather than a walk back down the tree.
  useEffect(() => {
    void load(run?.run?.video ?? null);
  }, [load, run?.run?.video]);

  useEffect(() => {
    if (listing?.path) setTyped(listing.path);
  }, [listing?.path]);

  /**
   * Go to a typed or pasted path.
   *
   * The reason this exists: footage is wherever it is, and finding it by
   * clicking from the filesystem root is a dozen round trips. Anyone who knows
   * where their clip lives — or has the path on their clipboard from the
   * terminal they just came from — can say so directly. A file path jumps to
   * its directory AND selects it, so pasting the clip's full path is the whole
   * interaction.
   */
  const goTo = async (raw: string) => {
    const target = raw.trim();
    if (!target) return;
    const data = await load(target, step === "video" ? target : undefined);
    if (!data || data.error) return;
    const hit = data.files.find((f) => f.path === target);
    if (!hit) return;
    if (hit.kind === "video") {
      setVideo(hit.path);
      if (data.suggestedCsv) setCsv(data.suggestedCsv);
      setStep("csv");
    } else {
      setCsv(hit.path);
    }
  };

  const pickVideo = async (entry: MediaEntry) => {
    setVideo(entry.path);
    setRefused(null);
    const data = await load(entry.path, entry.path);
    // Preselected, not silently applied: the operator sees which log was
    // chosen and can pick another before starting.
    if (data?.suggestedCsv) setCsv(data.suggestedCsv);
    setStep("csv");
  };

  const start = () => {
    setRefused(null);
    send("run_start", { video, csv, startFrame, workers });
  };

  /**
   * Run the footage that ships with the checkout.
   *
   * Anyone who has this folder and none of the recordings it was developed
   * against still has one flight to look at, and finding it means knowing it
   * is in pipeline/1xzoom1.5ms20m. The relay reports the pair it found there
   * (probe().sample) and this starts it in one click — the fields are filled
   * in as well, so what is running is visible rather than magic.
   */
  const sample = cfg?.sample ?? null;
  const runSample = () => {
    if (!sample) return;
    setRefused(null);
    setVideo(sample.video);
    setCsv(sample.csv);
    setStartFrame(sample.startFrame);
    setStep("csv");
    void load(sample.video);
    send("run_start", {
      video: sample.video,
      csv: sample.csv,
      startFrame: sample.startFrame,
      workers,
    });
  };

  const stop = () => send("run_stop", {});

  // A refusal comes back on the ack, and the socket helper in the context does
  // not expose acks — so surface the launcher's own error instead, which the
  // relay puts on the status it broadcasts either way.
  useEffect(() => {
    if (run?.state === "error" && run.error) setRefused(run.error);
  }, [run?.state, run?.error]);

  const outputRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = outputRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [run?.output?.length]);

  const videos = listing?.files.filter((f) => f.kind === "video") ?? [];
  const csvs = listing?.files.filter((f) => f.kind === "csv") ?? [];
  const shown = step === "video" ? videos : csvs;

  return (
    <div className="drawer-backdrop launch-backdrop" onClick={onClose}>
      <div className="launch" onClick={(e) => e.stopPropagation()}>
        <header className="pane-head">
          <span className="pane-title">Start a run</span>
          <span className="pane-sub">
            {step === "video" ? "Choose the video" : "Choose the flight log"}
          </span>
          <span className="spacer" />
          <button className="btn" onClick={onClose}>Close</button>
        </header>

        <div className="launch-body">
          {/* ── What has been chosen so far ───────────────────────────── */}
          <div className="launch-picked">
            <button
              className={`btn${step === "video" ? " active" : ""}`}
              onClick={() => { setStep("video"); void load(video ?? listing?.path ?? null); }}
            >
              1 · Video
            </button>
            <span className="launch-path" title={video ?? undefined}>{tail(video)}</span>

            <button
              className={`btn${step === "csv" ? " active" : ""}`}
              onClick={() => { setStep("csv"); void load(video ?? listing?.path ?? null); }}
              disabled={!video}
              title={video ? "Choose the telemetry CSV" : "Choose a video first"}
            >
              2 · Telemetry CSV
            </button>
            <span className="launch-path" title={csv ?? undefined}>
              {csv ? tail(csv) : <em style={{ color: theme.warning }}>none</em>}
            </span>

            {/* In the step column rather than the header: it fills both steps
                at once, so it belongs beside them — and it lands under Close,
                in the corner the two rows leave empty. */}
            {sample && (
              <button
                className="btn launch-sample"
                onClick={runSample}
                disabled={active || relayDown}
                title={relayDown
                  ? "The relay is not running — start it with: cd Back && npm run start:prod"
                  : active
                    ? "A run is already going — stop it first"
                    : `Run the footage bundled with this checkout: ${sample.video.split("/").pop()}` +
                      `${sample.csv ? ` with ${sample.csv.split("/").pop()}` : " with no telemetry"}` +
                      `, from frame ${sample.startFrame}`}
              >
                Sample
              </button>
            )}
          </div>

          {/* ── The consequence of no CSV, stated where the choice is made ── */}
          {step === "csv" && !csv && (
            <div className="launch-warn">
              Without a flight log the pipeline has no position or attitude for
              the frame it is looking at, so no detection can be placed on the
              ground: no casualty pins on the map, no geographic gate on
              re-identification, and no drone to follow. Detections and crops
              still work.
            </div>
          )}

          {/* ── Browser ───────────────────────────────────────────────── */}
          <div className="launch-crumbs">
            {listing?.parent && (
              <button
                className="btn"
                onClick={() => void load(listing.parent!)}
                title={`Go up to ${listing.parent}`}
              >
                ↑ Up
              </button>
            )}
            {(listing?.shortcuts ?? []).map((sc) => (
              <button
                key={sc.path}
                className="btn"
                onClick={() => void load(sc.path)}
                title={`Jump to ${sc.path}`}
              >
                {sc.name}
              </button>
            ))}
          </div>

          {/* Type or paste any path. Faster than clicking down a tree, and the
              only practical way to reach footage on a mount you would otherwise
              have to walk to from the filesystem root. */}
          <form
            className="launch-goto"
            onSubmit={(e) => { e.preventDefault(); void goTo(typed); }}
          >
            <input
              className="launch-pathbox"
              value={typed}
              spellCheck={false}
              autoComplete="off"
              placeholder="/path/to/folder — or paste a file path and press Enter"
              onChange={(e) => setTyped(e.target.value)}
              title="Type or paste an absolute path. A folder opens it; a file selects it."
            />
            <button className="btn" type="submit" title="Open this path">Go</button>
          </form>

          <div className="launch-list">
            {loading && <div className="launch-note">Reading…</div>}

            {!loading && listing?.error && (
              <div className="launch-note launch-error">{listing.error}</div>
            )}

            {!loading && listing?.dirs.map((d) => (
              <button key={d.path} className="launch-row dir" onClick={() => void load(d.path)}>
                <span className="launch-row-name">{d.name}/</span>
              </button>
            ))}

            {!loading && shown.map((f) => {
              const selected = step === "video" ? video === f.path : csv === f.path;
              return (
                <button
                  key={f.path}
                  className={`launch-row${selected ? " selected" : ""}`}
                  onClick={() => (step === "video" ? void pickVideo(f) : setCsv(f.path))}
                >
                  <span className="launch-row-name">{f.name}</span>
                  <span className="launch-row-meta">{fmtSize(f.size)}</span>
                </button>
              );
            })}

            {!loading && !listing?.error && shown.length === 0 && listing?.dirs.length === 0 && (
              <div className="launch-note">
                No {step === "video" ? "video files" : "CSV files"} here.
              </div>
            )}
          </div>

          {/* ── Options ───────────────────────────────────────────────── */}
          <div className="launch-opts">
            <label>
              Start frame
              <input
                className="launch-num"
                type="number"
                min={0}
                step={1}
                value={startFrame}
                onChange={(e) => setStartFrame(Math.max(0, Number(e.target.value) || 0))}
                title="Enter a long clip part-way through. The CSV follows the same frame."
              />
            </label>
            <label title="Depth and traversability, off the pipeline's own stream. Two more model loads on the same cores.">
              <input
                type="checkbox"
                checked={workers}
                onChange={(e) => setWorkers(e.target.checked)}
              />
              Depth + traversability feeds
            </label>
            {csv && (
              <button className="btn" onClick={() => setCsv(null)} title="Run without telemetry">
                Clear CSV
              </button>
            )}
          </div>

          {/* ── Anything missing, said before the run is started ───────── */}
          {cfg && (
            <div className="launch-cfg">
              {(["pipelineScript", "pipelinePy", "guiPy", "yoloeWeights", "reidCkpt"] as const)
                .map((k) => (
                  <span key={k} className="chip" title={cfg[k].path || "not configured"}>
                    <span className="dot" style={{ background: cfg[k].ok ? theme.success : theme.danger }} />
                    {k}
                  </span>
                ))}
              <span className="chip" title="The pipeline's MJPEG server, read by the Geolocalisation feed">
                stream :{cfg.streamPort}
              </span>
            </div>
          )}

          {relayDown && (
            <div className="launch-warn" style={{ color: theme.danger }}>
              The console is not connected to the relay. Nothing can be started
              until it is.
            </div>
          )}

          {refused && <div className="launch-warn" style={{ color: theme.danger }}>{refused}</div>}

          {/* ── What the run is doing ─────────────────────────────────── */}
          {run && run.state !== "idle" && (
            <>
              <div className="launch-procs">
                <span className="chip" style={{ color: active ? theme.success : theme.textDim }}>
                  {run.state}
                </span>
                {run.procs.map((p) => (
                  <span
                    key={p.name}
                    className="chip"
                    title={p.exitCode == null ? `pid ${p.pid ?? "—"}` : `exit ${p.exitCode}`}
                  >
                    <span
                      className="dot"
                      style={{
                        background: p.state === "running" ? theme.success
                          : p.state === "finished" ? theme.textDim : theme.danger,
                      }}
                    />
                    {p.name}
                  </span>
                ))}
              </div>
              <div className="launch-output" ref={outputRef}>
                {run.output.map((o, i) => (
                  <div key={`${o.ts}-${i}`} className="launch-out-line">
                    <span className="launch-out-proc">{o.proc}</span>
                    {o.line}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        <footer className="launch-foot">
          <span className="launch-hint">
            {active
              ? "The pipeline is running. Casualties appear as it finds them."
              : video
                ? csv
                  ? "The drone flies the log in step with the frame being processed."
                  : "No flight log — detections only."
                : "Pick a video to start."}
          </span>
          <span className="spacer" />
          {active ? (
            <button
              className="btn"
              onClick={stop}
              style={{ color: theme.danger }}
              title="Stop the pipeline, the watcher and the telemetry replay"
            >
              Stop run
            </button>
          ) : (
            <button
              className="btn active"
              onClick={start}
              disabled={!video || relayDown}
              title={relayDown
                ? "The relay is not running — start it with: cd Back && npm run start:prod"
                : video
                  ? `Run the pipeline on ${video.split("/").pop()}${csv ? ` with ${csv.split("/").pop()}` : " with no telemetry"}`
                  : "Choose a video first"}
            >
              Start run
            </button>
          )}
        </footer>
      </div>
    </div>
  );
}
