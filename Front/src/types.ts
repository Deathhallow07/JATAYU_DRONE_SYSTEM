// Wire types. These mirror what Python/mission_watcher.py and the workers emit
// — keep the two in step.

export interface LogLine {
  seq: number;
  ts: number;
  level: string;
  msg: string;
}

export interface TrackPoint {
  frame: number;
  conf: number;
  sharpness: number;
  lat: number | string;
  lon: number | string;
}

export interface BufferRecord {
  event: string;
  tid: number | null;
  wall_time?: string;
  mission_timestamp?: string;
  cumulative_gallery?: number;
  buffer_gallery_size?: number;
  entryCount: number;
  [k: string]: unknown;
}

export interface Gid {
  gid: number;
  created?: string;
  event: string;
  updates: number;
  tid: number | null;

  lat: number | null;
  lon: number | null;
  gpsSamples: number | null;

  gallerySize: number;
  bestConf: number | null;
  bestSharpness: number | null;
  firstFrame: number | null;
  lastFrame: number | null;
  missionTimestamp: string | null;
  rosTime: number | null;

  representative: string | null;
  crops: string[];
  buffers: BufferRecord[];
  track: TrackPoint[];
}

export interface Mission {
  runFolder: string;
  name: string;
  startedAt: string;
  replay: boolean;
  speed: number;
  runs?: { name: string; path: string; gids: number }[];
}

export interface Stats {
  frames: number;
  gids: number;
  logLines: number;
  fpsLast: number | null;
  fpsMin: number | null;
  fpsMax: number | null;
  levels: Record<string, number>;
  elapsed?: number;
  paused?: boolean;
  speed?: number;
  startedAt: number;
}

export interface Frame {
  worker: "depth" | "seg" | string;
  seq: number;
  ts: number;
  jpeg: string;
  width: number;
  height: number;
  latencyMs: number;
  backend?: string;
  device?: string;
  /** Seconds into the clip, on the detect feed from mission_run.py. */
  videoTime?: number;

  // depth
  depthMin?: number;
  depthMax?: number;
  depthMean?: number;

  // seg / traversability
  grid?: number[][];
  gridSize?: [number, number];
  path?: [number, number][];
  pathCells?: number;
  pathLengthCells?: number;
  start?: [number, number] | null;
  goal?: [number, number] | null;
  traversableFraction?: number;
  reachable?: boolean;
}

export interface WorkerStatus {
  worker: string;
  state: "loading" | "running" | "stopped" | "error";
  error?: string;
  source?: string;
}

/**
 * One MAVLink sample, republished by Python/telem_worker.py.
 *
 * Field names mirror the existing GCS's UAVs/Telem.py so the two consoles' system
 * panels are interchangeable — hence the capitalised Status/Last_Heartbeat,
 * which are wire names, not a style slip.
 */
export interface Telemetry {
  uavId: number;
  latitude: number;
  longitude: number;
  altitude: number | null;
  groundspeed: number | null;
  /** Pack voltage, volts. */
  battery: number | null;
  /** Remaining charge, percent. null when the firmware has no capacity set. */
  batteryPct: number | null;
  current: number | null;
  armed: boolean;
  mode: string | null;
  heading: number | null;
  Status: string;
  /** Seconds between the last two heartbeats. */
  Last_Heartbeat: number;
  ts: number;

  // ── Only from telem_csv.py (a flight log being replayed) ────────────────
  /**
   * Where this sample came from. Absent means a live MAVLink link
   * (telem_worker.py); "csv-pipeline" is a recording being replayed in step
   * with the pipeline, "csv-realtime" one played at wall-clock speed.
   *
   * Shown in the System pane, because a replayed flight and an aircraft that
   * is actually airborne must never look the same on a wall display.
   */
  source?: "csv-pipeline" | "csv-realtime" | string;
  /** Video frame this fix belongs to — the frame the pipeline is processing. */
  frame?: number | null;
  roll?: number | null;
  pitch?: number | null;
  yaw?: number | null;
  zoom?: number | null;
}

// ── Launching a run from the console ─────────────────────────────────────────
// Mirrors Back/pipeline.js. The relay owns the pipeline process; these are what
// it reports back about it.

export interface MediaEntry {
  name: string;
  path: string;
  kind: "video" | "csv";
  size: number | null;
  mtime: number | null;
}

export interface MediaListing {
  /** Null at the top level, where `dirs` is the list of MEDIA_ROOTS. */
  path: string | null;
  parent?: string | null;
  roots: string[];
  /** One-click jumps: home, mounts, the configured roots. */
  shortcuts?: { name: string; path: string }[];
  dirs: { name: string; path: string }[];
  files: MediaEntry[];
  suggestedCsv?: string | null;
  error?: string;
}

/** One path the launcher needs, and whether it is actually there. */
export interface LauncherPath {
  path: string;
  ok: boolean;
}

export interface LauncherConfig {
  pipelineScript: LauncherPath;
  pipelinePy: LauncherPath;
  guiPy: LauncherPath;
  yoloeWeights: LauncherPath;
  reidCkpt: LauncherPath;
  streamPort: number;
  /** Set only when the pipeline is on another host; otherwise derived from the port. */
  streamUrl: string | null;
  mediaRoots: string[];
  /**
   * Footage bundled with the checkout, for the Sample button. Null when the
   * folder it lives in is absent, in which case the button is not offered.
   */
  sample: { video: string; csv: string | null; startFrame: number } | null;
}

export interface RunStatus {
  state: "idle" | "starting" | "running" | "stopping" | "stopped" | "error";
  run: {
    video: string;
    csv: string | null;
    startFrame: number;
    workers: boolean;
    fps: number;
    missionRoot: string;
    streamPort: number;
    startedAt: string;
  } | null;
  error: string | null;
  config: LauncherConfig;
  procs: {
    name: string;
    pid: number | null;
    state: "running" | "finished" | "error" | "killed";
    exitCode: number | null;
  }[];
  /** stdout + stderr of every child, so a crash is readable in the dialog. */
  output: { proc: string; line: string; ts: number }[];
}

/** What the medic's notification panel shows, distilled from logs + GIDs. */
export interface Notice {
  id: string;
  kind: "found" | "again" | "mission" | "problem";
  /** Headline, written for someone who has never seen the pipeline. */
  text: string;
  detail?: string;
  gid?: number;
  lat?: number | null;
  lon?: number | null;
  /** Wall-clock ms. */
  ts: number;
}

export interface Snapshot {
  mission: Mission | null;
  logs: LogLine[];
  gids: Gid[];
  stats: Stats | null;
  frames: Record<string, Frame>;
  telemetry: Telemetry | null;
  run: RunStatus | null;
  pythonConnected: boolean;
}
