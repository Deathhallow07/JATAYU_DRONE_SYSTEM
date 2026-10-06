import {
  createContext, useCallback, useContext, useEffect, useMemo, useRef, useState,
} from "react";
import type { ReactNode } from "react";
import { io, type Socket } from "socket.io-client";
import { RELAY_URL } from "../config";
import type {
  Frame, Gid, LogLine, Mission, RunStatus, Snapshot, Stats, Telemetry, WorkerStatus,
} from "../types";

// Log retention in the browser. The relay caps its own snapshot at 2000 lines;
// this is larger so scrollback survives a long run, but still bounded — a
// four-hour mission emits hundreds of thousands of lines and an unbounded array
// of them is the one thing certain to kill the tab mid-demo.
const MAX_LOGS = 5000;

// A telemetry sample older than this means the link is down, not that the
// drone is hovering perfectly still. Three missed 4 Hz samples plus slack.
const TELEM_STALE_MS = 3000;

// How much of the drone's path to keep on the map.
//
// Counted in DISTINCT positions, not samples: a stationary drone adds nothing
// (see the dedupe below), so this is 12000 places the aircraft has actually
// been. Replaying a flight log frame-by-frame that is the whole of a long
// clip, which is the point — "where has it already searched" is the question
// the trail answers, and a trail that silently drops the first half of the
// sweep answers it wrongly.
//
// Still bounded: an unbounded trail is a GeoJSON LineString that grows all
// mission and is re-serialised to the map on every fix.
const TRAIL_MAX = 12000;

const num = (v: unknown): number | null => {
  const n = typeof v === "string" ? Number(v) : (v as number);
  return typeof n === "number" && Number.isFinite(n) ? n : null;
};

interface MissionState {
  connected: boolean;
  producerConnected: boolean;
  mission: Mission | null;
  logs: LogLine[];
  gids: Gid[];
  stats: Stats | null;
  frames: Record<string, Frame>;
  workers: Record<string, WorkerStatus>;
  /** Newest MAVLink sample, or null if no telemetry has ever arrived. */
  telemetry: Telemetry | null;
  /** False once telemetry stops arriving — the panel shows NO TELEM. */
  telemetryLive: boolean;
  /**
   * Where to draw the drone, [lng, lat]. Telemetry when the link is up;
   * otherwise the newest GPS fix in the mission log, so a replay with no
   * aircraft attached still shows the sweep. `droneFromLog` says which.
   */
  dronePos: [number, number] | null;
  droneFromLog: boolean;
  /** Breadcrumb trail behind the drone, [lng, lat][]. */
  trail: [number, number][];
  /** The run the relay is hosting — the pipeline it launched, and its output. */
  run: RunStatus | null;
  send: (event: string, payload?: unknown) => void;
}

const Ctx = createContext<MissionState | null>(null);

export const useMission = () => {
  const v = useContext(Ctx);
  if (!v) throw new Error("useMission must be used inside <MissionProvider>");
  return v;
};

export function MissionProvider({ children }: { children: ReactNode }) {
  const [connected, setConnected] = useState(false);
  const [producerConnected, setProducerConnected] = useState(false);
  const [mission, setMission] = useState<Mission | null>(null);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [gids, setGids] = useState<Gid[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [frames, setFrames] = useState<Record<string, Frame>>({});
  const [workers, setWorkers] = useState<Record<string, WorkerStatus>>({});
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [telemetryLive, setTelemetryLive] = useState(false);
  const [trail, setTrail] = useState<[number, number][]>([]);
  const [run, setRun] = useState<RunStatus | null>(null);

  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    const socket = io(`${RELAY_URL}/react`, {
      transports: ["websocket", "polling"],
      rejectUnauthorized: false,
    });
    socketRef.current = socket;

    socket.on("connect", () => setConnected(true));
    socket.on("disconnect", () => setConnected(false));

    // The relay pushes a snapshot on connect, so a reload or a late-joining
    // second screen comes up populated instead of waiting for the next event.
    socket.on("snapshot", (s: Snapshot) => {
      setMission(s.mission);
      setLogs(s.logs ?? []);
      setGids(s.gids ?? []);
      setStats(s.stats);
      setFrames(s.frames ?? {});
      setTelemetry(s.telemetry ?? null);
      setRun(s.run ?? null);
      setProducerConnected(s.pythonConnected);
    });

    socket.on("mission_start", (m: Mission) => {
      setMission(m);
      setLogs([]);
      setGids([]);
      setStats(null);
      // A new run is a new flight. Keeping the old breadcrumb would draw the
      // previous clip's path behind a drone now flying a different one.
      setTrail([]);
    });

    // The relay's own launcher state, not a producer's. It arrives on every
    // change, including each line of child output, so the dialog can show a
    // Python traceback rather than just an exit code.
    socket.on("run_status", (r: RunStatus) => setRun(r));

    socket.on("logs", (batch: LogLine[]) => {
      setLogs((prev) => {
        const next = prev.concat(batch);
        return next.length > MAX_LOGS ? next.slice(next.length - MAX_LOGS) : next;
      });
    });

    // A GID arrives once per buffer assigned to it, so the same id is re-sent
    // with a fuller record. Replace in place and keep the list ordered by id,
    // rather than appending and showing the same casualty several times.
    socket.on("gid", (g: Gid) => {
      setGids((prev) => {
        const i = prev.findIndex((x) => x.gid === g.gid);
        if (i === -1) return [...prev, g].sort((a, b) => a.gid - b.gid);
        const next = prev.slice();
        next[i] = g;
        return next;
      });
    });

    socket.on("stats", (s: Stats) => setStats(s));
    socket.on("frame", (f: Frame) => setFrames((prev) => ({ ...prev, [f.worker]: f })));
    // 4 Hz. Appending to the trail here rather than in the map component keeps
    // the path intact when the map unmounts (Raw Feeds taking the centre pane).
    socket.on("telemetry", (t: Telemetry) => {
      setTelemetry(t);
      setTelemetryLive(true);
      const lng = num(t.longitude);
      const lat = num(t.latitude);
      if (lng == null || lat == null) return;
      setTrail((prev) => {
        const last = prev[prev.length - 1];
        // Skip duplicate fixes — a stationary drone would otherwise fill the
        // whole trail buffer with one point and erase the flown path.
        if (last && last[0] === lng && last[1] === lat) return prev;
        const next = [...prev, [lng, lat] as [number, number]];
        return next.length > TRAIL_MAX ? next.slice(next.length - TRAIL_MAX) : next;
      });
    });

    socket.on("worker_status", (s: WorkerStatus) =>
      setWorkers((prev) => ({ ...prev, [s.worker]: s })));
    socket.on("producer_status", (s: { connected: boolean }) =>
      setProducerConnected(s.connected));

    return () => {
      socket.removeAllListeners();
      socket.disconnect();
    };
  }, []);

  // Staleness is a clock question, not an event one: the link going quiet
  // produces no message to listen for.
  useEffect(() => {
    const id = setInterval(() => {
      setTelemetryLive(
        telemetry != null && Date.now() - telemetry.ts * 1000 < TELEM_STALE_MS,
      );
    }, 1000);
    return () => clearInterval(id);
  }, [telemetry]);

  // Fallback drone position: the newest per-frame fix the pipeline logged.
  // Track points carry the frame number, so "newest" is the highest frame
  // across every GID, not the last one to arrive over the socket.
  const logPos = useMemo<[number, number] | null>(() => {
    let best: { frame: number; pos: [number, number] } | null = null;
    for (const g of gids) {
      for (const p of g.track ?? []) {
        const lng = num(p.lon);
        const lat = num(p.lat);
        if (lng == null || lat == null) continue;
        if (!best || p.frame > best.frame) best = { frame: p.frame, pos: [lng, lat] };
      }
    }
    return best?.pos ?? null;
  }, [gids]);

  const telemPos = useMemo<[number, number] | null>(() => {
    if (!telemetry) return null;
    const lng = num(telemetry.longitude);
    const lat = num(telemetry.latitude);
    return lng == null || lat == null ? null : [lng, lat];
  }, [telemetry]);

  const useLog = !telemetryLive || telemPos == null;
  const dronePos = useLog ? logPos : telemPos;

  const send = useCallback((event: string, payload?: unknown) => {
    socketRef.current?.emit(event, payload ?? {});
  }, []);

  const value = useMemo<MissionState>(
    () => ({
      connected, producerConnected, mission, logs, gids, stats, frames, workers,
      telemetry, telemetryLive, dronePos, droneFromLog: useLog, trail, run, send,
    }),
    [connected, producerConnected, mission, logs, gids, stats, frames, workers,
     telemetry, telemetryLive, dronePos, useLog, trail, run, send],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
