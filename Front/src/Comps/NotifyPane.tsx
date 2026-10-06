import { useEffect, useRef, useState } from "react";
import Pane from "./Pane";
import { useMission } from "../context/MissionContext";
import { artifactUrl } from "../config";
import { gidColor, theme } from "../theme";
import type { LogLine, Notice } from "../types";

const MAX_NOTICES = 200;

const KIND_LABEL: Record<Notice["kind"], string> = {
  found: "FOUND",
  again: "SEEN AGAIN",
  mission: "MISSION",
  problem: "PROBLEM",
};

const KIND_COLOR: Record<Notice["kind"], string> = {
  found: theme.success,
  again: theme.accent,
  mission: theme.textDim,
  problem: theme.danger,
};

const clock = (ts: number) =>
  new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

const coords = (lat?: number | null, lon?: number | null) =>
  lat != null && lon != null && Number.isFinite(lat) && Number.isFinite(lon)
    ? `${Number(lat).toFixed(5)}, ${Number(lon).toFixed(5)}`
    : null;

/**
 * Which raw log lines a medic should ever see.
 *
 * The pipeline's log is an engineering instrument — DET/TRACK/REID/GPS fire
 * hundreds of times a minute and describe internal state, not anything anyone
 * can act on. Everything here is either a mission bookend or a fault someone
 * has to do something about. Anything unrecognised is dropped, deliberately:
 * the raw log is still one click away in the drawer, and a notification panel
 * that degrades into the log it replaced is no use to the person it is for.
 */
function noticeFromLog(line: LogLine): Notice | null {
  const msg = line.msg.trim();
  const level = line.level.toUpperCase();
  const at = line.ts * 1000;
  const id = `log-${line.seq}`;

  // Bookends. These are the only INFO lines that matter to an operator.
  if (/^=+$/.test(msg)) return null;
  if (/MISSION COMPLETE/i.test(msg))
    return { id, kind: "mission", text: "Search complete", ts: at };
  if (/REPLAY COMPLETE/i.test(msg))
    return { id, kind: "mission", text: "Replay finished", ts: at };
  if (/SEARCH & RESCUE PIPELINE/i.test(msg))
    return { id, kind: "mission", text: "Search started", ts: at };
  if (/Video source:/i.test(msg))
    return { id, kind: "mission", text: "Camera feed connected", detail: msg, ts: at };
  if (/Telemetry source:/i.test(msg))
    return { id, kind: "mission", text: "Telemetry connected", detail: msg, ts: at };

  // Faults. ERROR is always actionable; WARN almost never is — "blur reject"
  // is the detector working correctly and fires on a large share of frames.
  if (level === "ERROR")
    return { id, kind: "problem", text: msg, ts: at };
  if (level === "WARN" && /(lost|fail|timeout|disconnect|no gps|unable|cannot)/i.test(msg))
    return { id, kind: "problem", text: msg, ts: at };

  return null;
}

/** Newest first, capped, and stable — the list is a feed, not a scrollback. */
const prepend = (prev: Notice[], added: Notice[]) => {
  if (added.length === 0) return prev;
  const next = [...added.reverse(), ...prev];
  return next.length > MAX_NOTICES ? next.slice(0, MAX_NOTICES) : next;
};

export default function NotifyPane({ area, onSelectGid }: {
  area: string; onSelectGid?: (gid: number) => void;
}) {
  const { logs, gids, connected, producerConnected, mission } = useMission();
  const [notices, setNotices] = useState<Notice[]>([]);
  const [muteProblems, setMuteProblems] = useState(false);

  // ── Casualties ───────────────────────────────────────────────────────────
  // A GID is re-sent every time a buffer is assigned to it, so the same id
  // arrives many times. `updates` is the pipeline's own revision counter:
  // first sight is news, and so is each later merge — that re-identification
  // is the whole point of the system and the one thing worth interrupting a
  // medic for. Intermediate re-sends that change nothing are not.
  const seen = useRef(new Map<number, number>());
  useEffect(() => {
    const added: Notice[] = [];
    for (const g of gids) {
      const prev = seen.current.get(g.gid);
      if (prev === g.updates) continue;
      seen.current.set(g.gid, g.updates);

      const where = coords(g.lat, g.lon);
      const first = prev === undefined;

      // A brand-new GID whose event is already "merge" is still a first
      // sighting as far as this console is concerned — we never showed it.
      if (first) {
        added.push({
          id: `gid-${g.gid}-new`,
          kind: "found",
          text: `Casualty ${g.gid} found`,
          detail: where ? `${where} · ${g.gallerySize} photos` : `${g.gallerySize} photos`,
          gid: g.gid,
          lat: g.lat,
          lon: g.lon,
          ts: Date.now(),
        });
      } else {
        added.push({
          id: `gid-${g.gid}-u${g.updates}`,
          kind: "again",
          text: `Casualty ${g.gid} seen again`,
          detail: where
            ? `${where} · now ${g.gallerySize} photos`
            : `now ${g.gallerySize} photos`,
          gid: g.gid,
          lat: g.lat,
          lon: g.lon,
          ts: Date.now(),
        });
      }
    }
    if (added.length) setNotices((prev) => prepend(prev, added));
  }, [gids]);

  // ── Mission bookends and faults ──────────────────────────────────────────
  const lastSeq = useRef(0);
  useEffect(() => {
    const fresh = logs.filter((l) => l.seq > lastSeq.current);
    if (fresh.length === 0) return;
    lastSeq.current = fresh[fresh.length - 1].seq;

    const added = fresh.map(noticeFromLog).filter((n): n is Notice => n !== null);
    if (added.length) setNotices((prev) => prepend(prev, added));
  }, [logs]);

  // ── Link faults ──────────────────────────────────────────────────────────
  // These have no log line to key off: the thing that would have written one
  // is the thing that went away.
  const linkState = useRef({ relay: true, pipeline: true });
  useEffect(() => {
    const added: Notice[] = [];
    const now = Date.now();
    if (!connected && linkState.current.relay)
      added.push({ id: `relay-${now}`, kind: "problem", text: "Console lost the relay — display may be out of date", ts: now });
    if (connected && !linkState.current.relay)
      added.push({ id: `relay-ok-${now}`, kind: "mission", text: "Console reconnected", ts: now });
    if (!producerConnected && linkState.current.pipeline && connected)
      added.push({ id: `pipe-${now}`, kind: "problem", text: "Search pipeline stopped sending", ts: now });
    if (producerConnected && !linkState.current.pipeline)
      added.push({ id: `pipe-ok-${now}`, kind: "mission", text: "Search pipeline running", ts: now });

    linkState.current = { relay: connected, pipeline: producerConnected };
    if (added.length) setNotices((prev) => prepend(prev, added));
  }, [connected, producerConnected]);

  // A new run clears the feed along with everything else on screen.
  useEffect(() => {
    setNotices([]);
    seen.current.clear();
    lastSeq.current = 0;
  }, [mission?.runFolder, mission?.startedAt]);

  const shown = muteProblems ? notices.filter((n) => n.kind !== "problem") : notices;
  const found = notices.filter((n) => n.kind === "found").length;

  return (
    <Pane
      area={area}
      title="Notifications"
      subtitle={found ? `${found} located` : "standing by"}
      actions={
        <button
          className={`btn${muteProblems ? " active" : ""}`}
          onClick={() => setMuteProblems((v) => !v)}
          title={muteProblems
            ? "Currently hiding system problems. Click to show them again."
            : "Showing everything. Click to hide system problems and leave only casualties."}
        >
          {muteProblems ? "Casualties only" : "Showing all"}
        </button>
      }
    >
      {shown.length === 0 ? (
        <div className="empty-state">
          Nothing to report yet.
          <br />
          Casualties appear here as the drone finds them.
        </div>
      ) : (
        <div className="notice-list">
          {shown.map((n) => {
            const gidRec = n.gid != null ? gids.find((g) => g.gid === n.gid) : undefined;
            // Versioned, so a "seen again" notice shows the photo as it is
            // now rather than the one cached from the first sighting.
            const thumb = artifactUrl(gidRec?.representative, gidRec?.updates);
            const color = n.gid != null ? gidColor(n.gid) : KIND_COLOR[n.kind];
            const clickable = n.gid != null && onSelectGid;

            return (
              <div
                key={n.id}
                className={`notice notice-${n.kind}${clickable ? " clickable" : ""}`}
                style={{ borderLeftColor: color }}
                onClick={clickable ? () => onSelectGid(n.gid as number) : undefined}
                title={clickable ? `Open casualty ${n.gid}` : undefined}
                role={clickable ? "button" : undefined}
                tabIndex={clickable ? 0 : undefined}
                onKeyDown={clickable ? (e) => {
                  if (e.key === "Enter" || e.key === " ") onSelectGid(n.gid as number);
                } : undefined}
              >
                {thumb ? (
                  <img
                    className="notice-thumb"
                    src={thumb}
                    alt=""
                    title={`Casualty ${n.gid}`}
                    loading="lazy"
                  />
                ) : (
                  <span className="notice-tag" style={{ color: KIND_COLOR[n.kind] }}>
                    {KIND_LABEL[n.kind]}
                  </span>
                )}
                <div className="notice-body">
                  <div className="notice-text" style={{ color: n.kind === "problem" ? theme.danger : theme.text }}>
                    {n.text}
                  </div>
                  {n.detail && <div className="notice-detail">{n.detail}</div>}
                </div>
                <time className="notice-time">{clock(n.ts)}</time>
              </div>
            );
          })}
        </div>
      )}
    </Pane>
  );
}
