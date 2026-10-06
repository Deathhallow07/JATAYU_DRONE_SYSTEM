import { useEffect, useMemo, useRef, useState } from "react";
import Pane from "./Pane";
import { useMission } from "../context/MissionContext";
import { artifactUrl } from "../config";
import { gidColor, theme } from "../theme";
import { fmtMeters, haversineMeters, parseLatLon } from "../geo";
import type { Gid } from "../types";

const FRESH_MS = 1400;

const fmtCoord = (v: number | null) =>
  v == null || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(6);

function GidCard({ gid, fresh, selected, onSelect, distance, best }: {
  gid: Gid; fresh: boolean; selected: boolean; onSelect: () => void;
  /** Metres from the coordinate being searched for, when one is entered. */
  distance?: number | null;
  /** True for the single closest casualty to that coordinate. */
  best?: boolean;
}) {
  const color = gidColor(gid.gid);
  // Versioned by the record's own revision counter: the pipeline overwrites
  // representative.jpg in place with a sharper crop as it sees more of this
  // person, so an unversioned URL would pin the card to the first, blurriest
  // photo for the whole run. See artifactUrl().
  const rep = artifactUrl(gid.representative, gid.updates);

  return (
    <button
      className={`gid-card${fresh ? " fresh" : ""}${selected ? " selected" : ""}${best ? " match" : ""}`}
      style={{ borderLeftColor: color }}
      onClick={onSelect}
      title={`Open casualty ${gid.gid}: every photo, the identity record and the route button`}
    >
      {rep ? (
        <img
          className="gid-thumb"
          src={rep}
          alt={`Casualty ${gid.gid}, best crop`}
          title={`Sharpest crop of casualty ${gid.gid} so far · ${gid.gallerySize} photos`}
          loading="lazy"
        />
      ) : (
        <div className="gid-thumb empty">no crop</div>
      )}

      <div className="gid-meta">
        <div className="row">
          <span style={{ color, fontWeight: 700, fontSize: "0.78rem" }}>GID {gid.gid}</span>
          <span
            className="chip"
            style={{
              color: gid.event === "merge" ? theme.warning : theme.success,
              background: "transparent",
              padding: 0,
            }}
            title={gid.event === "merge"
              ? "Re-identified: this person was matched to a casualty already found"
              : "First sighting of this casualty"}
          >
            {gid.event === "merge" ? "SEEN AGAIN" : "NEW"}
          </span>
        </div>
        <div className="row">
          <span className="k">GPS</span>
          <span className="v">{fmtCoord(gid.lat)}, {fmtCoord(gid.lon)}</span>
        </div>
        {distance !== undefined && (
          <div className="row">
            <span className="k" title="Great-circle distance from this casualty's fix to the coordinate you entered">
              {best ? "Δ GT · closest" : "Δ GT"}
            </span>
            <span
              className="v"
              style={{ color: best ? theme.success : theme.textDim, fontWeight: best ? 700 : 400 }}
            >
              {distance == null ? "no fix" : fmtMeters(distance)}
            </span>
          </div>
        )}
        <div className="row">
          <span className="k" title="Photos of this person, and GPS fixes used to place the pin">Gallery</span>
          <span className="v">
            {gid.gallerySize} · {gid.gpsSamples ?? 0} fixes
          </span>
        </div>
        <div className="row">
          <span className="k" title="Best detection confidence and image sharpness across the gallery">Conf / Sharp</span>
          <span className="v">
            {gid.bestConf != null ? gid.bestConf.toFixed(3) : "—"} ·{" "}
            {gid.bestSharpness != null ? gid.bestSharpness.toFixed(0) : "—"}
          </span>
        </div>
        <div className="row">
          <span className="k" title="Time into the mission when this casualty was recorded">T+</span>
          <span className="v">{gid.missionTimestamp ?? "—"}</span>
        </div>
      </div>
    </button>
  );
}

/**
 * The casualty list — the mission's index, and the one pane that is always on
 * screen.
 *
 * Selection is owned by App rather than by this pane: the same casualty can be
 * opened from a pin on the map or a line in the notification feed, and all
 * three must drive one detail drawer instead of three competing ones.
 */
export default function GidPane({ area, selected, onSelect }: {
  area: string;
  selected?: number | null;
  onSelect?: (gid: number) => void;
}) {
  const { gids } = useMission();
  const [fresh, setFresh] = useState<Set<number>>(new Set());

  // A ground-truth coordinate to check the pipeline against: paste the known
  // position of a casualty and the list says which GID it came out as, and by
  // how far it missed. Reading every card's lat/lon by eye was the alternative.
  const [probeText, setProbeText] = useState("");
  const probe = useMemo(() => parseLatLon(probeText), [probeText]);

  // Ranked by distance from the probe, nearest first; casualties without a fix
  // yet sort last rather than disappearing, because an unplaced casualty is
  // still one the operator has to account for.
  const ranked = useMemo(() => {
    if (!probe) return gids.map((gid) => ({ gid, distance: undefined as number | undefined }));
    const withD = gids.map((gid) => ({
      gid,
      distance: gid.lat == null || gid.lon == null
        ? null
        : haversineMeters(probe.lat, probe.lon, Number(gid.lat), Number(gid.lon)),
    }));
    return withD.sort((a, b) => (a.distance ?? Infinity) - (b.distance ?? Infinity));
  }, [gids, probe]);

  const best = probe && ranked.length && ranked[0].distance != null ? ranked[0] : null;

  // Flash a card when its record changes. Keyed on update count, not on the id,
  // so a GID that grows by a merge flashes again — a re-identification is as
  // much news as a first sighting.
  const seen = useRef(new Map<number, number>());
  useEffect(() => {
    const newly: number[] = [];
    for (const g of gids) {
      if (seen.current.get(g.gid) !== g.updates) {
        seen.current.set(g.gid, g.updates);
        newly.push(g.gid);
      }
    }
    if (newly.length === 0) return;

    setFresh((prev) => new Set([...prev, ...newly]));
    const t = setTimeout(() => {
      setFresh((prev) => {
        const next = new Set(prev);
        newly.forEach((g) => next.delete(g));
        return next;
      });
    }, FRESH_MS);
    return () => clearTimeout(t);
  }, [gids]);

  return (
    <Pane
      area={area}
      title="Casualties"
      subtitle={probe ? `${gids.length} found · by distance` : `${gids.length} found`}
      column
      status={{
        color: gids.length ? theme.success : theme.textFaint,
        label: gids.length ? "Casualties located" : "None located yet",
      }}
    >
      <div className="gid-search fixed">
        <input
          className="btn coord-input"
          placeholder="ground truth lat, lon"
          title="Enter a known position — e.g. 28.613939, 77.209023 — to find the casualty closest to it, and how far the pipeline's fix landed from it"
          value={probeText}
          onChange={(e) => setProbeText(e.target.value)}
          spellCheck={false}
          inputMode="decimal"
        />
        {probeText && (
          <button
            className="btn"
            onClick={() => setProbeText("")}
            title="Clear the coordinate and put the list back in the order casualties were found"
          >
            Clear
          </button>
        )}

        {probeText.trim() !== "" && !probe && (
          <div className="gid-search-note" style={{ color: theme.warning }}>
            Not a coordinate — give decimal degrees as “lat, lon”.
          </div>
        )}

        {probe && (
          <div className="gid-search-note">
            {best ? (
              <button
                className="btn match-hit"
                onClick={() => onSelect?.(best.gid.gid)}
                title={`Open casualty ${best.gid.gid} — the closest fix to the coordinate entered`}
              >
                <span style={{ color: gidColor(best.gid.gid) }}>GID {best.gid.gid}</span>
                <span style={{ color: theme.success }}>{fmtMeters(best.distance as number)}</span>
              </button>
            ) : (
              <span style={{ color: theme.textFaint }}>
                {gids.length === 0 ? "No casualties to match yet" : "No casualty has a GPS fix yet"}
              </span>
            )}
            <span className="probe-echo" title="The coordinate being measured from">
              from {probe.lat.toFixed(6)}, {probe.lon.toFixed(6)}
            </span>
          </div>
        )}
      </div>

      {gids.length === 0 ? (
        <div className="empty-state grow">
          No casualties found yet.
          <br />
          They appear here the moment the drone locates one.
        </div>
      ) : (
        <div className="gid-list grow" style={{ overflowY: "auto" }}>
          {ranked.map(({ gid: g, distance }) => (
            <GidCard
              key={g.gid}
              gid={g}
              fresh={fresh.has(g.gid)}
              selected={selected === g.gid}
              onSelect={() => onSelect?.(g.gid)}
              distance={distance}
              best={best?.gid.gid === g.gid}
            />
          ))}
        </div>
      )}
    </Pane>
  );
}
