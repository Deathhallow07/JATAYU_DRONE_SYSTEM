import { artifactUrl } from "../config";
import { gidColor, theme } from "../theme";
import type { Gid } from "../types";

const Row = ({ k, v }: { k: string; v: unknown }) => (
  <>
    <dt>{k}</dt>
    <dd>{v === null || v === undefined || v === "" ? "—" : String(v)}</dd>
  </>
);

/**
 * Everything in gid_N/, in one panel: the identity record from metadata.txt,
 * the representative crop, and every crop the gallery kept.
 *
 * Crops are lazy <img> against the relay's artifact route — a GID can hold a
 * few hundred, and loading them all eagerly on open stalls the pane.
 */
export default function GidDetail({ gid, onClose, onRoute }: {
  gid: Gid; onClose: () => void; onRoute?: () => void;
}) {
  const color = gidColor(gid.gid);

  return (
    <div
      className="drawer-backdrop"
      onClick={onClose}
      role="presentation"
    >
      <aside
        className="drawer"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label={`Global ID ${gid.gid}`}
      >
        <header className="pane-head">
          <span className="dot" style={{ background: color }} />
          <span className="pane-title" style={{ color }}>Casualty {gid.gid}</span>
          <span
            className="pane-sub"
            title={gid.event === "merge"
              ? "This person has been re-identified after leaving and re-entering frame"
              : "First sighting"}
          >
            {gid.event === "merge" ? "seen again" : "first sighting"} · {gid.updates} update
            {gid.updates === 1 ? "" : "s"}
          </span>
          <span className="spacer" />
          {onRoute && (
            <button
              className="btn"
              onClick={onRoute}
              title="Plan a route to this casualty across the traversability grid"
            >
              Plan route here
            </button>
          )}
          <button className="btn" onClick={onClose} title="Close this casualty record (Esc)">
            Close
          </button>
        </header>

        <div className="pane-body">
          {gid.representative && (
            <div style={{ display: "flex", justifyContent: "center", padding: 12, background: theme.surface0 }}>
              <img
                src={artifactUrl(gid.representative, gid.updates)}
                alt={`Casualty ${gid.gid}, sharpest crop`}
                title="The sharpest crop of this person so far — replaced in place as better ones arrive"
                style={{ maxHeight: 260, borderRadius: 6, border: `1px solid ${theme.border}` }}
              />
            </div>
          )}

          <dl className="kv">
            <Row k="Created" v={gid.created} />
            <Row k="Mission time" v={gid.missionTimestamp} />
            <Row k="ROS time" v={gid.rosTime != null ? `${gid.rosTime} s` : null} />
            <Row k="Latitude" v={gid.lat} />
            <Row k="Longitude" v={gid.lon} />
            <Row k="GPS fixes" v={gid.gpsSamples} />
            <Row k="Gallery size" v={gid.gallerySize} />
            <Row k="Best confidence" v={gid.bestConf?.toFixed(4)} />
            <Row k="Best sharpness" v={gid.bestSharpness?.toFixed(2)} />
            <Row k="Frame span" v={`${gid.firstFrame ?? "?"} → ${gid.lastFrame ?? "?"}`} />
            <Row k="Last track id" v={gid.tid} />
            <Row k="Buffers merged" v={gid.buffers.length} />
          </dl>

          <div style={{ padding: "4px 12px" }}>
            <span className="pane-title" style={{ color: theme.textDim }}>
              Buffer history
            </span>
          </div>
          <div style={{ padding: "0 12px 8px", fontFamily: theme.fontMono, fontSize: "0.65rem", color: theme.textDim }}>
            {gid.buffers.map((b, i) => (
              <div key={i} style={{ display: "flex", gap: 10, padding: "3px 0", borderBottom: `1px solid ${theme.border}` }}>
                <span style={{ color: b.event === "merge" ? theme.warning : theme.success, width: 52 }}>
                  {b.event}
                </span>
                <span>TID {b.tid ?? "—"}</span>
                <span>{b.entryCount} entries</span>
                <span style={{ color: theme.textFaint }}>{String(b.mission_timestamp ?? "")}</span>
              </div>
            ))}
          </div>

          <div style={{ padding: "4px 12px" }}>
            <span className="pane-title" style={{ color: theme.textDim }}>
              Gallery crops ({gid.crops.length})
            </span>
          </div>
          <div className="crop-grid">
            {/* Not versioned, unlike the representative: each crop is written
                once under its own frame number and never rewritten, so a
                cache-buster here would refetch the whole gallery on every
                update to the record. */}
            {gid.crops.map((c) => (
              <img key={c} src={artifactUrl(c)} alt="" loading="lazy" title={c.split("/").pop()} />
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}
