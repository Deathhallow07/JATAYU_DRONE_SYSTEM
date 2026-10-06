import { useEffect, useRef, useState } from "react";
import mapboxgl, { Map as MapboxMap, Marker } from "mapbox-gl";
import type { GeoJSONSource } from "mapbox-gl";
import type { Feature } from "geojson";
import { useMission } from "../context/MissionContext";
import { DEFAULT_CENTER, MAPBOX_TOKEN } from "../config";
import { gidColor, theme } from "../theme";
import "mapbox-gl/dist/mapbox-gl.css";

mapboxgl.accessToken = MAPBOX_TOKEN;

const TRAIL_SRC = "uav-trail";
const TRAIL_CASING = "uav-trail-casing";

/**
 * Camera state, kept OUTSIDE the component on purpose.
 *
 * Swapping a raw feed into the main panel moves this map to a different parent
 * in the tree, which React implements as an unmount and a fresh mount — a new
 * mapboxgl.Map, back at the default centre and zoom. Everything else the map
 * draws is rebuilt from context (the trail, the pins, the drone), but the
 * camera and FOLLOW are the operator's own state and belong to the session,
 * not to one mounting of the component. Losing the zoom someone had set every
 * time they looked at a video feed is the kind of thing that makes a console
 * feel broken.
 */
const camera: { center: [number, number] | null; zoom: number; follow: boolean } = {
  center: null,
  zoom: 17,
  follow: true,
};

const hexToRgba = (hex: string, alpha: number) => {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
};

/** Drone icon, rotated to heading. Same airframe silhouette as the existing GCS. */
function makeDroneEl() {
  const el = document.createElement("div");
  el.style.width = "46px";
  el.style.height = "46px";
  el.style.backgroundImage = "url('https://iili.io/FSIe3iB.png')";
  el.style.backgroundSize = "contain";
  el.style.backgroundPosition = "center";
  el.style.backgroundRepeat = "no-repeat";
  el.style.filter = `drop-shadow(0 0 6px ${theme.accent})`;
  el.style.transition = "transform 0.2s linear";
  return el;
}

/**
 * Casualty pin: a numbered medical cross in the GID's identity colour, so a
 * pin on the map and a card in the left bar are obviously the same person.
 */
function makeCasualtyEl(gid: number, onClick: () => void) {
  const color = gidColor(gid);
  const el = document.createElement("div");
  el.style.cursor = "pointer";
  el.style.width = "34px";
  el.style.height = "44px";
  el.style.display = "flex";
  el.style.flexDirection = "column";
  el.style.alignItems = "center";
  el.title = `Casualty ${gid}`;
  el.innerHTML = `
    <svg viewBox="0 0 24 24" width="30" height="30" xmlns="http://www.w3.org/2000/svg">
      <circle cx="12" cy="12" r="10" fill="${color}" stroke="#000" stroke-width="1.5"/>
      <rect x="6" y="10.5" width="12" height="3" fill="#0a0a0a" rx="1"/>
      <rect x="10.5" y="6" width="3" height="12" fill="#0a0a0a" rx="1"/>
    </svg>
    <span style="
      font: 700 10px/1.2 ui-monospace, monospace; color: ${color};
      background: rgba(0,0,0,0.78); padding: 1px 5px; border-radius: 6px;
      margin-top: -3px; white-space: nowrap;">${gid}</span>`;
  el.addEventListener("click", (e) => {
    e.stopPropagation();
    onClick();
  });
  return el;
}

/**
 * The mission map: where the drone is, where it has been, and every casualty
 * it has located.
 *
 * Casualty pins come from each GID's final GPS — the medoid of that person's
 * fixes, which is the pipeline's own best estimate of where they are lying.
 * GIDs with no fix (the pipeline logs "Synthesised" coordinates, or geolocation
 * failed) are simply not pinned rather than pinned at 0,0.
 */
export default function MissionMap({ onSelectGid }: { onSelectGid?: (gid: number) => void }) {
  const { gids, dronePos, droneFromLog, trail, telemetry } = useMission();

  const container = useRef<HTMLDivElement | null>(null);
  const map = useRef<MapboxMap | null>(null);
  const ready = useRef(false);

  const droneMarker = useRef<Marker | null>(null);
  const lastTrailDraw = useRef(0);
  const casualtyMarkers = useRef<Map<number, Marker>>(new Map());

  // Follow is the default: on a wall display nobody is panning the map, and a
  // drone that flies off the edge unnoticed is worse than a map that moves.
  // Any manual drag turns it off, so an operator who reaches for the map keeps
  // control until they press FOLLOW again.
  const [follow, setFollow] = useState(camera.follow);
  const followRef = useRef(camera.follow);
  const [styleKey, setStyleKey] = useState<"satellite" | "dark">("satellite");
  const [hasCentred, setHasCentred] = useState(camera.center != null);

  useEffect(() => { followRef.current = follow; camera.follow = follow; }, [follow]);

  // ── Map instance ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (map.current || !container.current) return;

    const m = new mapboxgl.Map({
      container: container.current,
      style: styleKey === "satellite"
        ? "mapbox://styles/mapbox/satellite-streets-v12"
        : "mapbox://styles/mapbox/dark-v11",
      center: camera.center ?? DEFAULT_CENTER,
      zoom: camera.zoom,
      attributionControl: false,
    });
    map.current = m;

    // Remembered for the next mount — see `camera` above.
    const remember = () => {
      const c = m.getCenter();
      camera.center = [c.lng, c.lat];
      camera.zoom = m.getZoom();
    };
    m.on("moveend", remember);

    m.addControl(new mapboxgl.NavigationControl({ visualizePitch: true }), "top-right");
    m.addControl(new mapboxgl.ScaleControl({ maxWidth: 110, unit: "metric" }), "bottom-left");

    m.on("load", () => { ready.current = true; });

    // A drag is the operator taking over; a programmatic easeTo is not, so
    // this listens for the user-originated event only.
    m.on("dragstart", () => setFollow(false));

    // The Raw Feeds pane opening resizes the centre column under the map.
    const ro = new ResizeObserver(() => m.resize());
    ro.observe(container.current);

    const markers = casualtyMarkers.current;
    return () => {
      ro.disconnect();
      m.remove();
      map.current = null;
      ready.current = false;
      droneMarker.current = null;
      markers.clear();
    };
    // styleKey is handled by setStyle below, not by rebuilding the map.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Basemap switch ───────────────────────────────────────────────────────
  // setStyle drops every source and layer we added, so the trail is rebuilt on
  // the next telemetry effect — which is why ready is lowered here.
  useEffect(() => {
    const m = map.current;
    if (!m) return;
    ready.current = false;
    m.setStyle(styleKey === "satellite"
      ? "mapbox://styles/mapbox/satellite-streets-v12"
      : "mapbox://styles/mapbox/dark-v11");
    const onIdle = () => { ready.current = true; };
    m.once("styledata", onIdle);
  }, [styleKey]);

  // ── Drone marker + trail ─────────────────────────────────────────────────
  useEffect(() => {
    const m = map.current;
    if (!m || !dronePos) return;

    if (!droneMarker.current) {
      droneMarker.current = new mapboxgl.Marker({ element: makeDroneEl() })
        .setLngLat(dronePos)
        .addTo(m);
    } else {
      droneMarker.current.setLngLat(dronePos);
    }

    // setRotation only — writing the element's transform directly would fight
    // Marker's own positioning transform and compound on every sample.
    const heading = telemetry?.heading;
    if (typeof heading === "number" && Number.isFinite(heading)) {
      droneMarker.current.setRotation(heading);
    }

    // First fix of the mission: jump rather than glide, so the map does not
    // animate across the planet from the fallback centre.
    if (!hasCentred) {
      m.jumpTo({ center: dronePos, zoom: 18 });
      setHasCentred(true);
    } else if (followRef.current) {
      // Duration matched to the 4 Hz sample rate. A 400 ms glide started every
      // 250 ms is a camera animation that is always being interrupted a third
      // of the way through by the next one, which reads as the map stuttering
      // rather than following.
      m.easeTo({ center: dronePos, duration: 240, essential: true });
    }
  }, [dronePos, telemetry, hasCentred]);

  // ── Traversed path ───────────────────────────────────────────────────────
  //
  // Drawn as two layers: a dark casing under a bright line, so the track stays
  // readable over pale satellite imagery as well as over the dark basemap. The
  // gradient fades toward the OLDEST point rather than disappearing at it — the
  // whole flown path has to stay visible, because "where has it already looked"
  // is the question the map is there to answer, and the fade is only there to
  // show which end is now.
  useEffect(() => {
    const m = map.current;
    if (!m || trail.length < 2) return;

    const feature: Feature = {
      type: "Feature",
      properties: {},
      geometry: { type: "LineString", coordinates: trail },
    };

    const draw = () => {
      // setStyle() discards every source and layer we added, so after a
      // basemap switch getSource returns undefined and these are re-added.
      // Guarding on our own `ready` ref alone was not enough: the effect also
      // runs on the styleKey change itself, while the new style is still
      // loading, and the layer would then not be re-added until the next fix
      // arrived — which on a finished run is never.
      const src = m.getSource(TRAIL_SRC) as GeoJSONSource | undefined;
      if (src) {
        src.setData(feature);
        return;
      }
      m.addSource(TRAIL_SRC, { type: "geojson", data: feature, lineMetrics: true });
      m.addLayer({
        id: TRAIL_CASING,
        type: "line",
        source: TRAIL_SRC,
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-width": 6, "line-color": "#000000", "line-opacity": 0.45 },
      });
      m.addLayer({
        id: TRAIL_SRC,
        type: "line",
        source: TRAIL_SRC,
        layout: { "line-cap": "round", "line-join": "round" },
        paint: {
          "line-width": 3,
          "line-gradient": [
            "interpolate", ["linear"], ["line-progress"],
            0, hexToRgba(theme.accent, 0.35),
            1, hexToRgba(theme.accent, 1),
          ],
        },
      });
    };

    // Throttled, because the cost here scales with the WHOLE path, not with
    // the one point that just arrived: setData re-serialises every coordinate
    // in the LineString, and the trail holds up to 12000 of them by the end of
    // a long clip. Doing that on all four samples a second is most of a frame
    // budget spent redrawing a line that moved by a metre.
    //
    // A fix arrives every 250 ms, so at 500 ms the trail is at most one sample
    // behind the drone marker — which is itself moving smoothly, and is what
    // the eye actually tracks.
    const now = Date.now();
    const overdue = now - lastTrailDraw.current > 500;
    if (!m.isStyleLoaded()) {
      m.once("idle", () => { lastTrailDraw.current = Date.now(); draw(); });
      return;
    }
    if (!overdue) {
      const id = setTimeout(() => {
        lastTrailDraw.current = Date.now();
        if (map.current?.isStyleLoaded()) draw();
      }, 500 - (now - lastTrailDraw.current));
      return () => clearTimeout(id);
    }
    lastTrailDraw.current = now;
    draw();
  }, [trail, styleKey]);

  // ── Casualty pins ────────────────────────────────────────────────────────
  useEffect(() => {
    const m = map.current;
    if (!m) return;

    const live = new Set<number>();

    for (const g of gids) {
      const lat = Number(g.lat);
      const lon = Number(g.lon);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
      live.add(g.gid);

      const existing = casualtyMarkers.current.get(g.gid);
      if (existing) {
        // A merge refines the medoid, so the pin moves as more fixes arrive.
        existing.setLngLat([lon, lat]);
        continue;
      }
      const marker = new mapboxgl.Marker({
        element: makeCasualtyEl(g.gid, () => onSelectGid?.(g.gid)),
        anchor: "bottom",
      })
        .setLngLat([lon, lat])
        .addTo(m);
      casualtyMarkers.current.set(g.gid, marker);
    }

    // A GID only disappears when the run is replaced, but leaking markers
    // across runs would leave the previous mission's casualties on the map.
    for (const [gid, marker] of casualtyMarkers.current) {
      if (!live.has(gid)) {
        marker.remove();
        casualtyMarkers.current.delete(gid);
      }
    }
  }, [gids, onSelectGid, styleKey]);

  const pinned = gids.filter((g) => Number.isFinite(Number(g.lat))).length;

  return (
    <div className="map-wrap">
      <div ref={container} className="map-canvas" />

      <div className="map-controls">
        <button
          className={`btn${follow ? " active" : ""}`}
          onClick={() => {
            setFollow(true);
            if (dronePos) map.current?.easeTo({ center: dronePos, zoom: 18, duration: 600 });
          }}
          title={follow
            ? "The map is following the drone. Drag the map to take over."
            : "Re-centre on the drone and keep it centred as it flies"}
        >
          {follow ? "Following drone" : "Follow drone"}
        </button>
        {/* Labelled with the basemap it switches TO. Labelling it with the
            current one reads as a state chip and nobody presses it. */}
        <button
          className="btn"
          onClick={() => setStyleKey((k) => (k === "satellite" ? "dark" : "satellite"))}
          title={styleKey === "satellite"
            ? "Switch to the plain dark basemap"
            : "Switch to satellite imagery"}
        >
          {styleKey === "satellite" ? "Dark basemap" : "Satellite view"}
        </button>
        <button
          className="btn"
          disabled={pinned === 0}
          onClick={() => {
            const m = map.current;
            if (!m) return;
            const b = new mapboxgl.LngLatBounds();
            let n = 0;
            for (const g of gids) {
              const lat = Number(g.lat);
              const lon = Number(g.lon);
              if (Number.isFinite(lat) && Number.isFinite(lon)) { b.extend([lon, lat]); n++; }
            }
            if (dronePos) b.extend(dronePos);
            if (n > 0) {
              setFollow(false);
              m.fitBounds(b, { padding: 90, maxZoom: 19, duration: 700 });
            }
          }}
          title={pinned === 0
            ? "No casualties have been located yet"
            : "Zoom out until every casualty pin and the drone are on screen"}
        >
          Fit all pins
        </button>
      </div>

      <div className="map-legend">
        <span title="The aircraft, rotated to its heading">
          <span className="dot" style={{ background: theme.accent }} /> Drone
        </span>
        <span title="The path the drone has flown so far, brightest at the current position">
          <span className="trail-key" /> Path flown
        </span>
        <span>
          <span className="dot" style={{ background: theme.danger }} />
          {pinned} {pinned === 1 ? "casualty" : "casualties"}
        </span>
        {droneFromLog && (
          <span style={{ color: theme.warning }} title="No MAVLink link — the drone position is the newest GPS fix in the mission log">
            TRACK FROM LOG
          </span>
        )}
      </div>

      {!dronePos && (
        <div className="map-waiting">
          Waiting for a position fix…
          <span>Start a run with a telemetry CSV to fly the recorded path.</span>
        </div>
      )}
    </div>
  );
}
