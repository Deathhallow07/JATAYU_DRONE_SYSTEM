import { useCallback, useEffect, useState } from "react";
import GidPane from "./Comps/GidPane";
import GidDetail from "./Comps/GidDetail";
import MissionMap from "./Comps/MissionMap";
import SystemPane from "./Comps/SystemPane";
import NotifyPane from "./Comps/NotifyPane";
import RawFeeds, { FeedPane, FEED_LABEL, FEED_HINT } from "./Comps/RawFeeds";
import type { FeedKey } from "./Comps/RawFeeds";
import LogDrawer from "./Comps/LogDrawer";
import StatusBar from "./Comps/StatusBar";
import LaunchDialog from "./Comps/LaunchDialog";
import { useMission } from "./context/MissionContext";
import { theme } from "./theme";

const ALL_FEEDS: FeedKey[] = ["pipeline", "seg", "depth"];

export default function App() {
  const { connected, mission, gids, run, send } = useMission();

  // Raw feeds are off by default: the dashboard is for the medic, and each
  // feed costs a video decode while it is mounted.
  const [rawOpen, setRawOpen] = useState(false);
  // Only the geolocalisation feed is on to begin with. It is the one that
  // answers "what did the pipeline actually see?"; traversability and depth
  // are diagnostics for when that answer looks wrong, and each is another
  // video decode. Both stay one click away in the topbar toggles.
  const [feeds, setFeeds] = useState<FeedKey[]>(["pipeline"]);

  // Which feed, if any, has traded places with the mission map. The map is
  // the dashboard's centre of gravity, so this is deliberately one at a time
  // and always reversible from the same button that did it.
  const [mainFeed, setMainFeed] = useState<FeedKey | null>(null);

  const [logOpen, setLogOpen] = useState(false);
  const [launchOpen, setLaunchOpen] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);

  // Clicking a pin on the map or a line in the notification feed opens that
  // casualty's full record — the same drawer the GID list opens, so there is
  // one place a casualty is ever looked at in detail.
  const selectGid = useCallback((gid: number) => setSelected(gid), []);

  // Promoting a feed implies opening the strip: the map has to go somewhere,
  // and it goes into the slot the feed just left.
  const expandFeed = useCallback((feed: FeedKey) => {
    setRawOpen(true);
    setFeeds((prev) => (prev.includes(feed) ? prev : ALL_FEEDS.filter((f) => f === feed || prev.includes(f))));
    setMainFeed(feed);
  }, []);

  const restoreMap = useCallback(() => setMainFeed(null), []);

  // Closing the strip has to put the map back first, or the promoted feed
  // would take the map's slot with nowhere for the map to be.
  const closeRaw = useCallback(() => {
    setMainFeed(null);
    setRawOpen(false);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.isContentEditable)) return;

      if (e.key === "Escape") {
        if (launchOpen) setLaunchOpen(false);
        else if (selected != null) setSelected(null);
        else if (logOpen) setLogOpen(false);
        else if (mainFeed) restoreMap();
        else if (rawOpen) setRawOpen(false);
        return;
      }
      if (launchOpen) return;   // those keys are typing while the dialog is up

      if (e.key.toLowerCase() === "r") (rawOpen ? closeRaw() : setRawOpen(true));
      if (e.key.toLowerCase() === "l") setLogOpen((v) => !v);
      if (e.key.toLowerCase() === "n") setLaunchOpen(true);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected, logOpen, rawOpen, launchOpen, mainFeed, restoreMap, closeRaw]);

  const toggleFeed = (key: FeedKey) =>
    setFeeds((prev) => {
      const next = prev.includes(key)
        ? prev.filter((f) => f !== key)
        : ALL_FEEDS.filter((f) => f === key || prev.includes(f));
      // Hiding the feed that is holding the main panel takes the map with it.
      if (!next.includes(key) && mainFeed === key) setMainFeed(null);
      return next;
    });

  const detail = gids.find((g) => g.gid === selected) ?? null;

  // The pipeline's state belongs in the topbar, not only inside the dialog
  // that started it: a run that died is the reason the panes stopped filling,
  // and nobody watching the map would think to open the launcher to find out.
  const runChip =
    run?.state === "starting" ? { text: "PIPELINE STARTING", color: theme.warning }
      : run?.state === "running" ? { text: "PIPELINE RUNNING", color: theme.success }
        : run?.state === "error" ? { text: "PIPELINE FAILED", color: theme.danger }
          : run?.state === "stopped" ? { text: "RUN ENDED", color: theme.textDim }
            : null;

  const map = <MissionMap onSelectGid={selectGid} />;

  // The swap only holds while the strip is open, because the strip is where
  // the map goes. Every path that closes the strip also clears mainFeed, but
  // deriving it here means a missed one costs a stale layout rather than a
  // console with no map on it at all.
  const promoted = rawOpen && mainFeed && feeds.includes(mainFeed) ? mainFeed : null;

  return (
    <div className="console">
      <header className="topbar">
        <span className="topbar-title">SEARCH &amp; RESCUE</span>

        <span className="spacer" />

        {!connected && (
          <span
            className="chip"
            style={{ color: theme.danger }}
            title="The browser cannot reach the relay. Start it with: cd Back && npm run start:prod"
          >
            RELAY OFFLINE
          </span>
        )}
        {mission?.replay && (
          <span className="chip" style={{ color: theme.warning }} title="Replaying a finished run, not a live pipeline">
            REPLAY
          </span>
        )}
        {runChip && (
          <span
            className="chip"
            style={{ color: runChip.color }}
            title={run?.run?.video ? `Video: ${run.run.video}` : "Open New run for details"}
          >
            {runChip.text}
          </span>
        )}

        {rawOpen && (
          <span className="feed-toggles">
            {ALL_FEEDS.map((f) => (
              <button
                key={f}
                className={`btn${feeds.includes(f) ? " active" : ""}`}
                onClick={() => toggleFeed(f)}
                title={`${feeds.includes(f) ? "Hide" : "Show"} the ${FEED_LABEL[f].toLowerCase()} feed — ${FEED_HINT[f]}`}
              >
                {FEED_LABEL[f]}
              </button>
            ))}
          </span>
        )}

        <button
          className="btn"
          onClick={() => setLaunchOpen(true)}
          title="Choose a video and a telemetry CSV, and run the pipeline on them (N)"
        >
          New run…
        </button>
        <button
          className={`btn${rawOpen ? " active" : ""}`}
          onClick={() => (rawOpen ? closeRaw() : setRawOpen(true))}
          title={rawOpen
            ? "Hide the pipeline's video feeds (R)"
            : "Show the pipeline's video feeds: detections, traversability and depth (R)"}
        >
          Raw feeds
        </button>
        <button
          className={`btn${logOpen ? " active" : ""}`}
          onClick={() => setLogOpen((v) => !v)}
          title="Open the pipeline's full mission log (L)"
        >
          Mission log
        </button>
      </header>

      {/* raw-solo: one feed in the strip, so it and the map split the centre
          column evenly instead of the map keeping the lion's share. Two panes
          showing the same scene are being compared, and that reads badly when
          one of them is half the size. */}
      <main className={`dash${rawOpen ? " raw-open" : ""}${rawOpen && feeds.length === 1 ? " raw-solo" : ""}`}>
        <GidPane area="gids" selected={selected} onSelect={selectGid} />

        {/* The main panel: the mission map, unless a feed has been swapped
            into it. Both are given the same grid area, so the layout does not
            know or care which one is currently there. */}
        {promoted ? (
          <FeedPane feed={promoted} area="map" expanded onExpand={restoreMap} />
        ) : (
          <section className="pane area-map">{map}</section>
        )}

        {rawOpen && feeds.length > 0 && (
          <RawFeeds
            feeds={feeds}
            mainFeed={promoted}
            onExpand={expandFeed}
            swapped={
              <section className="pane">
                <header className="pane-head">
                  <span className="pane-title">Mission map</span>
                  <span className="spacer" />
                  <button
                    className="btn"
                    onClick={restoreMap}
                    title="Put the mission map back in the main panel"
                  >
                    Back to main
                  </button>
                </header>
                <div className="pane-body">{map}</div>
              </section>
            }
          />
        )}

        <SystemPane area="system" />
        <NotifyPane area="notify" onSelectGid={selectGid} />
      </main>

      <StatusBar />

      {detail && (
        <GidDetail
          gid={detail}
          onClose={() => setSelected(null)}
          onRoute={() => send("plan_path", { gid: detail.gid })}
        />
      )}

      {logOpen && <LogDrawer onClose={() => setLogOpen(false)} />}

      {launchOpen && <LaunchDialog onClose={() => setLaunchOpen(false)} />}
    </div>
  );
}
