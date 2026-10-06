import type { ReactNode } from "react";
import PipelinePane from "./PipelinePane";
import SegPane from "./SegPane";
import DepthPane from "./DepthPane";

export type FeedKey = "pipeline" | "seg" | "depth";

export const FEED_LABEL: Record<FeedKey, string> = {
  pipeline: "Geolocalisation",
  seg: "Traversability",
  depth: "Depth",
};

/** What each feed actually shows, for the tooltip on its toggle. */
export const FEED_HINT: Record<FeedKey, string> = {
  pipeline: "The pipeline's own annotated video: detection boxes, track ids and the geolocalisation overlay",
  seg: "Ground segmentation and the planned route across it",
  depth: "Per-pixel depth estimate of the same frame",
};

/**
 * One feed, rendered wherever it currently lives.
 *
 * `area` is a grid-area name, so the same component is the bottom-strip pane
 * or the main panel depending only on which name it is given — the swap in
 * App.tsx is a change of slot, not a different component with different state.
 * That matters for the Geolocalisation feed in particular: remounting it would
 * tear down the MJPEG connection and re-open it, and the pipeline would spend
 * a second encoding for a viewer that had gone away.
 */
export function FeedPane({ feed, area, expanded, onExpand }: {
  feed: FeedKey;
  area: string;
  expanded?: boolean;
  onExpand?: () => void;
}) {
  const props = { area, expanded, onExpand };
  if (feed === "pipeline") return <PipelinePane {...props} />;
  if (feed === "seg") return <SegPane {...props} />;
  return <DepthPane {...props} />;
}

/**
 * The engineer's view of what the drone is actually seeing, in pipeline order:
 * detections and geolocalisation, then traversability, then depth.
 *
 * Hidden by default. A medic never needs it, and the three feeds together are
 * three video decodes — keeping it collapsed keeps them off the wire entirely,
 * because each pane stops its own stream when it unmounts.
 *
 * When a feed has been swapped into the main panel, `swapped` is rendered in
 * its place here rather than the feed itself — that is the other half of the
 * trade, and it keeps the strip the same width whichever way round it is.
 */
export default function RawFeeds({ feeds, mainFeed, onExpand, swapped }: {
  feeds: FeedKey[];
  /** The feed currently occupying the main panel, if any. */
  mainFeed: FeedKey | null;
  onExpand: (feed: FeedKey) => void;
  /** What to show in the promoted feed's slot — the mission map. */
  swapped?: ReactNode;
}) {
  return (
    <section className="raw-feeds area-raw">
      {feeds.map((f) =>
        f === mainFeed
          ? <div key={f} className="raw-slot">{swapped}</div>
          : <FeedPane key={f} feed={f} area={`raw-${f}`} onExpand={() => onExpand(f)} />,
      )}
    </section>
  );
}
