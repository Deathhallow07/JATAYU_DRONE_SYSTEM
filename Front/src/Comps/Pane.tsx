import type { ReactNode } from "react";
import { theme } from "../theme";

interface PaneProps {
  title: string;
  subtitle?: string;
  area: string;
  actions?: ReactNode;
  children: ReactNode;
  center?: boolean;
  /**
   * Lay the body out as a flex column instead of a scroll box. For panes whose
   * body is a fixed strip plus one flexible region (a filter bar over a log, a
   * frame over its readouts) — those must not size the flexible part with
   * calc(100% - Npx), because the strip's height is not fixed: readouts wrap to
   * a second line in a narrow pane and the content then overflows by exactly
   * one line.
   */
  column?: boolean;
  /** Pane-level status light: colour + tooltip. */
  status?: { color: string; label: string };
  /**
   * Swap this pane with the main panel (where the mission map normally sits).
   *
   * Given to the raw feeds, which are otherwise a 190px strip along the bottom
   * — too small to read a detection box in. `expanded` is true for the pane
   * currently occupying the main panel, and the button then puts the map back.
   */
  onExpand?: () => void;
  expanded?: boolean;
}

/**
 * Shared pane chrome: title bar, optional status light, actions, scroll body.
 *
 * Every pane is the same shape so the grid reads as one instrument rather than
 * four widgets, and so the focus (maximise) affordance is in the same place in
 * each.
 */
export default function Pane({
  title, subtitle, area, actions, children, center, column, status, onExpand, expanded,
}: PaneProps) {
  return (
    <section className={`pane area-${area}`}>
      <header className="pane-head">
        {status && (
          <span
            className="dot"
            style={{ background: status.color }}
            title={status.label}
            aria-label={status.label}
          />
        )}
        <span className="pane-title">{title}</span>
        {subtitle && <span className="pane-sub">{subtitle}</span>}
        <span className="spacer" />
        {actions}
        {onExpand && (
          // Labelled, not an icon alone: an arrows-out glyph on a feed could
          // equally mean browser fullscreen, and this does something else —
          // it trades places with the map.
          <button
            className={`btn${expanded ? " active" : ""}`}
            onClick={onExpand}
            title={expanded
              ? "Put the mission map back in the main panel and return this feed to the strip"
              : `Swap this feed into the main panel, moving the mission map down here`}
          >
            <ExpandIcon collapse={expanded} />
            {expanded ? "Back to map" : "Main panel"}
          </button>
        )}
      </header>
      <div className={`pane-body${center ? " center" : ""}${column ? " column" : ""}`}>
        {children}
      </div>
    </section>
  );
}

/** Arrows out of / into a box — the direction the pane is about to move. */
const ExpandIcon = ({ collapse }: { collapse?: boolean }) => (
  <svg className="btn-icon" viewBox="0 0 16 16" width="11" height="11" aria-hidden="true">
    {collapse ? (
      <path
        d="M7 1v5H2M9 15v-5h5"
        fill="none" stroke="currentColor" strokeWidth="1.6"
        strokeLinecap="round" strokeLinejoin="round"
      />
    ) : (
      <path
        d="M10 1h5v5M6 15H1v-5M15 1l-5 5M1 15l5-5"
        fill="none" stroke="currentColor" strokeWidth="1.6"
        strokeLinecap="round" strokeLinejoin="round"
      />
    )}
  </svg>
);

export const EmptyState = ({ children }: { children: ReactNode }) => (
  <div className="empty-state" style={{ color: theme.textFaint }}>{children}</div>
);
