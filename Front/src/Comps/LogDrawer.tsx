import LogPane from "./LogPane";

/**
 * The full mission log, kept for debugging but out of the medic's way.
 *
 * The dashboard's notification panel deliberately drops almost everything the
 * pipeline writes. When something looks wrong the raw log is the only way to
 * find out why, so it stays reachable — behind a toggle, as a drawer over the
 * dashboard rather than a pane competing with it.
 */
export default function LogDrawer({ onClose }: { onClose: () => void }) {
  return (
    <div
      className="drawer-backdrop"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="drawer log-drawer"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Mission log"
      >
        <LogPane area="drawer" onClose={onClose} />
      </div>
    </div>
  );
}
