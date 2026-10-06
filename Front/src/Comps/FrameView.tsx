import { useRef } from "react";
import type { ReactNode } from "react";

/**
 * A worker's JPEG with an optional interactive overlay.
 *
 * The overlay is anchored to a box that is exactly the picture, not to the
 * pane. object-fit: contain letterboxes the image inside the pane, so a canvas
 * pinned to the PANE would be offset from the picture by however much
 * letterboxing there is and every click would land on the wrong cell.
 *
 * That box is sized by ASPECT RATIO plus max-width/max-height rather than by
 * shrink-wrapping the image. A shrink-wrap box has an auto height, and the
 * image's own max-height: 100% then has no definite height to resolve against —
 * so the image renders at natural size and spills out of the pane wherever the
 * pane is shorter than the frame. With the ratio fixed, both constraints apply
 * to the box itself and the image simply fills it.
 */
export default function FrameView({
  jpeg, alt, width, height, onPick, children,
}: {
  jpeg: string;
  alt: string;
  /** The frame's pixel dimensions, used to give the box its aspect ratio. */
  width: number;
  height: number;
  /** Called with the click position as fractions of the image, both in [0,1). */
  onPick?: (fx: number, fy: number, ev: React.MouseEvent) => void;
  children?: ReactNode;
}) {
  const boxRef = useRef<HTMLDivElement | null>(null);

  const handle = (ev: React.MouseEvent) => {
    const el = boxRef.current;
    if (!el || !onPick) return;
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return;
    const fx = Math.min(0.999999, Math.max(0, (ev.clientX - r.left) / r.width));
    const fy = Math.min(0.999999, Math.max(0, (ev.clientY - r.top) / r.height));
    onPick(fx, fy, ev);
  };

  return (
    <div className="frame-wrap">
      <div
        className="frame-inner"
        ref={boxRef}
        onClick={onPick ? handle : undefined}
        style={{
          aspectRatio: `${width} / ${height}`,
          cursor: onPick ? "crosshair" : "default",
        }}
      >
        <img className="frame-img" src={`data:image/jpeg;base64,${jpeg}`} alt={alt} />
        {children}
      </div>
    </div>
  );
}
