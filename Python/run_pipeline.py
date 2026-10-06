#!/usr/bin/env python3
"""
Run Pipeline_yoloe_logs_robust.py on a video and CSV chosen in the console.

The pipeline keeps its configuration in module-level constants at the top of
the file — VIDEO_PATH, TELEM_CSV_PATH, DISPLAY_MODE and so on — and they are
read while the module is still importing (CoordinateTransformer is constructed
at import, and the robust-detect backend is resolved there too). So there is no
function to call with different arguments, and `import`ing it and reassigning
the globals afterwards is already too late.

This launcher therefore reads the pipeline's source, rewrites those constant
ASSIGNMENTS textually, and executes the result as `__main__`:

    python run_pipeline.py --video clip.mp4 --telem fcb.csv --out mission_logs/

Why rewrite the source rather than edit the file:

  * the pipeline on disk stays byte-for-byte what is being graded. Nothing the
    console does can slow it down, crash it, or leave a half-applied edit
    behind after a Ctrl-C.
  * two operators can launch two runs with different clips from the same
    checkout without racing each other through the same file.
  * every override is asserted to have applied. A constant that got renamed
    upstream fails loudly here, rather than running the whole clip against the
    path the file happened to be holding.

`__file__` is set to the real pipeline path, so BASE_DIR, the RUN_FOLDER it
derives, and the sibling imports (coordinate_transformer, robust_detect) all
resolve exactly as they do when it is run directly.

CUDA is not required to launch: the pipeline picks its own device and will run
on CPU, slowly. That is the box this was written on.
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PIPELINE = os.environ.get("PIPELINE_SCRIPT") or os.path.join(
    HERE, "..", "pipeline", "Pipeline_yoloe_logs_robust.py",
)


def _literal(value):
    """Python source for a constant, so a Windows-ish path cannot escape."""
    return repr(value)


def override(src, name, value, required=True):
    """
    Replace a module-level `NAME = ...` assignment with `NAME = <value>`.

    Anchored at column 0 so it only ever hits the configuration block at the
    top of the file, never a rebinding inside a function (`GALLERY_SAVE_DIR`
    is assigned in both places) and never a commented-out alternative — of
    which the config block has several for every path.
    """
    pattern = re.compile(rf"^{re.escape(name)}\s*=[^\n]*$", re.MULTILINE)
    new, n = pattern.subn(f"{name} = {_literal(value)}", src)
    if n == 0:
        if required:
            raise SystemExit(
                f"run_pipeline: no top-level `{name} = ...` in the pipeline — "
                f"it has been renamed or moved, and this override would have "
                f"been silently ignored."
            )
        return src
    return new


def redirect_run_folder(src, out_dir):
    """
    Point the run folder at the console's MISSION_ROOT.

    RUN_FOLDER is built inside `if __name__ == "__main__"` from BASE_DIR, so it
    is not a constant `override()` can reach:

        RUN_FOLDER = os.path.join(
            BASE_DIR,
            "mission_logs",
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    Only the BASE_DIR/"mission_logs" pair is replaced; the timestamped leaf is
    left alone, so a run still gets its own folder and the watcher still sees a
    new one appear. The relay serves crops out of MISSION_ROOT and refuses
    anything that resolves outside it, so without this every crop in the
    console 403s.
    """
    pattern = re.compile(
        r"RUN_FOLDER\s*=\s*os\.path\.join\(\s*\n\s*BASE_DIR\s*,\s*\n\s*\"mission_logs\"\s*,"
    )
    new, n = pattern.subn(
        lambda _: f"RUN_FOLDER = os.path.join(\n        {_literal(out_dir)},", src
    )
    if n != 1:
        raise SystemExit(
            "run_pipeline: could not redirect RUN_FOLDER — the pipeline's "
            "__main__ no longer builds it from BASE_DIR/mission_logs. Point "
            "MISSION_ROOT at the pipeline's own mission_logs/ instead."
        )
    return new


def downscale_stream(src, quality, width):
    """
    Make the MJPEG monitoring feed affordable to watch over a network.

    The pipeline encodes it with a bare `cv2.imencode(".jpg", frame)` — no
    quality argument, so OpenCV's default of 95, at the full 1920x1080. Measured
    on the DGX that is 779 KB a frame at 17.8 fps: **113 Mbps**, for a feed
    whose job is to let someone see that the detector is working.

    A gigabit LAN carries it. Nothing else does — over a VPN the browser falls
    progressively further behind, and because MJPEG has no timestamps and no
    frame dropping, "behind" means the video pane shows a different minute of
    the flight from the map and the casualty list beside it. The feeds do not
    disagree; one of them is just late.

    q70 at 1280 wide is visually the same overlay at roughly an eighth of the
    bytes. The DETECTOR still runs at full native resolution — this is the
    monitoring copy, downscaled after everything has been drawn on it, and it
    changes nothing about what the pipeline finds or records.
    """
    anchor = '                    ok, buf = cv2.imencode(".jpg", frame)'
    if anchor not in src:
        raise SystemExit(
            "run_pipeline: could not find the stream server's imencode call — "
            "the MJPEG encoder has been rewritten upstream. Drop "
            "--stream-quality/--stream-width to run without this."
        )

    # Built as a list of lines to keep the pipeline's own indentation exact:
    # this lands inside a nested method and Python is unforgiving about it.
    lines = [
        "                    _f = frame",
    ]
    if width:
        lines += [
            f"                    if _f.shape[1] > {width}:",
            f"                        _s = {width} / float(_f.shape[1])",
            "                        _f = cv2.resize(_f, None, fx=_s, fy=_s,",
            "                                        interpolation=cv2.INTER_AREA)",
        ]
    if quality:
        lines += [
            '                    ok, buf = cv2.imencode(".jpg", _f,',
            f"                        [int(cv2.IMWRITE_JPEG_QUALITY), {quality}])",
        ]
    else:
        lines.append('                    ok, buf = cv2.imencode(".jpg", _f)')

    replacement = "\n".join(lines)
    return src.replace(anchor, replacement, 1)


def build(args):
    with open(args.pipeline) as fh:
        src = fh.read()

    src = override(src, "VIDEO_SOURCE", "video")
    src = override(src, "VIDEO_PATH", args.video)
    src = override(src, "START_FRAME", args.start_frame)

    if args.telem:
        src = override(src, "TELEM_SOURCE", "csv")
        src = override(src, "TELEM_CSV_PATH", args.telem)
    else:
        # No CSV means no attitude and no fix, so every detection fails to
        # project. Say so here rather than letting the run report 0/N located.
        print("run_pipeline: no telemetry CSV — detections cannot be "
              "projected to the ground and casualty pins will not be placed.",
              file=sys.stderr)

    # "stream" is what makes the Geolocalisation raw feed work: the browser
    # reads this MJPEG server directly (Front/src/config.ts), bypassing the
    # relay. "window" needs a display the pipeline host may not have.
    src = override(src, "DISPLAY_MODE", args.display)
    src = override(src, "STREAM_PORT", args.stream_port)

    if args.weights:
        src = override(src, "YOLOE_WEIGHTS", args.weights)
    if args.reid:
        src = override(src, "REID_ONNX_PATH", args.reid)
    if args.zoom is not None:
        src = override(src, "ZOOM_RATIO", args.zoom, required=False)

    if args.out:
        src = redirect_run_folder(src, os.path.abspath(args.out))

    if args.display == "stream" and (args.stream_quality or args.stream_width):
        src = downscale_stream(src, args.stream_quality, args.stream_width)

    return src


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="the clip to run the pipeline on")
    ap.add_argument("--telem", help="flight-controller CSV covering the clip")
    ap.add_argument("--out", default=os.environ.get("MISSION_ROOT"),
                    help="mission_logs root; the run folder is created inside it")
    ap.add_argument("--pipeline", default=DEFAULT_PIPELINE,
                    help="Pipeline_yoloe_logs_robust.py (env: PIPELINE_SCRIPT)")
    ap.add_argument("--weights", default=os.environ.get("YOLOE_WEIGHTS"),
                    help="YOLOE .pt (env: YOLOE_WEIGHTS)")
    ap.add_argument("--reid", default=os.environ.get("REID_CKPT"),
                    help="ReID checkpoint (env: REID_CKPT)")
    ap.add_argument("--start-frame", type=int, default=0,
                    help="begin part-way into the clip; the CSV follows")
    ap.add_argument("--display", default=os.environ.get("PIPELINE_DISPLAY", "stream"),
                    choices=("stream", "window", "none"))
    ap.add_argument("--stream-port", type=int,
                    default=int(os.environ.get("PIPELINE_STREAM_PORT", 8080)))
    ap.add_argument("--zoom", type=float, default=None,
                    help="fallback zoom ratio for frames the telemetry cannot "
                         "speak for; a CSV with a zoom_ratio column drives the "
                         "intrinsics per frame and ignores this")
    ap.add_argument("--stream-quality", type=int,
                    default=int(os.environ.get("PIPELINE_STREAM_QUALITY", 70)),
                    help="JPEG quality of the MONITORING feed only, 1-100 "
                         "(env: PIPELINE_STREAM_QUALITY). 0 leaves the "
                         "pipeline's own encoding alone.")
    ap.add_argument("--stream-width", type=int,
                    default=int(os.environ.get("PIPELINE_STREAM_WIDTH", 1280)),
                    help="downscale the monitoring feed to this width "
                         "(env: PIPELINE_STREAM_WIDTH). 0 keeps full size. "
                         "The detector always sees the native frame.")
    ap.add_argument("--pythonpath", default=os.environ.get("PIPELINE_PYTHONPATH", ""),
                    help="extra import directories, colon-separated "
                         "(env: PIPELINE_PYTHONPATH) — for modules the pipeline "
                         "imports from outside its own folder")
    ap.add_argument("--dry-run", action="store_true",
                    help="apply the overrides, print the config block, do not run")
    args = ap.parse_args()

    args.pipeline = os.path.abspath(args.pipeline)
    if not os.path.isfile(args.pipeline):
        raise SystemExit(f"run_pipeline: no pipeline at {args.pipeline}")

    args.video = os.path.abspath(args.video)
    if not os.path.isfile(args.video):
        raise SystemExit(f"run_pipeline: no video at {args.video}")

    if args.telem:
        args.telem = os.path.abspath(args.telem)
        if not os.path.isfile(args.telem):
            raise SystemExit(f"run_pipeline: no telemetry CSV at {args.telem}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    src = build(args)

    if args.dry_run:
        for line in src.splitlines():
            if re.match(r"^(VIDEO_|TELEM_|DISPLAY_MODE|STREAM_PORT|START_FRAME|"
                        r"YOLOE_WEIGHTS|REID_ONNX_PATH|ZOOM_RATIO)", line):
                print(line)
        print(re.search(r"RUN_FOLDER = os\.path\.join\(.*?\n\s*\)", src, re.S).group(0))
        return 0

    pipe_dir = os.path.dirname(args.pipeline)

    # The pipeline's own folder first, so its sibling modules (robust_detect,
    # and usually coordinate_transformer) import the way they do when it is run
    # directly. Then anything PIPELINE_PYTHONPATH adds.
    for d in reversed([pipe_dir] + [x for x in args.pythonpath.split(":") if x.strip()]):
        d = os.path.abspath(d.strip())
        if d not in sys.path:
            sys.path.insert(0, d)

    os.chdir(pipe_dir)

    # The pipeline hardcodes sys.path.append() for the machine it was written
    # on. Those paths do not exist here, and the resulting failure is a bare
    # ModuleNotFoundError three seconds in that says nothing about which
    # setting is missing — so say it now, while the message is still next to
    # the thing that caused it.
    missing = [m.group(1) for m in re.finditer(
        r'^sys\.path\.append\(\s*["\']([^"\']+)["\']\s*\)', src, re.MULTILINE)]
    dead = [d for d in missing if not os.path.isdir(d)]
    if dead:
        print(f"run_pipeline: the pipeline adds import paths that do not exist "
              f"on this machine: {', '.join(dead)}", file=sys.stderr)
        print(f"  If it then fails with ModuleNotFoundError, point "
              f"PIPELINE_PYTHONPATH in Python/.env at the directory holding "
              f"that module.", file=sys.stderr)
        print(f"  Currently PIPELINE_PYTHONPATH={args.pythonpath or '<unset>'}",
              file=sys.stderr)

    # argv is reset so the pipeline — and anything it imports that parses
    # arguments, ultralytics included — does not see this launcher's flags.
    sys.argv = [args.pipeline]

    print(f"run_pipeline: {args.pipeline}", file=sys.stderr)
    print(f"  video  : {args.video}", file=sys.stderr)
    print(f"  telem  : {args.telem or '<none>'}", file=sys.stderr)
    print(f"  out    : {args.out or pipe_dir + '/mission_logs'}", file=sys.stderr)
    print(f"  display: {args.display}:{args.stream_port}", file=sys.stderr)
    if args.pythonpath:
        print(f"  imports: {args.pythonpath}", file=sys.stderr)
    if args.display == "stream" and (args.stream_quality or args.stream_width):
        print(f"  feed   : q{args.stream_quality} at {args.stream_width or 'native'}px wide "
              f"(monitoring copy only; the detector sees the native frame)",
              file=sys.stderr)

    code = compile(src, args.pipeline, "exec")
    g = {"__name__": "__main__", "__file__": args.pipeline, "__package__": None}
    exec(code, g)  # noqa: S102 — running the pipeline is the entire point
    return 0


if __name__ == "__main__":
    sys.exit(main())
