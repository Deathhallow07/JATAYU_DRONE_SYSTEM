#!/usr/bin/env bash
# One video in, all three panes out.
#
#   ./Demo.sh /path/to/clip.mp4
#   ./Demo.sh /path/to/clip.mp4 --telem /path/to/fcb.csv   # real GPS
#   ./Demo.sh /path/to/clip.mp4 --stride 10 --reuse
#
# Runs YOLOE over the video once to build a run folder (real detections, real
# crops), then brings up the console with:
#
#   Global IDs      replaying that run
#   Depth           running on the same video
#   Traversability  running on the same video
#
# The ingest pass is the slow part — a few minutes on CPU. It is cached per
# video, so --reuse skips it on later runs of the same clip.
#
# For the REAL pipeline (YOLOE + ReID + geolocalisation) instead of the ingest
# pass, use ./Run.sh live — see README.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$PWD"
SESSION="sihdemo"

VIDEO="${1:-}"
shift || true

if [ -z "$VIDEO" ] || [ ! -f "$VIDEO" ]; then
  echo "usage: $0 <video> [--telem <csv>] [--stride N] [--reuse] [--max-frames N]" >&2
  [ -n "$VIDEO" ] && echo "not a file: $VIDEO" >&2
  exit 64
fi
VIDEO="$(readlink -f "$VIDEO")"

TELEM=""
REUSE=0
INGEST_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --telem)      TELEM="$(readlink -f "$2")"; INGEST_ARGS+=(--telem "$TELEM"); shift 2 ;;
    --reuse)      REUSE=1; shift ;;
    --stride|--max-frames|--conf|--imgsz|--min-hits)
                  INGEST_ARGS+=("$1" "$2"); shift 2 ;;
    *)            INGEST_ARGS+=("$1"); shift ;;
  esac
done

# Two environments, because no single one here has both stacks:
#   INGEST_PY needs ultralytics (YOLOE)
#   PY        needs python-socketio + opencv (watcher, workers)
# Point both at the same interpreter if you have one with everything.
INGEST_PY="${INGEST_PY:-$HOME/miniconda3/envs/work_env/bin/python}"
PY="${PY:-$HOME/miniconda3/envs/gcs_gui/bin/python}"
[ -x "$INGEST_PY" ] || INGEST_PY="$(command -v python3)"
[ -x "$PY" ]        || PY="$(command -v python3)"

# shellcheck disable=SC1091
[ -f Python/.env ] && set -a && . Python/.env && set +a
GUI_PORT="${GUI_PORT:-7100}"
export GUI_URL="${GUI_URL:-http://127.0.0.1:$GUI_PORT}"

# One cache directory per video, so re-running a clip is instant and two clips
# never land in the same run folder.
#
# Sanitised with bash substitution, not `tr -c`: tr would also translate the
# newline basename emits into the replacement character, giving every slug a
# trailing "_" — which never matches the directory written by a previous run,
# so --reuse would silently re-ingest every time.
SLUG="$(basename "${VIDEO%.*}")"
SLUG="${SLUG//[^A-Za-z0-9_.-]/_}"
export MISSION_ROOT="$ROOT/demo_runs/$SLUG"
mkdir -p "$MISSION_ROOT"

EXISTING="$(find "$MISSION_ROOT" -mindepth 2 -maxdepth 2 -name mission.log 2>/dev/null | head -1 || true)"

if [ -n "$EXISTING" ] && [ "$REUSE" = "1" ]; then
  echo "Reusing cached ingest: $(dirname "$EXISTING")"
else
  [ -n "$EXISTING" ] && echo "Cached ingest exists (pass --reuse to skip re-running it)."
  echo "Ingesting $VIDEO — YOLOE over the video, this takes a few minutes on CPU…"
  "$INGEST_PY" Python/ingest_video.py --video "$VIDEO" --out "$MISSION_ROOT" "${INGEST_ARGS[@]}"
fi

# Both launchers bind the same port, so they cannot run side by side — drop the
# other one rather than leaving a stale relay to win the bind and serve a
# different mission than the one just started.
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux kill-session -t sihgui     2>/dev/null || true
tmux new-session -d -s "$SESSION" -n relay -c "$ROOT/Back"  "npm run start:prod"
tmux new-window  -t "$SESSION"    -n front -c "$ROOT/Front" "npm run dev -- --host"
sleep 1
tmux new-window  -t "$SESSION"    -n watch -c "$ROOT/Python" \
  "$PY mission_watcher.py --root '$MISSION_ROOT' --replay --speed ${SPEED:-4}; bash"
tmux new-window  -t "$SESSION"    -n depth -c "$ROOT/Python/workers" \
  "$PY depth_worker.py --source '$VIDEO' --backend ${DEPTH_BACKEND:-stub} --fps ${WORKER_FPS:-4}; bash"
tmux new-window  -t "$SESSION"    -n seg   -c "$ROOT/Python/workers" \
  "$PY seg_worker.py --source '$VIDEO' --backend ${SEG_BACKEND:-stub} --fps ${WORKER_FPS:-3}; bash"

cat <<EOS

  console : http://localhost:5273
  video   : $VIDEO
  run     : $MISSION_ROOT

  tmux attach -t $SESSION      (ctrl-b n / p to move between panes, ctrl-b d to detach)
  tmux kill-session -t $SESSION

EOS
