#!/usr/bin/env bash
# One video + one telemetry CSV in, the whole console live.
#
#   ./Live.sh clip.mp4 fcb.csv
#   ./Live.sh clip.mp4 fcb.csv --realtime --stride 10
#   ./Live.sh clip.mp4                       # no CSV: GPS is synthesised
#
# Unlike Demo.sh — which runs the detector over the whole clip first and then
# replays the finished run — this drives the console as it processes:
#
#   the drone flies the CSV's path on the map, dragging its trail
#   a casualty pin drops the moment a GID is created
#   the raw feeds show detections, traversability and depth on the same frame
#
# Everything comes from one process (Python/mission_run.py), so there is no
# watcher and nothing to replay.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$PWD"
SESSION="sihlive"

VIDEO="${1:-}"
shift || true
TELEM="${1:-}"
# Only treat the second argument as the CSV if it is not a flag.
case "$TELEM" in
  --*|"") TELEM="" ;;
  *)      shift ;;
esac

if [ -z "$VIDEO" ] || [ ! -f "$VIDEO" ]; then
  echo "usage: $0 <video.mp4> [telem.csv] [--realtime] [--stride N] [--reid PATH] ..." >&2
  [ -n "$VIDEO" ] && echo "not a file: $VIDEO" >&2
  exit 64
fi
VIDEO="$(readlink -f "$VIDEO")"

RUN_ARGS=(--video "$VIDEO")
if [ -n "$TELEM" ]; then
  [ -f "$TELEM" ] || { echo "not a file: $TELEM" >&2; exit 64; }
  RUN_ARGS+=(--telem "$(readlink -f "$TELEM")")
else
  echo "No telemetry CSV — GPS will be synthesised and the drone will not move."
fi
RUN_ARGS+=("$@")

# mission_run.py needs ultralytics AND python-socketio in ONE interpreter.
# Neither env here has both out of the box:
#   work_env   has ultralytics + torch
#   gcs_gui  has python-socketio + opencv
# Point RUN_PY at whichever you have completed:
#   ~/miniconda3/envs/work_env/bin/pip install "python-socketio[client]"
RUN_PY="${RUN_PY:-$HOME/miniconda3/envs/work_env/bin/python}"
PY="${PY:-$HOME/miniconda3/envs/gcs_gui/bin/python}"
[ -x "$RUN_PY" ] || RUN_PY="$(command -v python3)"
[ -x "$PY" ]     || PY="$(command -v python3)"

if ! "$RUN_PY" -c "import ultralytics, socketio" 2>/dev/null; then
  echo >&2
  echo "  $RUN_PY is missing ultralytics and/or python-socketio." >&2
  echo "  mission_run.py needs both in the SAME interpreter:" >&2
  echo >&2
  echo "      $RUN_PY -m pip install ultralytics 'python-socketio[client]'" >&2
  echo >&2
  echo "  or point this script at an interpreter that has both:" >&2
  echo "      RUN_PY=/path/to/python $0 $VIDEO ${TELEM:-}" >&2
  exit 1
fi

# shellcheck disable=SC1091
[ -f Python/.env ] && set -a && . Python/.env && set +a
GUI_PORT="${GUI_PORT:-7100}"
export GUI_URL="${GUI_URL:-http://127.0.0.1:$GUI_PORT}"

# The relay serves crops from here, and mission_run writes its run folder into
# it — so the two must agree or every crop 403s.
MISSION_ROOT="${DEMO_ROOT:-$ROOT/demo_runs}"
mkdir -p "$MISSION_ROOT"
export MISSION_ROOT
RUN_ARGS+=(--out "$MISSION_ROOT")

# The ReID checkpoint shipped with the repo, if it is there. Without it the
# console still works; identities are matched on GPS proximity alone.
if [ -z "${REID_MODEL:-}" ] && [ -f "$ROOT/mobileclip2_b.ts" ]; then
  export REID_MODEL="$ROOT/mobileclip2_b.ts"
fi
[ -n "${REID_MODEL:-}" ] && echo "ReID model: $REID_MODEL"

# Run.sh and Demo.sh bind the same relay port; only one console at a time.
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux kill-session -t sihgui     2>/dev/null || true
tmux kill-session -t sihdemo    2>/dev/null || true

tmux new-session -d -s "$SESSION" -n relay -c "$ROOT/Back"  "npm run start:prod"
tmux new-window  -t "$SESSION"    -n front -c "$ROOT/Front" "npm run dev -- --host"

# Give the relay a moment to bind, so mission_run's first emits are not dropped
# into a socket that is still connecting.
sleep 2

tmux new-window -t "$SESSION" -n mission -c "$ROOT/Python" \
  "$RUN_PY mission_run.py $(printf '%q ' "${RUN_ARGS[@]}"); bash"

echo
echo "  console : http://localhost:5273"
echo "  relay   : http://localhost:$GUI_PORT/health"
echo "  video   : $VIDEO"
echo "  telem   : ${TELEM:-<synthesised>}"
echo "  runs    : $MISSION_ROOT"
echo
echo "  mission_run spawns depth + traversability itself, against its own"
echo "  MJPEG hub on :8090, so all three raw feeds show the same frame."
echo
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
  tmux attach -t "$SESSION"
else
  echo "  tmux attach -t $SESSION"
fi
