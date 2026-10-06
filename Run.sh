#!/usr/bin/env bash
# Bring the console up in a tmux session, one pane per process.
#
#   ./Run.sh console       relay + console only — then press NEW RUN in the GUI
#   ./Run.sh               replay the newest run in demo_runs/ (no GPU, no pipeline)
#   ./Run.sh live          tail the newest run under MISSION_ROOT
#   ./Run.sh live <folder> tail a specific run folder
#
# `console` is the mode to use with the GUI's New run dialog. It starts NOTHING
# but the relay and the browser app, because the relay spawns the pipeline, the
# watcher and the telemetry replay itself when you press Start run.
#
# The other two modes start a mission_watcher of their own. Running one of them
# alongside a launched run puts TWO watchers on the same mission.log, and every
# GID and log line arrives at the console twice.
#
# Source it (". ./Run.sh") to be dropped into the session; run it to be told
# how to attach.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$PWD"
SESSION="sihgui"

MODE="${1:-replay}"
RUN_ARG="${2:-}"

# Conda env holding opencv + python-socketio. Override: PY=/path/to/python ./Run.sh
PY="${PY:-$HOME/miniconda3/envs/gcs_gui/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

# ── Node ─────────────────────────────────────────────────────────────────────
#
# Vite 7 needs Node >= 20.19. Ubuntu 24.04 ships 18, and on a box where that is
# the only one on PATH the dev server dies on an import error one line into
# startup. Look for a newer one installed under ~/.local/node (no root needed,
# see the message below) before giving up.
#
# Checked HERE rather than left to fail inside tmux: a tmux window whose
# command exits is destroyed, so the error would scroll past into a window that
# no longer exists and the only symptom is a session with a window missing.
node_ok() {
  local v
  v="$("$1" -v 2>/dev/null)" || return 1
  v="${v#v}"
  [ "$(printf '%s\n20.19.0\n' "$v" | sort -V | head -1)" = "20.19.0" ]
}

NODE_BIN="${NODE_BIN:-}"
if [ -z "$NODE_BIN" ]; then
  if node_ok "$(command -v node 2>/dev/null)"; then
    NODE_BIN="$(dirname "$(command -v node)")"
  elif node_ok "$HOME/.local/node/bin/node"; then
    NODE_BIN="$HOME/.local/node/bin"
  fi
fi

if [ -z "$NODE_BIN" ]; then
  cat >&2 <<'MSG'

  No Node >= 20.19 found, and Vite 7 will not start without one.

  Install one into your home directory -- no root, nothing system-wide:

      VER=$(curl -s https://nodejs.org/dist/index.json \
            | python3 -c 'import json,sys;print([r["version"] for r in json.load(sys.stdin) if r["lts"]][0])')
      ARCH=$(uname -m); [ "$ARCH" = "x86_64" ] && ARCH=x64 || ARCH=arm64
      mkdir -p ~/.local/node
      curl -fsSL "https://nodejs.org/dist/$VER/node-$VER-linux-$ARCH.tar.xz" \
        | tar -xJ -C ~/.local/node --strip-components=1

  This script picks it up from there automatically. Or point NODE_BIN at one:

      NODE_BIN=/path/to/node/bin ./Run.sh console

MSG
  exit 1
fi
export PATH="$NODE_BIN:$PATH"

# ── Dependencies ─────────────────────────────────────────────────────────────
#
# node_modules holds platform-native binaries (esbuild, rollup), so it is never
# copied between machines -- and an absent one fails the same silent way the
# old Node did.
for d in Back Front; do
  if [ ! -d "$ROOT/$d/node_modules" ]; then
    echo "$d/node_modules is missing -- installing."
    (cd "$ROOT/$d" && npm install --no-audit --no-fund) || {
      echo "npm install failed in $d/" >&2; exit 1; }
  fi
done

# shellcheck disable=SC1091
[ -f Python/.env ] && set -a && . Python/.env && set +a
GUI_PORT="${GUI_PORT:-7100}"
export GUI_URL="${GUI_URL:-http://127.0.0.1:$GUI_PORT}"

# The demo source for the model panes. Any video, image, camera index or MJPEG
# URL; the pipeline's own stream (DISPLAY_MODE="stream") is a good live choice.
WORKER_SOURCE="${WORKER_SOURCE:-}"

case "$MODE" in
  console)
    # Nothing to watch yet: the run does not exist until the operator picks a
    # clip. MISSION_ROOT still has to be exported, because it is both where the
    # launcher tells the pipeline to write and the jail the relay serves crops
    # from — the two must be the same directory or every crop 403s.
    : "${MISSION_ROOT:=$ROOT/mission_logs}"
    mkdir -p "$MISSION_ROOT"
    WATCH_ARGS=""
    ;;
  replay)
    MISSION_ROOT="${DEMO_ROOT:-$ROOT/demo_runs}"
    if [ -z "$(ls -A "$MISSION_ROOT" 2>/dev/null)" ]; then
      echo "No demo run found — generating one."
      "$PY" Python/make_demo_run.py --out "$MISSION_ROOT"
    fi
    WATCH_ARGS="--root $MISSION_ROOT --replay --speed 4"
    ;;
  live)
    : "${MISSION_ROOT:?MISSION_ROOT is not set — put it in Python/.env}"
    WATCH_ARGS="--root $MISSION_ROOT"
    [ -n "$RUN_ARG" ] && WATCH_ARGS="$WATCH_ARGS --run $RUN_ARG"
    ;;
  *)
    echo "usage: $0 [console|replay|live] [run-folder]" >&2
    exit 64
    ;;
esac
export MISSION_ROOT

# Demo.sh binds the same port; only one console at a time.
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux kill-session -t sihdemo    2>/dev/null || true
# `; exec bash` on every window on purpose. Without it tmux destroys a window
# the moment its command exits, so a process that dies at startup takes its own
# error message with it -- and the session just comes up with a window missing,
# which looks like the script never tried to start it.
hold() { printf '%s; echo; echo "[%s exited -- scroll up for why. Ctrl-b d to detach]"; exec bash' "$1" "$2"; }

tmux new-session -d -s "$SESSION" -n relay -c "$ROOT/Back" \
  "$(hold 'npm run start:prod' relay)"
tmux new-window  -t "$SESSION"    -n front -c "$ROOT/Front" \
  "$(hold 'npm run dev -- --host' front)"
sleep 1
if [ -n "$WATCH_ARGS" ]; then
  tmux new-window -t "$SESSION" -n watch -c "$ROOT/Python" "$PY mission_watcher.py $WATCH_ARGS; bash"
fi

if [ "$MODE" = "console" ]; then
  echo "console mode — the relay starts the pipeline, the watcher, the telemetry"
  echo "replay and (optionally) the depth/traversability workers when you press"
  echo "START RUN in the GUI. Nothing else needs to be running."
elif [ -n "$WORKER_SOURCE" ]; then
  tmux new-window -t "$SESSION" -n depth -c "$ROOT/Python/workers" \
    "$PY depth_worker.py --source '$WORKER_SOURCE' --backend ${DEPTH_BACKEND:-stub}; bash"
  tmux new-window -t "$SESSION" -n seg   -c "$ROOT/Python/workers" \
    "$PY seg_worker.py   --source '$WORKER_SOURCE' --backend ${SEG_BACKEND:-stub}; bash"
else
  echo "WORKER_SOURCE not set — depth and traversability feeds will stay empty."
  echo "  WORKER_SOURCE=/path/to/clip.mp4 ./Run.sh $MODE"
fi

# UAV telemetry for the map and the system panel. Opt-in: with no aircraft or
# SITL on the far end the worker would just retry forever in a pane nobody
# reads, and the console already falls back to the mission log's GPS track.
if [ "$MODE" = "console" ]; then
  # No MAVLink worker here even with ENABLE_UAV0 set. A launched run flies the
  # CSV (telem_csv.py), and a second producer emitting `telemetry` would fight
  # it for the drone marker — the map would jump between the recorded flight
  # and whatever the aircraft on the bench is reporting.
  echo "telemetry comes from the CSV you pick in NEW RUN."
elif [ "${ENABLE_UAV0:-false}" = "true" ]; then
  tmux new-window -t "$SESSION" -n telem -c "$ROOT/Python" \
    "$PY telem_worker.py --uav 0; bash"
else
  echo "ENABLE_UAV0 is not true — no MAVLink telemetry."
  echo "  the map will track the drone from the mission log's GPS fixes instead."
fi

echo
# The address to actually open. Vite prints these too, but in a window nobody
# attaches to -- and on a remote box "localhost" is the wrong answer.
LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "  console : http://localhost:5273${LAN_IP:+   (remote: http://$LAN_IP:5273)}"
echo "  relay   : http://localhost:$GUI_PORT/health"
echo "  mode    : $MODE   root: $MISSION_ROOT"
if [ "$MODE" = "console" ]; then
  echo
  echo "  Open the console and press NEW RUN (or N) to pick a video and a CSV."
fi
echo
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
  tmux attach -t "$SESSION"
else
  echo "  tmux attach -t $SESSION"
fi
