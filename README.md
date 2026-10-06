# SIH — Search & Rescue Console

An operator console for the search-and-rescue pipeline, built for the person
who acts on it: where the drone is, who it has found, and where they are lying.

Built to sit alongside the existing GCS — it shares that console's
visual language, its Mapbox map and its telemetry field names — but it is a
separate app with a much smaller backend.

```
┌──────────────┬─────────────────────────┬─────────────────┐
│              │                         │  System         │
│  Casualties  │   Mission map           │  UAV + health   │
│  (found so   │   drone · path · pins   ├─────────────────┤
│   far)       │                         │  Notifications  │
│              ├─────────────────────────┤  (plain words)  │
│              │   Raw feeds  (toggle)   │                 │
└──────────────┴─────────────────────────┴─────────────────┘
                        status bar
```

The map is the dashboard: the drone with a fading breadcrumb trail, and a
numbered pin for every casualty found, coloured to match its card in the left
bar. Clicking a pin, a card or a notification opens the same casualty record.

`N` starts a run, `R` toggles the raw feeds, `L` opens the full mission log,
`Esc` backs out one level.

### Raw feeds

Hidden by default, and in pipeline order when shown:

1. **Geolocalisation** — the pipeline's own annotated output, pulled straight
   from its MJPEG server (`DISPLAY_MODE = "stream"`). Not proxied through the
   relay; see `Front/src/Comps/PipelinePane.tsx` for why.
2. **Traversability** — segmentation grid + the planned route.
3. **Depth** — Depth-Anything, or the stub.

Each feed stops its stream when hidden, so a closed panel costs nothing.

Each has a **Main panel** button that trades places with the mission map: the
feed moves into the centre and the map drops into the slot it left. A 190px
strip is too small to read a detection box in, and the trade is reversible from
the same button, which now reads **Back to map**.

### Notifications, not logs

The mission log is an engineering instrument — `DET`/`TRACK`/`REID` fire
hundreds of times a minute and describe internal state. The notification panel
shows only what someone can act on:

| | |
|---|---|
| `Casualty 6 found` | a new global ID, with its coordinates and photo count |
| `Casualty 3 seen again` | a merge — the re-identification this system exists to do |
| `Search complete` | mission bookends |
| problems | errors, lost links, a stopped pipeline — never blur rejects |

Anything unrecognised is dropped on purpose. The raw log is still one key away
(`L`), so this panel never has to degrade into the thing it replaced.

---

## Start a run from the console

```bash
./Run.sh console        # relay + console, nothing else
```

Then open <http://localhost:5273>, press **New run…** (or `N`), pick the clip,
pick the flight log beside it, press **Start run**. The console launches the
real pipeline itself.

`console` is the mode for this. `./Run.sh replay` and `./Run.sh live` each
start a `mission_watcher` of their own, and running one alongside a launched
run puts two watchers on the same `mission.log` — every GID and every log line
then arrives at the console twice.

```
New run… ─► Back/pipeline.js spawns three processes
             │
             ├─ run_pipeline.py    Pipeline_yoloe_logs_robust.py, on your clip
             ├─ mission_watcher.py tails the run folder it creates
             └─ telem_csv.py       replays the CSV as telemetry
```

### The file picker browses the server, not your machine

`<input type="file">` cannot work here. A browser hands back a sandboxed Blob
and a bare filename — never a path — and the process that has to open the clip
is the pipeline, on the box with the GPU. The picker therefore lists the
relay's own filesystem (`GET /media`), confined to `MEDIA_ROOTS`, and what
crosses the wire is a path. Every path is resolved and then required to sit
inside a root, so `../../etc` is refused rather than served.

Choosing the video preselects a CSV sitting next to it — these recordings are
written as a pair, `clip.avi` beside `clip.csv` — so the log does not have to
be found twice in the same folder.

### The pipeline is still unmodified

`run_pipeline.py` reads the pipeline's source, rewrites the configuration
constants at the top of it *in memory*, and executes the result as `__main__`
with `__file__` pointed at the real path — so `BASE_DIR`, the sibling imports
(`coordinate_transformer`, `robust_detect`) and everything else resolve exactly
as they do when it is run directly.

| Constant | Set to |
|---|---|
| `VIDEO_SOURCE` / `VIDEO_PATH` | the clip you picked |
| `TELEM_SOURCE` / `TELEM_CSV_PATH` | the CSV you picked |
| `START_FRAME` | the dialog's start frame |
| `DISPLAY_MODE` / `STREAM_PORT` | `"stream"` — this is the Geolocalisation raw feed |
| `YOLOE_WEIGHTS`, `REID_ONNX_PATH` | `YOLOE_WEIGHTS`, `REID_CKPT` from `.env` |
| `RUN_FOLDER` | redirected into `MISSION_ROOT`, so the relay can serve the crops |

Editing the file instead would mean the pipeline being graded is not the
pipeline on disk, two operators could not launch different clips from one
checkout, and a Ctrl-C would leave a half-applied edit behind. Every override
is asserted to have matched, so a constant renamed upstream fails loudly at
launch rather than running the whole clip against the path the file happened
to be holding.

### Telemetry follows the pipeline, not the wall clock

`telem_csv.py` emits the same `telemetry` payload `telem_worker.py` does, so
the map's drone marker, its trail and every row of the System pane light up
with no frontend change. What differs is which row is "now".

Not the wall clock. The pipeline on a CPU box runs at a fraction of real time,
so a drone flown at wall-clock speed would be a minute ahead of the frame being
processed — and every casualty pin would drop far behind a marker that had
already flown past it.

So the pipeline's own progress is the clock. It writes exactly one

```
[FRAME  ] FPS=12.3
```

per frame it finishes, and its `frame_id` starts at the frame the capture
seeked to and increments once per such line. Counting them in `mission.log`
gives the frame being processed — no change to the pipeline, and nothing to
keep in sync but a file it is already writing. That frame then indexes the CSV
directly through its `frame` column, which is an exact mapping rather than two
clocks that were never synchronised.

The System pane says `flight log · in step with the pipeline` and shows the
attitude and frame number, because a replayed flight and an aircraft that is
actually airborne must never look the same on a wall display.

`--realtime` plays a log at wall-clock speed with no pipeline attached.

### Without a CSV

The dialog says so before you start, and means it: with no flight log the
pipeline has no fix and no attitude for the frame it is looking at, so no
detection can be projected to the ground. No casualty pins, no geographic gate
on re-identification, and nothing to fly the drone marker. Detections and crops
still work.

### Paths

All in `Python/.env`, and all logged at relay boot with a warning against any
that is missing — these live on a different machine from the one this is developed on,
and the time to find out a checkpoint is absent is before the demo.

| Key | Meaning |
|---|---|
| `PIPELINE_SCRIPT` | `Pipeline_yoloe_logs_robust.py` |
| `PIPELINE_PY` | interpreter with ultralytics + torch + onnxruntime + sklearn |
| `GUI_PY` | interpreter with python-socketio + opencv |
| `YOLOE_WEIGHTS`, `REID_CKPT` | weights; empty keeps the pipeline's own paths |
| `PIPELINE_STREAM_PORT` | the MJPEG port, and `VITE_PIPELINE_STREAM` must agree |
| `MEDIA_ROOTS` | colon-separated jail for the file picker |

Two interpreters because no single env on either machine has every import:
`work_env` has ultralytics, `gcs_gui` has python-socketio.

**CUDA is not required to launch.** The pipeline picks its own device and runs
on CPU, slowly. Child stdout and stderr stream into the dialog, so an import
that fails is readable there rather than being an exit code.

---

## Live: one video, one CSV, the whole console

```bash
./Live.sh clip.mp4 fcb-20260903-175559.csv
```

One process — `Python/mission_run.py` — reads the clip and the flight log and
drives every pane as it goes:

- the drone **flies the CSV's path** on the map, dragging its trail
- a **casualty pin drops the moment a GID is created**, not at the end
- the **raw feeds** show detections, traversability and depth *on the same frame*
- the notification panel says `Casualty 2 found`, then `Casualty 2 seen again`

There is no watcher and nothing to replay: what you are looking at is the frame
being processed.

```
┌──────────────────────── mission_run.py ────────────────────────┐
│  video ──┬─► detect + track + ReID ──► gid  ──────┐            │
│          │         └─► annotated frame ──► detect │            │
│          └─► MJPEG hub :8090 ─┐                   ├──► relay   │
│  csv   ────► telemetry @ video time ──────────────┘            │
└───────────────────────────────┼────────────────────────────────┘
                                ▼
                depth_worker / seg_worker subprocesses
```

**Why the hub.** The workers could each open the mp4 themselves, but then three
decoders advance independently and the panes show three different moments of the
flight. One decode, published once, keeps all three feeds on the same frame.

**Time is video time, not wall time.** Telemetry is emitted for the timestamp of
the frame being processed, so on a CPU box running at a fifth of real speed the
drone still sits exactly where it was when that frame was captured. `--realtime`
switches to wall-clock pacing, dropping frames to keep up.

### Where a casualty pin actually goes

Not where the drone was — where the *person* is. Each detection is projected
through the camera intrinsics and the logged attitude onto the ground, reusing
the pipeline's own calibrated `coordinate_transformer.py` rather than a second
copy free to drift from it.

This is load-bearing, not a refinement. Pinning casualties at the drone's own
fix puts every pin on the flight path, and — because a moving drone sees the
same casualty from a different place each time — makes every sighting look like
a different location, so the geo gate can never recognise a return and one
person becomes a string of GIDs. On a two-person test clip that is the
difference between **9 GIDs and 2**.

`mission.log` ends with the coverage, so a run where the projection did not fire
says so rather than quietly pinning everything under the aircraft:

```
[INFO   ] Detections located: 94/94 (100% projected to ground)
```

Pass `--no-geo` to fall back deliberately.

### Re-identification

A finished track is matched against existing casualties on **appearance first,
geography as a gate** — the pipeline's own order. A track that looks like an
existing GID *and* lies within a few metres of it is that person coming back
into frame; the same clothing forty metres away is someone else.

```bash
./Live.sh clip.mp4 fcb.csv --reid ../mobileclip2_b.ts --reid-thresh 0.62
```

`Live.sh` picks up `mobileclip2_b.ts` from the repo root automatically. Without
any checkpoint, identities are matched on GPS proximity alone — which
`mission.log` states plainly, because it is the difference between recognising a
casualty and guessing from where they are lying.

### Requirements

`mission_run.py` needs `ultralytics` **and** `python-socketio` in ONE
interpreter. On this machine they are in different conda envs:

```bash
~/miniconda3/envs/work_env/bin/pip install "python-socketio[client]"
# or point Live.sh somewhere that has both:
RUN_PY=/path/to/python ./Live.sh clip.mp4 fcb.csv
```

### Telemetry CSV

Column names are matched by alias, so both recording formats in this repo work
as they are:

| Wanted | Accepted |
|---|---|
| time | `t_mono_s`, `timestamp_sec`, `timestamp`, `t`, `time` |
| position | `lat_deg`/`lon_deg`, `drone_lat`/`drone_lon`, `lat`/`lon` |
| altitude | `alt_rel_m`, `drone_altitude_agl`, `rel_alt`, `altitude` |
| attitude | `roll_deg`, `pitch_deg`, `yaw_deg` — **needed for projection** |
| heading | `heading_deg`, `heading`, `yaw_deg` |
| speed | `groundspeed_ms`, `groundspeed` |
| frame | `frame` — an exact video-frame mapping, preferred when present |

A `frame` column beats aligning two clocks that were never synchronised. Without
one, the CSV and the video are aligned at their starts; `--telem-offset` shifts
them by hand.

Rows with no fix are skipped rather than read as 0,0 — which would otherwise
teleport the drone into the Gulf of Guinea.

### Useful flags

```bash
./Live.sh clip.mp4 fcb.csv \
  --stride 10 \          # detect every Nth frame (default 5)
  --realtime \           # hold wall-clock pace, dropping frames
  --speed 2 \            # play video time at 2x
  --min-hits 3 \         # detections before a track counts as a casualty
  --max-frames 900 \     # stop early
  --no-workers            # skip depth/traversability
```

---

## Why it reads artifacts instead of importing the pipeline

`Pipeline_yoloe_logs_robust.py` already writes everything the console needs:

```
mission_logs/<YYYYmmdd_HHMMSS>/
    mission.log              "[OK     ] GID=3 Folder ready -> ..."
    gid_3/
        metadata.txt         identity record, appended once per buffer
        representative.jpg   sharpest crop
        crop_frame_000142.jpg
```

So the watcher reads those files rather than asking the pipeline to publish
anything. Three consequences worth knowing:

- **The pipeline is unmodified.** Nothing in this app can slow it down or crash
  it, which matters when the GUI is the thing being demoed and the pipeline is
  the thing being graded.
- **A run from last week replays exactly like a run happening now.** Same code
  path, same artifacts.
- **`mission.log` is the clock.** A GID is published when its announcement line
  appears in the log:
  ```
  GID=<n> Folder ready -> <path>
  ```
  The pipeline writes that at the *end* of `save_new_gid_artifacts`, so the
  metadata and crops are already on disk — the watcher never reads a
  half-written record, and in replay the GIDs appear at the same point in the
  log they appeared at live.

---

## Architecture

```
  Browser (React 19 + Vite)
      │  ws  /react            http  /artifacts/gid_3/representative.jpg
      ▼
  Back/  Node 22 + Socket.io  ──────────────► serves RUN_FOLDER (confined)
      ▲  /py
      │
  ┌───┴──────────────┬────────────────┬────────────────┬──────────────┐
  │ mission_watcher  │ depth_worker   │ seg_worker     │ telem_worker │
  │ tails/replays    │ Depth-Anything │ SegFormer→grid │ pymavlink →  │
  │ mission.log      │ or stub        │ → A* route     │ telemetry    │
  ├──────────────────┼────────────────┴────────────────┴──────────────┤
  │ run_pipeline     │ telem_csv                                      │
  │ the real         │ the flight log as telemetry, stepped by the    │
  │ pipeline, on     │ frame the pipeline is on                       │
  │ your clip        │                                                │
  └──────────────────┴────────────────────────────────────────────────┘

  ...all three of which the relay starts itself when you press New run.

  ...or ONE process doing all of it from a video + a CSV:

  ┌──────────────────────────────────────────────────────────────────┐
  │ mission_run   detect + ReID → gid · annotated frame → detect     │
  │               CSV → telemetry · MJPEG hub → depth/seg workers    │
  └──────────────────────────────────────────────────────────────────┘

  the pipeline's annotated MJPEG feed goes browser ◄── pipeline:8080
  directly, bypassing the relay entirely
```

| Directory | Stack | Purpose |
|---|---|---|
| `Front/` | React 19 + Vite 7 (no UI framework) | The console |
| `Back/` | Node 22 + Express 5 + Socket.io 4 | Relay + artifact server |
| `Python/` | python-socketio + OpenCV + pymavlink | Watcher, model workers, telemetry, live runner |

The watcher and workers need `python-socketio` + `opencv`; `ingest_video.py`
additionally needs `ultralytics`. If no single environment on the machine has
both, `Demo.sh` takes `INGEST_PY` and `PY` separately — it defaults to
`work_env` for the ingest and `gcs_gui` for everything else.

The relay forwards producer events to browsers and browser commands to
producers. Two deliberate differences from the existing relay, both explained in
`Back/index.js`:

1. It keeps a **bounded snapshot** (recent logs, the GID map, the newest frame
   per worker) and pushes it on connect. Telemetry re-arrives at 10 Hz so a
   stateless relay is fine there; a GID may be emitted once, an hour in, so a
   browser reload with no snapshot would show an empty screen for the rest of
   the run.
2. It **serves the run folder over HTTP** so crops are lazy `<img>` requests
   rather than megabytes of base64 through the socket. Every path is resolved
   and then required to sit inside `MISSION_ROOT`.

Telemetry is the one exception to the snapshot's reset-per-run rule: it
describes the aircraft, not the run, so `mission_start` does not clear it.

---

## Run it on a video (batch)

> Batch: this runs the detector over the whole clip **first**, then replays the
> finished run. For the drone to fly its path and pins to drop as they are
> found, use [`./Live.sh`](#live-one-video-one-csv-the-whole-console) instead.

One command, three panes, all showing the same clip:

```bash
./Demo.sh /path/to/clip.mp4
./Demo.sh /path/to/clip.mp4 --telem /path/to/fcb.csv   # real GPS instead of synthesised
./Demo.sh /path/to/clip.mp4 --reuse                    # skip re-ingesting a clip already done
```

It runs `Python/ingest_video.py` over the video once — YOLOE with
casualty-oriented open-vocabulary prompts, IoU tracking across sampled frames —
and writes a run folder in the pipeline's exact format. Then it brings up the
console with the Global ID pane replaying that run and the depth and
traversability workers reading the same video.

The ingest pass is the slow part: roughly 3 s per sampled frame on CPU, so
`--stride 30` over a 1500-frame clip is about three minutes. It is cached per
video under `demo_runs/<clip-name>/`, so `--reuse` makes later runs instant.

**What is real, and what is not.** This is the fallback path for when the full
pipeline cannot run — no ReID checkpoint, no CUDA, no telemetry, or no time.

| | |
|---|---|
| Detections | **Real** — YOLOE on your actual frames |
| Crops, confidence, sharpness, frame numbers | **Real** — measured from those detections |
| Tracks | **Real** — IoU association across sampled frames |
| Re-identification | **Absent.** No ReID stage, so a person who leaves frame and returns becomes a *new* GID. Re-identifying them as the same person is the pipeline's entire contribution; this does not attempt it. |
| GPS | Interpolated from `--telem` if given, otherwise **synthesised** (and marked as such in `mission.log`) |
| Depth / traversability | Stub backends until you pass `--backend dav2` / `--backend segformer` |

For the real thing — YOLOE **+ ReID + geolocalisation** — see *Live, against a
running pipeline* below.

---

## Quick start

No GPU, no weights, no footage — the console comes up fully populated:

```bash
cd SIH-GUI
(cd Back && npm install)
(cd Front && npm install)
pip install -r Python/requirements.txt      # or use an env that has cv2 + python-socketio

cp Python/.env.example Python/.env          # set MISSION_ROOT for live mode
./Run.sh                                    # generates a demo run and replays it
```

Then open <http://localhost:5273>.

To also light up the depth and traversability panes, give them something to
look at:

```bash
WORKER_SOURCE=/path/to/clip.mp4 ./Run.sh
```

### Against the real pipeline process

This is the real path: the pipeline does detection, ReID and geolocalisation,
and the watcher tails its artifacts as they are written.

Configure the pipeline first — `Pipeline_yoloe_logs_robust.py` has its paths at
the top of the file, currently pointing at `/home/uas/...` from another machine:

```python
YOLOE_WEIGHTS  = "/path/to/yoloe-26x-seg.pt"
REID_ONNX_PATH = "/path/to/epoch_003.pth"     # the ReID checkpoint
VIDEO_SOURCE   = "video"
VIDEO_PATH     = "/path/to/clip.mp4"
TELEM_SOURCE   = "csv"
TELEM_CSV_PATH = "/path/to/fcb-....csv"       # must cover the clip's timespan
DISPLAY_MODE   = "stream"                     # so the GUI workers can read the feed
```

It needs `ultralytics`, `sklearn`, `onnxruntime` and (realistically) CUDA.
Then:

```bash
# terminal 1 — the pipeline, writing mission_logs/<timestamp>/
cd ReID-Pipeline/YOLOE/testing && python Pipeline_yoloe_logs_robust.py

# terminal 2 — the console, tailing it
cd SIH-GUI && MISSION_ROOT=<.../testing/mission_logs> WORKER_SOURCE=http://localhost:8080/ ./Run.sh live
```

```bash
./Run.sh live                 # newest run under MISSION_ROOT
./Run.sh live /path/to/run    # a specific run folder
```

The watcher picks up a run folder the moment it appears and follows
`mission.log` as it grows, so it can be started before or after the pipeline.

`DISPLAY_MODE = "stream"` does double duty: it is the **Geolocalisation** raw
feed (the browser reads that MJPEG server directly — point
`VITE_PIPELINE_STREAM` at it if the pipeline is on another host) and it also
makes a good source for the depth and traversability workers:

```bash
WORKER_SOURCE=http://<pipeline-host>:8080/ ./Run.sh live
```

**Telemetry.** For a live drone marker and real battery/mode/armed readouts,
set `ENABLE_UAV0=true` in `Python/.env` and `Run.sh` brings up the telemetry
worker alongside everything else:

```bash
# Python/.env
ENABLE_UAV0=true
UAV0_PORT=14550        # or UAV0_ENDPOINT=udpin:0.0.0.0:14550
```

```bash
# or on its own, against SITL
cd Python && python telem_worker.py --endpoint udpin:0.0.0.0:14550
```

Without it nothing breaks: the map tracks the drone from the GPS fixes in
`mission.log` and the system panel shows `NO TELEM`.

### Manually

```bash
cd Back   && npm run start:prod                     # relay      :7100
cd Front  && npm run dev -- --host                  # console    :5273
cd Python && python mission_watcher.py --replay --speed 4
cd Python/workers && python depth_worker.py --source <src>
cd Python/workers && python seg_worker.py   --source <src>
```

---

## The panes

### Mission map

Mapbox, the same token and style toggle as the existing GCS. The drone marker
rotates to heading and drags a breadcrumb trail that fades toward the tail, so
the direction of travel reads without an arrow. Each casualty gets a numbered
pin at its DBSCAN-medoid GPS, in that casualty's identity colour; a merge moves
the pin as the medoid is refined.

**Follow** keeps the map centred on the drone and is on by default — on a wall
display nobody is panning, and a drone that flies off the edge unnoticed is
worse than a map that moves. Any drag hands control back to the operator until
they press it again. **Fit all** frames every casualty found so far.

Without MAVLink telemetry the marker falls back to the newest per-frame GPS fix
in `mission.log`, and the legend says `TRACK FROM LOG` so nobody mistakes a
replay for a live aircraft.

### System

UAV position and aircraft health: altitude, heading, ground speed, battery
(percentage with a bar when the firmware reports capacity, volts otherwise),
armed state, flight mode, and heartbeat interval.

Every field reads `—` when the link is down rather than holding its last value.
A battery frozen at 80% from four minutes ago is worse than no number at all.

### Casualties

One card per casualty: representative crop, DBSCAN-medoid GPS, gallery size,
best confidence and sharpness, mission timestamp. A card flashes when its record
changes — including on a **merge**, because a re-identification is as much news
as a first sighting. Clicking one opens the full identity record and every crop
in the gallery.

### Mission Log  (drawer — `L`)

The pipeline's own log, coloured by its `[PREFIX ]`. Filter by prefix or by
substring. `FRAME` is muted by default — it is one line per frame and it drowns
everything else; the FPS readout in the status bar carries the same information.
In replay mode the play/pause/speed controls live in this pane's header.

### Depth

`--backend stub` is a placeholder (vertical position + texture energy) so the
pane, the relay and the layout could be built before weights landed. It is
labelled as a placeholder in the frame and in the readouts.

`--backend dav2` runs Depth-Anything-V2 through `transformers`. Everything real
is in `_load_dav2` / `_infer_dav2`; the published payload is identical either
way, so nothing downstream changes.

### Traversability & Route

Three stages, and only the first is a placeholder:

1. **Segment** — `--backend stub` (colour + texture heuristic) or
   `--backend segformer` (your `SegFormer/segformer_best.pth`).
2. **Cost grid** — the mask downsampled with `INTER_AREA`, so a cell is free
   only if most of it is free. Free cells are weighted by distance to the
   nearest obstacle, so the planner prefers the middle of an open route over
   scraping a wall. Real.
3. **Plan** — A* over that grid. 8-connected, octile heuristic, no
   corner-cutting between diagonal obstacles. Real.

Click the frame to re-plan: **Start** sets the responder, **Goal** sets the
casualty, **Auto** returns to the automatic route. A goal on non-traversable
ground snaps to the nearest free cell — a casualty lying on rubble is the
scenario, not an error.

**Known limitation:** the grid is in *image* space, not world space. Routing to
a GID's lat/lon needs the pipeline's `CoordinateTransformer` plus that frame's
telemetry; until that is wired, the goal comes from the UI or `--goal`. The
"Route to" button in the GID drawer sends the GID and is the hook for it.

---

## Swapping a stub for a real model

Both workers subclass `BaseWorker` (`Python/workers/base.py`), which owns the
source handling, throttling, encoding, reconnection and command dispatch.
A subclass implements exactly two methods:

```python
class MyWorker(BaseWorker):
    NAME = "depth"                 # the pane key the frontend listens on

    def load(self):                # once, before the loop
        self.model = ...

    def infer(self, frame):        # per frame
        return vis_bgr, {"anything": "the pane needs"}
```

So switching backends is a change to `load()` and one `_infer_*` method.

---

## Configuration

**`Python/.env` is the only file with a path in it.** The relay, the producers
and the browser bundle all read it — Vite's `envDir` points there
(`Front/vite.config.ts`), so a `VITE_*` key set here reaches the console
instead of being silently ignored in a second file that does not exist. Moving
this console to another machine, or following the pipeline through a
reorganisation, is an edit to this one file.

Everything that is a path:

| Key | Meaning |
|---|---|
| `PIPELINE_SCRIPT` | `Pipeline_yoloe_logs_robust.py` — wherever it lives now |
| `PIPELINE_PY` | interpreter with ultralytics + torch + onnxruntime + sklearn |
| `GUI_PY` | interpreter with python-socketio + opencv |
| `YOLOE_WEIGHTS` | YOLOE `.pt`; empty keeps whatever the pipeline hardcodes |
| `REID_CKPT` | ReID checkpoint; empty keeps the pipeline's own |
| `MISSION_ROOT` | where run folders are written, and the artifact jail the relay serves crops from |
| `MEDIA_ROOTS` | colon-separated jail the file picker may browse |

Everything else:

| Key | Meaning |
|---|---|
| `GUI_PORT` | Relay port (default 7100) |
| `GUI_URL` | Where producers reach the relay |
| `PIPELINE_STREAM_PORT` | The pipeline's MJPEG port. Set once — the relay hands it to the pipeline it launches *and* reports it to the browser, so the Geolocalisation feed cannot end up pointed at the wrong port |
| `PIPELINE_STREAM_URL` | Only when the pipeline is on a different host from the browser |
| `ALLOWED_ORIGINS` | Comma-separated; unset allows all |
| `DEPTH_BACKEND` / `SEG_BACKEND` | `stub`, or a real backend |
| `ENABLE_UAV0` | Connect `telem_worker.py` to a real vehicle at all |
| `UAV0_PORT` | UDP port on localhost; `UAVn` defaults to `14550 + n*10` |
| `UAV0_ENDPOINT` | Full pymavlink connection string; wins over the port |
| `VITE_MAPBOX_TOKEN` | Mapbox token; set it in `Front/.env` |
| `VITE_RELAY_URL` | Only when the built bundle is served somewhere other than the relay |

Only `VITE_`-prefixed keys reach the browser bundle; the interpreters, the
weights and `MEDIA_ROOTS` stay server-side.

The telemetry names match the existing GCS's `Python/.env` exactly, so one convention
covers both consoles.

### What is NOT in it

Two other entry points predate the launcher and still take their paths on the
command line, because they are not launched from the console:

- `./Live.sh clip.mp4 fcb.csv` and `./Demo.sh clip.mp4` — the GUI's own
  detector (`mission_run.py`), not the pipeline. `RUN_PY=` / `PY=` override
  the interpreters; `YOLOE_WEIGHTS` from this same `.env` is picked up.
- `scripts/` — standalone research scripts with their own hardcoded paths.
  Nothing in the console runs them.

The relay serves HTTPS if `Back/server.key` and `Back/server.cert` exist, plain
HTTP otherwise. In dev, Vite proxies `/socket.io` and `/artifacts` to the relay
so the browser stays on one origin — no CORS preflight on the crops and no
certificate warning from a self-signed relay.

---

## Notes

- Log retention is bounded in both places (2000 lines in the relay, 5000 in the
  browser, 1200 rows in the DOM). A four-hour mission emits hundreds of
  thousands of lines, and an unbounded array of them is the one thing certain to
  kill the tab mid-demo.
- Worker frames go over the socket as base64 JPEG rather than a second MJPEG
  port: one connection, one reconnect path, and at 3–5 fps the overhead does not
  show. For full frame rate, point a pane at the pipeline's own MJPEG server.
- `Python/make_demo_run.py` generates a synthetic run folder in the exact
  on-disk format — including the in-place FINAL GPS block and the append-only
  buffer records — so anything that handles it handles a real run.
