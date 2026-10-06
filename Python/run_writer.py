"""
Writes run folders in the pipeline's exact on-disk format.

One writer, used by every producer that needs to create artifacts
(make_demo_run.py, ingest_video.py). The format is not ours — it is whatever
save_new_gid_artifacts() in Pipeline_yoloe_logs_robust.py produces — so it is
defined once here rather than reimplemented per script, where the two copies
would drift and the watcher would start failing on one of them.

Layout produced:

    <root>/<YYYYmmdd_HHMMSS>/
        mission.log
        gid_<n>/
            metadata.txt
            representative.jpg
            crop_frame_<frame:06d>.jpg
"""

import os
from datetime import datetime

import cv2

FINAL_GPS_START = "---- FINAL GPS (DBSCAN medoid, all buffers to date) ----\n"
FINAL_GPS_END = "---------------------------------------------------------\n"


class RunWriter:
    def __init__(self, root, name=None):
        self.root = os.path.abspath(root)
        self.run = os.path.join(self.root, name or datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.run, exist_ok=True)
        self.lines = []

    # -- log ---------------------------------------------------------------

    def log(self, prefix, msg):
        """One mission.log line, in the pipeline's '[PREFIX ] msg' shape."""
        self.lines.append(f"[{prefix:<7}] {msg}")

    def flush_log(self):
        with open(os.path.join(self.run, "mission.log"), "w") as f:
            f.write("=" * 80 + "\nSEARCH & RESCUE PIPELINE\n"
                    + datetime.now().isoformat() + "\n" + "=" * 80 + "\n\n")
            f.write("\n".join(self.lines) + "\n")

    # -- gid ---------------------------------------------------------------

    def gid_folder(self, gid):
        folder = os.path.join(self.run, f"gid_{gid}")
        os.makedirs(folder, exist_ok=True)
        return folder

    def write_buffer(self, gid, event, tid, entries, crops, lat, lon,
                     cumulative, first_frame, ros_time, representative=None):
        """
        Append one buffer record to gid_<n>/metadata.txt and save its crops.

        entries      [(frame, conf, sharpness, lat, lon), ...] for THIS buffer
        crops        {frame: BGR image} for this buffer
        cumulative   gallery size across every buffer so far
        first_frame  earliest frame across every buffer so far

        Emits the same log lines the pipeline does, ending with the
        "Folder ready" line the watcher keys GID publication off — so it must
        be written only after the files are on disk.
        """
        folder = self.gid_folder(gid)
        meta = os.path.join(folder, "metadata.txt")

        block = (FINAL_GPS_START
                 + f"Final GPS Samples   : {cumulative}\n"
                 + f"Final GPS Latitude  : {lat}\n"
                 + f"Final GPS Longitude : {lon}\n"
                 + FINAL_GPS_END)

        if not os.path.exists(meta):
            with open(meta, "w") as f:
                f.write("=" * 52 + "\nGLOBAL IDENTITY RECORD\n" + "=" * 52 + "\n\n")
                f.write(f"GID                 : {gid}\n")
                f.write(f"Created             : {datetime.now().isoformat()}\n")
                f.write(f"Mission Folder      : {self.run}\n\n")
                f.write(block + "\n")
        else:
            # The pipeline rewrites this block in place on every update, so the
            # header always shows the medoid over every buffer to date.
            with open(meta) as f:
                content = f.read()
            s = content.find(FINAL_GPS_START)
            e = content.find(FINAL_GPS_END, s) + len(FINAL_GPS_END)
            with open(meta, "w") as f:
                f.write(content[:s] + block + content[e:])

        secs = ros_time
        stamp = f"{int(secs // 3600):02d}:{int(secs % 3600 // 60):02d}:{secs % 60:06.3f}"

        with open(meta, "a") as f:
            f.write("-" * 52 + "\n")
            f.write(f"BUFFER {event.upper()}   TID={tid}\n")
            f.write("-" * 52 + "\n")
            f.write(f"Wall Time           : {datetime.now().isoformat()}\n")
            f.write(f"ROS Assignment Time : {ros_time:.3f} sec\n")
            f.write(f"Mission Timestamp   : {stamp}\n\n")

            f.write("------------ Gallery Summary (cumulative) ------------\n")
            f.write(f"Buffer Gallery Size : {len(entries)}\n")
            f.write(f"Cumulative Gallery  : {cumulative}\n")
            f.write(f"Buffer Crops Saved  : {len(crops)}\n")
            f.write(f"First Frame         : {first_frame}\n")
            f.write(f"Last Frame          : {max(e[0] for e in entries)}\n")
            f.write(f"Best Confidence     : {max(e[1] for e in entries):.4f}\n")
            f.write(f"Best Sharpness      : {max(e[2] for e in entries):.2f}\n")
            f.write(f"Representative Frame: {entries[0][0]}\n\n")

            f.write("------------ GID Medoid @ this update ------------\n")
            f.write(f"GPS Samples (cum.)  : {cumulative}\n")
            f.write(f"GID Medoid Latitude : {lat}\n")
            f.write(f"GID Medoid Longitude: {lon}\n\n")

            f.write("------------ This Buffer's Entries ------------\n")
            f.write("Frame      Conf      Sharpness      Latitude      Longitude\n")
            for fr, conf, sharp, plat, plon in sorted(entries):
                f.write(f"{fr:06d}    {conf:.4f}    {sharp:8.2f}    {plat}    {plon}\n")
            f.write("\n")

        saved = 0
        for fr, img in crops.items():
            if img is not None and img.size and cv2.imwrite(
                os.path.join(folder, f"crop_frame_{fr:06d}.jpg"), img
            ):
                saved += 1

        if representative is not None and representative.size:
            cv2.imwrite(os.path.join(folder, "representative.jpg"), representative)

        self.log("REID", f"buffer flush tid={tid} size={len(entries)} -> gid={gid} ({event})")
        self.log("GPS", f"GID={gid} medoid=({lat}, {lon}) samples={cumulative}")
        self.log("OK", f"GID={gid} metadata appended ({event}, buffer_gps={len(entries)}).")
        self.log("OK", f"GID={gid} Saved {saved}/{len(crops)} buffer crops.")
        # Watcher keys GID publication off exactly this line — it goes last.
        self.log("OK", f"GID={gid} Folder ready -> {folder}")
