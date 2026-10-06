import { useMission } from "../context/MissionContext";
import { theme } from "../theme";

const fmtElapsed = (s?: number) => {
  if (s == null) return "--:--:--";
  const t = Math.floor(s);
  return [t / 3600, (t % 3600) / 60, t % 60]
    .map((n) => String(Math.floor(n)).padStart(2, "0"))
    .join(":");
};

const Light = ({ on, label }: { on: boolean; label: string }) => (
  <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
    <span className="dot" style={{ background: on ? theme.success : theme.danger }} />
    {label}
  </span>
);

export default function StatusBar() {
  const { connected, producerConnected, mission, stats, gids } = useMission();

  return (
    <footer className="statusbar">
      <Light on={connected} label="RELAY" />
      <Light on={producerConnected} label="PIPELINE" />

      <span className="sep" />

      <span style={{ color: theme.textFaint }}>RUN</span>
      <span style={{ color: theme.text }}>{mission?.name ?? "—"}</span>
      {mission?.replay && (
        <span className="chip" style={{ color: theme.warning }}>
          REPLAY {stats?.speed ? `${stats.speed}×` : ""}{stats?.paused ? " · PAUSED" : ""}
        </span>
      )}

      <span className="sep" />

      <span style={{ color: theme.textFaint }}>ELAPSED</span>
      <span style={{ color: theme.text }}>{fmtElapsed(stats?.elapsed)}</span>

      <span style={{ color: theme.textFaint }}>GIDS</span>
      <span style={{ color: theme.accent }}>{gids.length}</span>

      <span style={{ color: theme.textFaint }}>FRAMES</span>
      <span style={{ color: theme.text }}>{stats?.frames ?? 0}</span>

      <span style={{ color: theme.textFaint }}>FPS</span>
      <span style={{ color: theme.text }}>
        {stats?.fpsLast != null ? stats.fpsLast.toFixed(1) : "—"}
        {stats?.fpsMin != null && stats?.fpsMax != null && (
          <span style={{ color: theme.textFaint }}>
            {" "}({stats.fpsMin.toFixed(0)}–{stats.fpsMax.toFixed(0)})
          </span>
        )}
      </span>

      <span style={{ color: theme.textFaint }}>LINES</span>
      <span style={{ color: theme.text }}>{stats?.logLines ?? 0}</span>

      <span className="spacer" />
      <span style={{ color: theme.textFaint, overflow: "hidden", textOverflow: "ellipsis" }}>
        {mission?.runFolder ?? ""}
      </span>
    </footer>
  );
}
