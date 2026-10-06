// Visual language, carried over from the existing GCS so the two consoles read as
// one system. Same values are mirrored as CSS variables in index.css.

export const theme = {
  bg: "#000000",
  surface0: "#0a0a0a",
  surface1: "#141414",
  surface2: "#1c1c1c",
  surface3: "#262626",
  border: "#242424",

  primary: "#2b4d5c",
  primaryHover: "#39647a",
  accent: "#4fc3f7",

  success: "#00e676",
  danger: "#ff1744",
  warning: "#ffb300",

  text: "#f2f4f6",
  textDim: "#8a929a",
  textFaint: "#4a5158",

  radius: 8,
  radiusLg: 12,

  fontMono: "ui-monospace, 'Cascadia Code', 'Fira Code', monospace",
} as const;

// mission.log prefixes -> colour. The pipeline writes one of these in every
// line ("[REID   ] ..."), so the log pane can colour by source with no parsing
// beyond the prefix the watcher already split off.
export const LEVEL_COLORS: Record<string, string> = {
  OK: theme.success,
  INFO: theme.textDim,
  WARN: theme.warning,
  ERROR: theme.danger,
  DET: "#ffee58",
  REID: "#ba68c8",
  TRACK: "#4fc3f7",
  GPS: "#69f0ae",
  MATCH: "#ff9100",
  FRAME: theme.textFaint,
  RAW: theme.textFaint,
};

export const levelColor = (level: string) => LEVEL_COLORS[level] ?? theme.textDim;

// Per-GID identity colour. A GID is an arbitrary integer, so hash it around the
// hue wheel rather than keeping a fixed table that runs out.
export const gidColor = (gid: number) => `hsl(${(gid * 137.508) % 360}deg 72% 62%)`;
