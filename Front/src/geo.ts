// Ground-truth checking helpers for the casualty list.
//
// The pipeline's own geolocalisation is in Python; this is only the arithmetic
// an operator does by hand when comparing a reported fix against a known one,
// so a sphere is accurate enough — over the few hundred metres a casualty is
// ever off by, the ellipsoid correction is well under the fix's own error.

const R_EARTH_M = 6371008.8;   // IUGG mean radius

const rad = (d: number) => (d * Math.PI) / 180;

/** Great-circle distance between two WGS-84 points, in metres. */
export function haversineMeters(
  lat1: number, lon1: number, lat2: number, lon2: number,
): number {
  const dLat = rad(lat2 - lat1);
  const dLon = rad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(rad(lat1)) * Math.cos(rad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R_EARTH_M * Math.asin(Math.min(1, Math.sqrt(a)));
}

/**
 * Read a latitude/longitude out of whatever an operator pasted: a comma pair,
 * a space-separated pair, or the tab-separated pair that comes off a
 * spreadsheet cell. Returns null rather than a half-parsed point, so the
 * caller can say "not a coordinate" instead of searching from 0,0.
 */
export function parseLatLon(text: string): { lat: number; lon: number } | null {
  const parts = text
    .trim()
    .split(/[\s,;]+/)
    .filter(Boolean)
    .map(Number);
  if (parts.length !== 2 || parts.some((n) => !Number.isFinite(n))) return null;

  const [lat, lon] = parts;
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return { lat, lon };
}

/** Metres, at the precision the number is actually good to. */
export const fmtMeters = (m: number) =>
  m < 10 ? `${m.toFixed(2)} m`
    : m < 1000 ? `${m.toFixed(1)} m`
      : `${(m / 1000).toFixed(2)} km`;
