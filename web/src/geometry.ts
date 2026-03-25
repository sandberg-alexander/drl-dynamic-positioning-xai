/**
 * Vessel geometry ported from milliampere_dp/rendering/geometry.py
 * and milliampere_dp/vessel.py constants.
 */

// --- Vessel physical constants (from milliampere_dp.vessel) ---
export const VESSEL_LENGTH = 5.06;
export const VESSEL_BEAM = 2.86;
export const THRUSTER_ARM_X = 1.8;
export const THRUSTER_ARM_Y = 0.8;

// --- Rendering constants ---
export const VESSEL_CORNER = 0.5;
export const VESSEL_TRIANGLE = 0.4;
export const VESSEL_MOMENT_MARKER = 0.6;

/**
 * Generate the 8-point vessel hull polygon with rounded corners.
 * Returns points centered at origin in vessel body frame.
 */
export function vesselHullPolygon(
  length = VESSEL_LENGTH,
  beam = VESSEL_BEAM,
  corner = VESSEL_CORNER,
): [number, number][] {
  const hl = length / 2;
  const hb = beam / 2;
  return [
    [hl - corner, -hb],
    [hl, -hb + corner],
    [hl, hb - corner],
    [hl - corner, hb],
    [-hl + corner, hb],
    [-hl, hb - corner],
    [-hl, -hb + corner],
    [-hl + corner, -hb],
  ];
}

/**
 * Forward-pointing triangle indicator.
 */
export function vesselTrianglePolygon(
  length = VESSEL_LENGTH,
  tri = VESSEL_TRIANGLE,
): [number, number][] {
  const hl = length / 2;
  return [
    [hl, 0],
    [hl - tri, -tri / 2],
    [hl - tri, tri / 2],
  ];
}

/**
 * Center-of-gravity moment marker line.
 */
export function vesselMomentMarkerLine(
  marker = VESSEL_MOMENT_MARKER,
): [number, number][] {
  // Python: [[0, 0], [marker, 0]] — starts at CoG, goes forward
  return [
    [0, 0],
    [marker, 0],
  ];
}
