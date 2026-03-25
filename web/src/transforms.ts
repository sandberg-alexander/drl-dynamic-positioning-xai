/**
 * Coordinate transforms ported from milliampere_dp/rendering/transforms.py.
 *
 * NED (North-East-Down) world frame -> screen pixel coordinates.
 * The display convention rotates NED 90° clockwise so North points up
 * on screen (x-screen = East, y-screen = -North).
 */

/** 2x2 rotation-scale matrix for NED->screen. */
export function R2(psiRad: number, scale: number): [number, number, number, number] {
  const cos = Math.cos(psiRad) * scale;
  const sin = Math.sin(psiRad) * scale;
  // [cos, sin; -sin, cos] rotated 90° CW for display
  return [sin, cos, cos, -sin];
}

/** Translation vector (screen center offset). */
export function T2(cx: number, cy: number): [number, number] {
  return [cx, cy];
}

/** Apply affine transform: result = R @ coord + T */
export function applyTransform(
  r: [number, number, number, number],
  t: [number, number],
  x: number,
  y: number,
): [number, number] {
  return [r[0] * x + r[1] * y + t[0], r[2] * x + r[3] * y + t[1]];
}

/** Transform an array of points. */
export function transformPoints(
  r: [number, number, number, number],
  t: [number, number],
  points: [number, number][],
): [number, number][] {
  return points.map(([x, y]) => applyTransform(r, t, x, y));
}

/**
 * Viewport encapsulating NED-to-pixel affine transform.
 * Mirrors Python Viewport.from_window().
 */
export class Viewport {
  readonly scale: number;
  readonly centerX: number;
  readonly centerY: number;
  readonly r: [number, number, number, number];
  readonly t: [number, number];

  constructor(scale: number, width: number, height: number) {
    this.scale = scale;
    this.centerX = width / 2;
    this.centerY = height / 2;
    // Default heading = 0 for viewport
    this.r = R2(0, scale);
    this.t = T2(this.centerX, this.centerY);
  }

  /** World meters -> pixel coords. */
  worldToPixels(x: number, y: number): [number, number] {
    return applyTransform(this.r, this.t, x, y);
  }

  /** Meters -> pixels (scalar). */
  metersToPixels(value: number): number {
    return this.scale * value;
  }
}

/**
 * Transform vessel shape (rotate by heading + translate to position).
 * Matches Python transform_shape(). Accepts heading in degrees.
 */
export function transformShape(
  x: number,
  y: number,
  psiDeg: number,
  shape: [number, number][],
  scale: number,
  cx: number,
  cy: number,
): [number, number][] {
  const psiRad = (psiDeg * Math.PI) / 180;
  const r = R2(psiRad, scale);
  const t = T2(cx, cy);
  // First rotate shape by heading, then translate to world position
  const cos = Math.cos(psiRad);
  const sin = Math.sin(psiRad);
  return shape.map(([sx, sy]) => {
    // Rotate shape point by vessel heading
    const rx = cos * sx - sin * sy + x;
    const ry = sin * sx + cos * sy + y;
    // Then apply viewport transform
    return applyTransform(r, t, rx, ry);
  });
}

/**
 * Degrees -> canvas angle convention.
 * Canvas: 0 = right, increases clockwise.
 * NED: 0 = north, increases clockwise.
 */
export function degreesToCanvas(angleDeg: number): number {
  return ((angleDeg - 90) * Math.PI) / 180;
}
