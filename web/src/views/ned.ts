/**
 * NED-frame map visualization.
 * Faithful port of milliampere_xai/rendering/_ned.py (NedRender).
 *
 * Coordinate convention (NED viewport at psi=0, scale S, center cx/cy):
 *   world (wx, wy) -> pixel: px = wy*S + cx,  py = wx*S + cy
 *   North (positive wx) goes DOWN on screen, East (positive wy) goes RIGHT.
 */

import { Colors } from "../colors.ts";
import {
  vesselHullPolygon,
  vesselTrianglePolygon,
} from "../geometry.ts";
import type { Renderer } from "../renderer.ts";
import type { RenderFrame } from "../types.ts";

const SCALE = 25; // pixels per meter

// ---------------------------------------------------------------------------
// Viewport helpers (inline, matching Python Viewport.from_window at psi=0)
// ---------------------------------------------------------------------------

/** World (wx, wy) -> local pixel within the sub-window. */
function worldToPixel(
  wx: number,
  wy: number,
  scale: number,
  cx: number,
  cy: number,
): [number, number] {
  // R = [[0, scale], [-scale, 0]]  T = [[cy], [cx]]   (Python convention)
  // But the actual TypeScript R2(0,s) gives [0,s,s,0] with T=[cx,cy]
  // => px = s*wy + cx,  py = s*wx + cy
  return [scale * wy + cy, -scale * wx + cx];
}

// ---------------------------------------------------------------------------
// Grid drawing (matches Python NedRender._draw_grid)
// ---------------------------------------------------------------------------

function drawGrid(
  r: Renderer,
  ox: number,
  oy: number,
  w: number,
  h: number,
  scale: number,
  cx: number,
  cy: number,
  targetPose: [number, number, number],
  spacing: number,
  color: string,
  lineWidth: number,
): void {
  const px = spacing * scale; // pixels per grid cell

  // Fractional pixel offset inside the grid cell from target_pose
  const offX = (targetPose[0] * scale) % px;
  const offY = (targetPose[1] * scale) % px;

  // Python: start_x = cx + off_x, then x = start_x - floor(start_x/px)*px
  let startX = cx + offX;
  let x = startX - Math.floor(startX / px) * px;
  if (x > px) x -= px;

  // Python: start_y = cy - off_y, then y = start_y - floor(start_y/px)*px
  let startY = cy - offY;
  let y = startY - Math.floor(startY / px) * px;
  if (y > px) y -= px;

  // Draw "vertical" lines (Python loops y < W, draws (y,0)-(y,H))
  while (y < w) {
    r.drawLine(ox + y, oy, ox + y, oy + h, color, lineWidth);
    y += px;
  }

  // Draw "horizontal" lines (Python loops x < H, draws (0,x)-(W,x))
  while (x < h) {
    r.drawLine(ox, oy + x, ox + w, oy + x, color, lineWidth);
    x += px;
  }
}

// ---------------------------------------------------------------------------
// Shape rotation helper
// ---------------------------------------------------------------------------

/** Rotate points by angle_rad (standard 2D rotation), translate by (tx,ty). */
function rotateAndTranslate(
  points: [number, number][],
  angleRad: number,
  tx: number,
  ty: number,
): [number, number][] {
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);
  return points.map(([sx, sy]) => [
    cos * sx - sin * sy + tx,
    sin * sx + cos * sy + ty,
  ]);
}

// ---------------------------------------------------------------------------
// Target drawing (dashed hull outline)
// ---------------------------------------------------------------------------

/**
 * Draw target at (0,0) rotated by target heading.
 * Matches Python: draw_target(0, 0, target_heading, ned=True)
 *   - ned=True: shape = _transform_vessel(0, 0, psi, shape) which rotates
 *     hull by heading, circle stays at origin, triangle rotated by heading.
 *
 * Python _transform_vessel(x, y, psi, shape) = transform_shape(x, y, psi, shape)
 *   = (R2(psi).T @ shape.T + T2(x,y)).T
 * At (0,0): just rotates shape by psi (standard rotation).
 */
function drawTarget(
  r: Renderer,
  ox: number,
  oy: number,
  scale: number,
  cx: number,
  cy: number,
  psiDeg: number,
  color: string,
): void {
  const hull = vesselHullPolygon();
  const tri = vesselTrianglePolygon();
  const angleRad = (psiDeg * Math.PI) / 180;

  // Rotate hull and triangle by heading around origin
  const rotatedHull = rotateAndTranslate(hull, angleRad, 0, 0);
  const rotatedTri = rotateAndTranslate(tri, angleRad, 0, 0);

  // Convert to pixels
  const hullPx: [number, number][] = rotatedHull.map(([wx, wy]) => {
    const [px, py] = worldToPixel(wx, wy, scale, cx, cy);
    return [ox + px, oy + py];
  });

  const triPx: [number, number][] = rotatedTri.map(([wx, wy]) => {
    const [px, py] = worldToPixel(wx, wy, scale, cx, cy);
    return [ox + px, oy + py];
  });

  // Dashed hull outline
  for (let i = 0; i < hullPx.length; i++) {
    const j = (i + 1) % hullPx.length;
    r.drawDashedLine(
      hullPx[i][0], hullPx[i][1],
      hullPx[j][0], hullPx[j][1],
      color, 10, 5, 2,
    );
  }

  // Dashed triangle outline
  for (let i = 0; i < triPx.length; i++) {
    const j = (i + 1) % triPx.length;
    r.drawDashedLine(
      triPx[i][0], triPx[i][1],
      triPx[j][0], triPx[j][1],
      color, 5, 5, 2,
    );
  }

  // Circle at center-of-gravity (origin in world)
  const [circlePx, circlePy] = worldToPixel(0, 0, scale, cx, cy);
  r.drawCircle(ox + circlePx, oy + circlePy, 3, color);
}

// ---------------------------------------------------------------------------
// Vessel drawing (solid hull at NED offset)
// ---------------------------------------------------------------------------

/**
 * Draw vessel at world position with given heading.
 * Matches Python NedRender._draw_vessel_ned(world_x, world_y, heading_deg).
 *
 * 1. Rotate hull/triangle by heading (standard rotation matrix)
 * 2. Translate by (world_x, world_y)
 * 3. Apply viewport to get pixel coordinates
 */
function drawVessel(
  r: Renderer,
  ox: number,
  oy: number,
  scale: number,
  cx: number,
  cy: number,
  worldX: number,
  worldY: number,
  headingDeg: number,
): void {
  const hull = vesselHullPolygon();
  const tri = vesselTrianglePolygon();
  const angleRad = (headingDeg * Math.PI) / 180;

  // Rotate shape by heading, translate to world position
  const transformedHull = rotateAndTranslate(hull, angleRad, worldX, worldY);
  const transformedTri = rotateAndTranslate(tri, angleRad, worldX, worldY);
  // Circle point is at origin, so rotated = (0,0) + (worldX, worldY)
  const circleWorld: [number, number] = [worldX, worldY];

  // Convert to pixels
  const hullPx: [number, number][] = transformedHull.map(([wx, wy]) => {
    const [px, py] = worldToPixel(wx, wy, scale, cx, cy);
    return [ox + px, oy + py];
  });

  const triPx: [number, number][] = transformedTri.map(([wx, wy]) => {
    const [px, py] = worldToPixel(wx, wy, scale, cx, cy);
    return [ox + px, oy + py];
  });

  const [circlePxX, circlePxY] = worldToPixel(
    circleWorld[0], circleWorld[1], scale, cx, cy,
  );

  // Hull filled blue
  r.drawPolygon(hullPx, Colors.AGENT_BLUE);
  // Bow triangle filled black
  r.drawPolygon(triPx, Colors.BLACK);
  // CoG circle
  r.drawCircle(ox + circlePxX, oy + circlePxY, 3, Colors.BLACK);
}

// ---------------------------------------------------------------------------
// Main render function
// ---------------------------------------------------------------------------

export function renderNed(
  r: Renderer,
  frame: RenderFrame,
  ox: number,
  oy: number,
  w: number,
  h: number,
): void {
  const scale = SCALE;
  const cx = w / 2;
  const cy = h / 2;

  const { target_pose, epsilon_ned } = frame.vessel;

  // Background
  r.drawRect(ox, oy, w, h, Colors.OCEAN_BLUE);

  // Grid pinned to target_pose
  drawGrid(r, ox, oy, w, h, scale, cx, cy, target_pose, 1.0, Colors.OCEAN_GRID, 1);

  // Target at center (0,0), rotated by target heading
  drawTarget(r, ox, oy, scale, cx, cy, target_pose[2], Colors.DESIRED_YELLOW);

  // Vessel: offset from center by NED error
  // error = target - vessel => vessel pos = -error (from target)
  // Python: vessel_heading = target_pose[2] - psi_err where psi_err = epsilon_ned[2]
  const vesselHeading = target_pose[2] - epsilon_ned[2];
  drawVessel(
    r, ox, oy, scale, cx, cy,
    -epsilon_ned[0], -epsilon_ned[1],
    vesselHeading,
  );

  // Title top-right
  r.drawText(
    "Global map in NED-frame",
    ox + w - 5, oy + 5,
    Colors.GRAY, 11, "right",
  );
}
