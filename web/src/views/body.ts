/**
 * Body-frame vessel visualization with thrusters and velocities.
 * Faithful port of milliampere_xai/rendering/_body.py (BodyRender).
 *
 * Coordinate convention (viewport at psi=0, scale S, center cx/cy):
 *   world (wx, wy) -> pixel: px = wy*S + cx,  py = wx*S + cy
 */

import { Colors } from "../colors.ts";
import {
  THRUSTER_ARM_X,
  THRUSTER_ARM_Y,
  VESSEL_MOMENT_MARKER,
  vesselHullPolygon,
  vesselMomentMarkerLine,
  vesselTrianglePolygon,
} from "../geometry.ts";
import type { Renderer } from "../renderer.ts";
import type { RenderFrame } from "../types.ts";

const SCALE = 50; // pixels per meter

const ACTUATOR_POSITIONS: [number, number][] = [
  [THRUSTER_ARM_X, -THRUSTER_ARM_Y],
  [THRUSTER_ARM_X, THRUSTER_ARM_Y],
  [-THRUSTER_ARM_X, THRUSTER_ARM_Y],
  [-THRUSTER_ARM_X, -THRUSTER_ARM_Y],
];

// Compass parameters (matches Python _make_compass_base(radius=30))
const COMPASS_RADIUS = 30;
const COMPASS_LETTER_OFFSET = 15;
const COMPASS_TRI_H = 30;
const COMPASS_TRI_W = 8;

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
  return [scale * wy + cy, -scale * wx + cx];
}

/** Scalar meters -> pixels. Matches Python _scalar2pygame. */
function metersToPixels(scale: number, value: number): number {
  return scale * value;
}

/** Degrees -> canvas angle. Matches Python degrees_to_pygame. */
function degreesToCanvas(angleDeg: number): number {
  return (Math.PI * angleDeg) / 180 - Math.PI / 2;
}

// ---------------------------------------------------------------------------
// Shape rotation helper
// ---------------------------------------------------------------------------

/**
 * Python transform_shape(x, y, psi_deg, shape):
 *   psi_rad = deg2rad(psi_deg)
 *   return (R2(psi_rad).T @ shape.T + T2(x,y)).T
 *
 * R2.T is a standard 2D rotation: [[cos, -sin], [sin, cos]]
 */
function transformShape(
  x: number,
  y: number,
  psiDeg: number,
  shape: [number, number][],
): [number, number][] {
  const psiRad = (psiDeg * Math.PI) / 180;
  const cos = Math.cos(psiRad);
  const sin = Math.sin(psiRad);
  return shape.map(([sx, sy]) => [
    cos * sx - sin * sy + x,
    sin * sx + cos * sy + y,
  ]);
}

// ---------------------------------------------------------------------------
// Body grid (matches Python VesselRender._draw_body_grid)
// ---------------------------------------------------------------------------

function drawBodyGrid(
  r: Renderer,
  ox: number,
  oy: number,
  w: number,
  h: number,
  scale: number,
  vpCx: number,
  vpCy: number,
  xErr: number,
  yErr: number,
  psiErr: number,
  targetPose: [number, number, number],
  spacing: number,
  color: string,
  lineWidth: number,
): void {
  // Step 0: Pre-compute
  // circle = transform_shape(x_err, y_err, psi_err, [[0,0]]) = (x_err, y_err)
  // cx, cy = _world_2_pixels(circle)
  const [cx, cy] = worldToPixel(xErr, yErr, scale, vpCx, vpCy);

  const W = w;
  const H = h;
  const s = scale;
  const px = spacing * s; // pixels per grid cell

  // x_n, y_n = _world_2_pixels(target_pose[:2])
  const [xN, yN] = worldToPixel(targetPose[0], targetPose[1], scale, vpCx, vpCy);

  const xP = xN - 25 - Math.trunc(xN / px) * px;
  const yP = yN - 25 - Math.trunc(yN / px) * px;

  // Step 1: Build unrotated line offsets
  const horiz: number[] = [];
  const vert: number[] = [];

  // Horizontal lines
  const y0 = cy - yP;
  horiz.push(y0);
  let dy = px;
  while (y0 + dy <= H) {
    horiz.push(y0 + dy);
    dy += px;
  }
  dy = px;
  while (y0 - dy >= 0) {
    horiz.push(y0 - dy);
    dy += px;
  }

  // Vertical lines
  const x0 = cx - xP;
  vert.push(x0);
  let dx = px;
  while (x0 + dx <= W) {
    vert.push(x0 + dx);
    dx += px;
  }
  dx = px;
  while (x0 - dx >= 0) {
    vert.push(x0 - dx);
    dx += px;
  }

  // Step 2: Rotate every line about (cx, cy) by (psiErr - target heading)
  const ang = ((psiErr - targetPose[2]) * Math.PI) / 180;
  const cosA = Math.cos(ang);
  const sinA = Math.sin(ang);
  const diag = Math.hypot(W, H);

  // Rotate a point around (cx, cy)
  function rot(ptX: number, ptY: number): [number, number] {
    const ddx = ptX - cx;
    const ddy = ptY - cy;
    return [cx + ddx * cosA - ddy * sinA, cy + ddx * sinA + ddy * cosA];
  }

  // Draw horizontal (world-Y) lines
  for (const yy of horiz) {
    const [x1, y1] = rot(-diag, yy);
    const [x2, y2] = rot(diag, yy);
    r.drawLine(ox + x1, oy + y1, ox + x2, oy + y2, color, lineWidth);
  }

  // Draw vertical (world-X) lines
  for (const xx of vert) {
    const [x1, y1] = rot(xx, -diag);
    const [x2, y2] = rot(xx, diag);
    r.drawLine(ox + x1, oy + y1, ox + x2, oy + y2, color, lineWidth);
  }
}

// ---------------------------------------------------------------------------
// Target drawing (dashed hull outline at error offset)
// ---------------------------------------------------------------------------

/**
 * Matches Python draw_target(x_tilde, y_tilde, psi_tilde, ned=False).
 * ned=False: transforms all shapes by (x,y,psi).
 */
function drawTarget(
  r: Renderer,
  ox: number,
  oy: number,
  scale: number,
  cx: number,
  cy: number,
  xErr: number,
  yErr: number,
  psiErr: number,
  color: string,
): void {
  const hull = vesselHullPolygon();
  const tri = vesselTrianglePolygon();

  const transformedHull = transformShape(xErr, yErr, psiErr, hull);
  const transformedTri = transformShape(xErr, yErr, psiErr, tri);
  const transformedCircle = transformShape(xErr, yErr, psiErr, [[0, 0]]);

  // Convert to pixels
  const hullPx: [number, number][] = transformedHull.map(([wx, wy]) => {
    const [px, py] = worldToPixel(wx, wy, scale, cx, cy);
    return [ox + px, oy + py];
  });

  const triPx: [number, number][] = transformedTri.map(([wx, wy]) => {
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
    // Dashed triangle (only for valid triangle indices)
    if (i < triPx.length) {
      const k = (i + 1) % triPx.length;
      r.drawDashedLine(
        triPx[i][0], triPx[i][1],
        triPx[k][0], triPx[k][1],
        color, 5, 5, 2,
      );
    }
  }

  // Circle at transformed center
  const [cirPx, cirPy] = worldToPixel(
    transformedCircle[0][0], transformedCircle[0][1], scale, cx, cy,
  );
  r.drawCircle(ox + cirPx, oy + cirPy, 3, color);
}

// ---------------------------------------------------------------------------
// Vessel drawing (solid hull at center, body frame)
// ---------------------------------------------------------------------------

/**
 * Matches Python draw_vessel(x_tilde, y_tilde, psi_tilde, ned=False).
 * ned=False: draws raw shape at origin with no transform.
 * Also draws the moment marker line (not drawn in NED mode).
 */
function drawVessel(
  r: Renderer,
  ox: number,
  oy: number,
  scale: number,
  cx: number,
  cy: number,
): void {
  const hull = vesselHullPolygon();
  const tri = vesselTrianglePolygon();
  const momentLine = vesselMomentMarkerLine();

  const toScreen = (pts: [number, number][]): [number, number][] =>
    pts.map(([wx, wy]) => {
      const [px, py] = worldToPixel(wx, wy, scale, cx, cy);
      return [ox + px, oy + py];
    });

  // Hull filled blue
  r.drawPolygon(toScreen(hull), Colors.AGENT_BLUE);
  // Bow triangle filled black
  r.drawPolygon(toScreen(tri), Colors.BLACK);
  // CoG circle
  const [cirPx, cirPy] = worldToPixel(0, 0, scale, cx, cy);
  r.drawCircle(ox + cirPx, oy + cirPy, 3, Colors.BLACK);
  // Moment marker line
  const linePx = toScreen(momentLine);
  r.drawLine(linePx[0][0], linePx[0][1], linePx[1][0], linePx[1][1], Colors.BLACK, 1);
}

// ---------------------------------------------------------------------------
// Actuator arrows
// ---------------------------------------------------------------------------

function drawActuatorRef(
  r: Renderer,
  ox: number,
  oy: number,
  scale: number,
  cx: number,
  cy: number,
  actuatorRef: [number, number][],
): void {
  for (let i = 0; i < ACTUATOR_POSITIONS.length; i++) {
    const [x, y] = ACTUATOR_POSITIONS[i];
    const [lpx, lpy] = worldToPixel(x, y, scale, cx, cy);
    const px = ox + lpx;
    const py = oy + lpy;

    const ref = actuatorRef[i];
    if (!ref) continue;

    const angle = degreesToCanvas(ref[1]);
    const perpAngle = angle + Math.PI / 2;
    const arrowLen = metersToPixels(scale, ref[0]);

    // Thrust arrow
    r.drawArrow(px, py, angle, arrowLen, Colors.DESIRED_LIGHT_YELLOW);

    // Perpendicular base line
    r.drawLine(
      px - 10 * Math.cos(perpAngle),
      py - 10 * Math.sin(perpAngle),
      px + 10 * Math.cos(perpAngle),
      py + 10 * Math.sin(perpAngle),
      Colors.DESIRED_LIGHT_YELLOW,
      2,
    );

    // Black circle at thruster position
    r.drawCircle(px, py, 3, Colors.BLACK);

    // Small indicator dot offset from thruster
    r.drawCircle(
      px - 7 * Math.sin(perpAngle),
      py + 7 * Math.cos(perpAngle),
      1,
      Colors.BLACK,
    );
  }
}

// ---------------------------------------------------------------------------
// Curved arrow (matches Python _draw_curved_arrow)
// ---------------------------------------------------------------------------

function drawCurvedArrow(
  r: Renderer,
  angle: number,
  centerX: number,
  centerY: number,
  color: string,
  radius = 10,
  lineWidthVal = 2,
  arrowheadLength = 10,
  arrowheadWidthVal = 10,
  numPoints = 30,
): void {
  if (angle === 0) return;

  const sign = angle > 0 ? 1 : -1;
  const totalArcLength = radius * Math.abs(angle);

  let lineArcLength: number;
  let currentArrowheadLength: number;
  let curArrowheadWidth = arrowheadWidthVal;

  if (totalArcLength > arrowheadLength) {
    lineArcLength = totalArcLength - arrowheadLength;
    currentArrowheadLength = arrowheadLength;
  } else {
    lineArcLength = 0;
    currentArrowheadLength = totalArcLength;
    curArrowheadWidth *= totalArcLength / arrowheadLength;
  }

  const lineArcAngle = lineArcLength / radius;
  const arrowheadArcAngle = currentArrowheadLength / radius;

  const startAngle = (3 * Math.PI) / 2;
  const tipAngle = startAngle + sign * (lineArcAngle + arrowheadArcAngle);

  // Draw arc line
  if (lineArcLength > 0) {
    const arcPoints: [number, number][] = [];
    for (let i = 0; i < numPoints; i++) {
      const t = i / (numPoints - 1);
      const theta = startAngle + sign * lineArcAngle * t;
      arcPoints.push([
        centerX + radius * Math.cos(theta),
        centerY + radius * Math.sin(theta),
      ]);
    }
    r.drawLines(arcPoints, color, false, lineWidthVal);
  }

  // Arrowhead triangle
  const rotX = Math.cos(tipAngle);
  const rotY = Math.sin(tipAngle);

  const tipX = centerX + radius * rotX;
  const tipY = centerY + radius * rotY;

  const baseCenterX = tipX - currentArrowheadLength * sign * (-rotY);
  const baseCenterY = tipY - currentArrowheadLength * sign * rotX;

  const leftX = baseCenterX + (curArrowheadWidth / 2) * sign * rotX;
  const leftY = baseCenterY + (curArrowheadWidth / 2) * sign * rotY;

  const rightX = baseCenterX - (curArrowheadWidth / 2) * sign * rotX;
  const rightY = baseCenterY - (curArrowheadWidth / 2) * sign * rotY;

  r.drawPolygon(
    [
      [tipX, tipY],
      [leftX, leftY],
      [rightX, rightY],
    ],
    color,
  );
}

// ---------------------------------------------------------------------------
// Force/moment arrows
// ---------------------------------------------------------------------------

function drawForceAndMoment(
  r: Renderer,
  ox: number,
  oy: number,
  scale: number,
  cx: number,
  cy: number,
  totThrust: number,
  totAngle: number,
  totAngularThrust: number,
): void {
  const [cPx, cPy] = worldToPixel(0, 0, scale, cx, cy);
  const centerX = ox + cPx;
  const centerY = oy + cPy;

  // Total thrust arrow
  r.drawArrow(
    centerX, centerY,
    degreesToCanvas(totAngle),
    metersToPixels(scale, totThrust),
    Colors.DESIRED_YELLOW,
  );

  // Curved arrow for moment
  const radiusPx = metersToPixels(scale, VESSEL_MOMENT_MARKER);
  const angularThrustScaled = totAngularThrust * scale / radiusPx;
  drawCurvedArrow(
    r, angularThrustScaled,
    centerX, centerY,
    Colors.DESIRED_YELLOW, radiusPx,
  );
}

// ---------------------------------------------------------------------------
// Velocity arrows
// ---------------------------------------------------------------------------

function drawVelocities(
  r: Renderer,
  ox: number,
  oy: number,
  scale: number,
  cx: number,
  cy: number,
  uHat: number,
  vHat: number,
  rHat: number,
): void {
  const [cPx, cPy] = worldToPixel(0, 0, scale, cx, cy);
  const centerX = ox + cPx;
  const centerY = oy + cPy;

  // Surge velocity (forward, 0 deg)
  r.drawArrow(
    centerX, centerY,
    degreesToCanvas(0),
    metersToPixels(scale, uHat),
    Colors.VELOCITY_GREEN,
  );

  // Sway velocity (right, 90 deg)
  r.drawArrow(
    centerX, centerY,
    degreesToCanvas(90),
    metersToPixels(scale, vHat),
    Colors.VELOCITY_GREEN,
  );

  // Yaw rate (curved arrow)
  const momentRadiusPx = metersToPixels(scale, VESSEL_MOMENT_MARKER);
  drawCurvedArrow(
    r, (rHat * Math.PI) / 180,
    centerX, centerY,
    Colors.VELOCITY_GREEN, momentRadiusPx,
  );
}

// ---------------------------------------------------------------------------
// Compass (drawn procedurally, rotated by heading)
// ---------------------------------------------------------------------------

/**
 * Draw a compass rose at top-left of the sub-window.
 * Python pre-builds a surface and rotates it. We draw procedurally,
 * rotating all elements around the compass center.
 *
 * Python: heading_deg = psi_tilde - target_pose[2]
 * Python: pygame.transform.rotate(surface, -heading_deg) rotates CCW by heading.
 * In pygame, positive rotate() is CCW. So rotate(-heading_deg) means CW by heading.
 */
function drawCompass(
  r: Renderer,
  ox: number,
  oy: number,
  headingDeg: number,
): void {
  const radius = COMPASS_RADIUS;
  const letterOffset = COMPASS_LETTER_OFFSET;
  const half = radius + letterOffset + 10;
  // Screen center of compass
  const ccx = ox + 10 + half;
  const ccy = oy + 10 + half;
  const col = Colors.OCEAN_GRID;

  // Python: pygame.transform.rotate(surf, -heading_deg)
  // pygame rotate() with positive angle = CCW on screen, so -heading = CW on screen.
  // Canvas rotation matrix [cos,-sin;sin,cos] with positive angle = CW on screen (y-down).
  // So we use positive heading_deg (no negation).
  const rotRad = (headingDeg * Math.PI) / 180;
  const cosR = Math.cos(rotRad);
  const sinR = Math.sin(rotRad);

  function rotPt(lx: number, ly: number): [number, number] {
    // Rotate (lx, ly) around (0,0), then offset to compass center
    return [ccx + lx * cosR - ly * sinR, ccy + lx * sinR + ly * cosR];
  }

  // Circle outline
  r.drawCircle(ccx, ccy, radius, col, false, 2);

  const triH = COMPASS_TRI_H;
  const triW = COMPASS_TRI_W;

  // N triangle (pointing up = negative y in local coords)
  r.drawPolygon(
    [
      rotPt(0, -radius),
      rotPt(triW, -radius + triH),
      rotPt(-triW, -radius + triH),
    ],
    col,
  );

  // E triangle (pointing right = positive x in local coords)
  r.drawPolygon(
    [
      rotPt(radius, 0),
      rotPt(radius - triH, triW),
      rotPt(radius - triH, -triW),
    ],
    col,
  );

  // S triangle (pointing down = positive y in local coords)
  r.drawPolygon(
    [
      rotPt(0, radius),
      rotPt(triW, radius - triH),
      rotPt(-triW, radius - triH),
    ],
    col,
  );

  // W triangle (pointing left = negative x in local coords)
  r.drawPolygon(
    [
      rotPt(-radius, 0),
      rotPt(-radius + triH, triW),
      rotPt(-radius + triH, -triW),
    ],
    col,
  );

  // N/E/S/W letters
  // Python places letters at angle = deg-90 from center, distance = radius+letterOffset
  // Then rotates the whole surface. We rotate the placement angle.
  const letters: [string, number][] = [
    ["N", 0],
    ["E", 90],
    ["S", 180],
    ["W", 270],
  ];

  for (const [letter, deg] of letters) {
    // Original placement angle (before rotation): deg - 90 in radians
    const placementRad = ((deg - 90) * Math.PI) / 180;
    // Apply compass rotation
    const totalRad = placementRad + rotRad;
    const lx = Math.cos(totalRad) * (radius + letterOffset);
    const ly = Math.sin(totalRad) * (radius + letterOffset);
    r.drawText(letter, ccx + lx, ccy + ly, col, 16, "center", "middle");
  }
}

// ---------------------------------------------------------------------------
// Legend
// ---------------------------------------------------------------------------

const LEGEND_ITEMS: [string, string][] = [
  [Colors.DESIRED_YELLOW, "Desired {'total thrust force', 'total trust moment', 'pose'}"],
  [Colors.DESIRED_LIGHT_YELLOW, "Desired thrust force"],
  [Colors.VELOCITY_GREEN, "{'surge', 'sway', 'angular'} velocity"],
];

function drawLegend(
  r: Renderer,
  ox: number,
  oy: number,
  w: number,
  h: number,
): void {
  const markerSize = 12;
  const spacing = 5;
  const padding = 5;
  const margin = 5;
  const fontSize = 12;

  const lineHeight = 16;
  const maxTextWidth = Math.max(
    ...LEGEND_ITEMS.map(([, label]) => r.measureText(label, fontSize)),
  );

  const totalHeight =
    LEGEND_ITEMS.length * lineHeight + (LEGEND_ITEMS.length - 1) * spacing;
  const boxWidth = padding * 2 + markerSize + spacing + maxTextWidth;
  const boxHeight = padding * 2 + totalHeight;

  // Position: bottom-right with margin, no padding_bottom (0)
  const boxX = ox + w - margin - boxWidth;
  const boxY = oy + h - margin - boxHeight;

  // Background box
  r.drawRect(boxX, boxY, boxWidth, boxHeight, Colors.LEGEND_BOX);

  // Items
  let yOffset = padding;
  for (const [markerColor, label] of LEGEND_ITEMS) {
    const markerY =
      boxY + yOffset + Math.floor((lineHeight - markerSize) / 2);
    r.drawRect(boxX + padding, markerY, markerSize, markerSize, markerColor);
    r.drawText(
      label,
      boxX + padding + markerSize + spacing,
      boxY + yOffset,
      Colors.BLACK,
      fontSize,
    );
    yOffset += lineHeight + spacing;
  }
}

// ---------------------------------------------------------------------------
// Main render function
// ---------------------------------------------------------------------------

export function renderBody(
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

  const { x_tilde, y_tilde, psi_tilde, u_hat, v_hat, r_hat, target_pose } =
    frame.vessel;
  const { actuator_ref, tot_thrust, tot_angle, tot_angular_thrust } =
    frame.actuators;

  // Background
  r.drawRect(ox, oy, w, h, Colors.OCEAN_BLUE);

  // Body grid (rotated by heading)
  drawBodyGrid(
    r, ox, oy, w, h, scale, cx, cy,
    x_tilde, y_tilde, psi_tilde, target_pose,
    1.0, Colors.OCEAN_GRID, 1,
  );

  // Target: dashed hull at (x_tilde, y_tilde, psi_tilde) offset
  drawTarget(
    r, ox, oy, scale, cx, cy,
    x_tilde, y_tilde, psi_tilde,
    Colors.DESIRED_YELLOW,
  );

  // Vessel: solid hull at center (0,0)
  drawVessel(r, ox, oy, scale, cx, cy);

  // Actuator arrows
  drawActuatorRef(r, ox, oy, scale, cx, cy, actuator_ref);

  // Force and moment
  drawForceAndMoment(
    r, ox, oy, scale, cx, cy,
    tot_thrust, tot_angle, tot_angular_thrust,
  );

  // Velocities
  drawVelocities(r, ox, oy, scale, cx, cy, u_hat, v_hat, r_hat);

  // Compass (heading = psi_tilde - target heading)
  drawCompass(r, ox, oy, psi_tilde - target_pose[2]);

  // Legend at bottom-right
  drawLegend(r, ox, oy, w, h);

  // Title top-right
  r.drawText(
    "State and desired actuator view in BODY-frame",
    ox + w - 5, oy + 5,
    Colors.GRAY, 11, "right",
  );
}
