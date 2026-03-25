/**
 * SHAP force/moment explanation visualization.
 * Faithful port of milliampere_xai/rendering/_shap_explain.py (ShapExplainRender).
 *
 * Shows SHAP contributions as force/moment arrows on a vessel body diagram.
 * The most influential feature is identified, then a light-yellow force arrow
 * and curved moment arrow show the estimated thrust from that single feature.
 * A red annotation visualises the feature itself (distance, angle, or velocity).
 * Accuracy metrics compare SHAP-estimated vs actual force/moment.
 *
 * Coordinate system (VesselRender body-frame):
 *   SCALE = 50,  center = (w/2, h/2)
 *   _world_2_pixels: px = wy * scale + cx,  py = wx * scale + cy
 *   _degrees2pygame(angle_deg) = pi * angle_deg / 180 - pi / 2
 *   _scalar2pygame(scalar) = scale * scalar
 */

import { Colors } from "../colors.ts";
import {
  THRUSTER_ARM_X,
  THRUSTER_ARM_Y,
  VESSEL_BEAM,
  VESSEL_LENGTH,
  vesselHullPolygon,
  vesselMomentMarkerLine,
  vesselTrianglePolygon,
} from "../geometry.ts";
import type { Renderer } from "../renderer.ts";
import type { RenderFrame } from "../types.ts";

// ── Constants matching Python ShapExplainRender ────────────────────────

const SCALE = 50; // pixels per meter
const EXPLAIN_OFFSET = 20; // px offset for feature annotations

const ACTUATOR_POS: [number, number][] = [
  [THRUSTER_ARM_X, -THRUSTER_ARM_Y],
  [THRUSTER_ARM_X, THRUSTER_ARM_Y],
  [-THRUSTER_ARM_X, THRUSTER_ARM_Y],
  [-THRUSTER_ARM_X, -THRUSTER_ARM_Y],
];

const NUM_FEATURES = 14;

// ── Persistent state (survives across frames, like Python instance vars) ───

let prevAngle1 = [135, -135, -45, 45];
let prevAngle2 = [135, -135, -45, 45];

// ── Viewport helpers (Python VesselRender body-frame convention) ────────

function worldToPixels(
  wx: number,
  wy: number,
  scale: number,
  cx: number,
  cy: number,
): [number, number] {
  // Python: R = [[0, scale], [-scale, 0]], T = [[cy], [cx]]
  // pixel = R @ [wx, wy]^T + T  =>  px = scale*wy + cx,  py = scale*wx + cy
  // But wait — Python R is [[0,s],[-s,0]], so:
  //   px = 0*wx + s*wy + cy  = s*wy + cy
  //   py = -s*wx + 0*wy + cx = -s*wx + cx
  // Hmm, let me re-read: T = [[center[1]], [center[0]]] = [[cy],[cx]]
  // Result row0 = 0*wx + s*wy + cy, row1 = -s*wx + 0*wy + cx
  // But world_to_pixels returns (R@coords.T + T).T whose columns are [row0, row1]
  // So pixel_x = s*wy + cy, pixel_y = -s*wx + cx
  // Actually this is the NED viewport convention.  In the NED views (ned.ts)
  // we use:  px = s*wy + cx,  py = s*wx + cy   (from TypeScript Viewport)
  // The Python VesselRender uses the same Viewport.from_window which gives
  // the R=[[0,s],[-s,0]] T=[[cy],[cx]] convention.
  //
  // For body-frame views (centered, no NED offset), the Python code at psi=0 gives:
  //   pixel_x = 0*wx + scale*wy + cy  = scale*wy + cy
  //   pixel_y = -scale*wx + 0*wy + cx = -scale*wx + cx
  //
  // Note Python T = [[center_y], [center_x]] — an intentional y,x swap.
  // So: result[0] = scale*wy + center_y,  result[1] = -scale*wx + center_x
  return [scale * wy + cy, -scale * wx + cx];
}

/** degrees_to_pygame: pi*angle/180 - pi/2 */
function degreesToPygame(angleDeg: number): number {
  return (Math.PI * angleDeg) / 180 - Math.PI / 2;
}

/** meters_to_pixels: scale * value */
function scalarToPygame(scalar: number): number {
  return SCALE * scalar;
}

// ── Clamp / SSA ────────────────────────────────────────────────────────

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Smallest signed angle (degrees). Maps to (-180, 180].
 *  JS % returns negative remainders unlike Python, so we correct. */
function ssa(angleDeg: number): number {
  let a = (angleDeg + 180) % 360;
  if (a < 0) a += 360;
  return a - 180;
}

// ── Shape transform (body-frame, matches _transform_vessel at psi in degrees) ──

function transformShape(
  x: number,
  y: number,
  psiDeg: number,
  shape: [number, number][],
): [number, number][] {
  const psiRad = (psiDeg * Math.PI) / 180;
  const cos = Math.cos(psiRad);
  const sin = Math.sin(psiRad);
  // Python: (R2(psi).T @ shape.T + T2(x,y)).T
  // R2(psi) = [[cos,sin],[-sin,cos]]  => R2.T = [[cos,-sin],[sin,cos]]
  return shape.map(([sx, sy]) => [
    cos * sx - sin * sy + x,
    sin * sx + cos * sy + y,
  ]);
}

// ── combine_actuator_ref (port of model_wrappers.py) ───────────────────

function combineActuatorRef(
  actuatorRef: [number, number][],
  actuatorPos: [number, number][],
): [number, number, number] {
  let thrustX = 0;
  let thrustY = 0;
  let totAngularThrust = 0;

  for (let i = 0; i < actuatorRef.length; i++) {
    const [thrust, angleDeg] = actuatorRef[i];
    const [x, y] = actuatorPos[i];
    const rad = ((90 * (i + 1) - angleDeg) * Math.PI) / 180;
    const angleRad = (angleDeg * Math.PI) / 180;

    thrustX += thrust * Math.cos(angleRad);
    thrustY += thrust * Math.sin(angleRad);

    let angThrust: number;
    if (x * y < 0) {
      angThrust =
        Math.abs(x) * thrust * Math.cos(rad) +
        Math.abs(y) * thrust * Math.sin(rad);
    } else {
      angThrust =
        Math.abs(x) * thrust * Math.sin(rad) +
        Math.abs(y) * thrust * Math.cos(rad);
    }
    totAngularThrust += angThrust;
  }

  const totThrust = Math.hypot(thrustX, thrustY);
  const totAngle = (Math.atan2(thrustY, thrustX) * 180) / Math.PI;

  return [totThrust, totAngle, totAngularThrust];
}

// ── SHAP computation (mirrors _find_explanation2) ──────────────────────

interface ShapResult {
  /** (4, 14) clipped distances per thruster pair per feature */
  distsClipped: number[][];
  /** (4, 2, 14) scaled SHAP values [pair][xy][feature] */
  scaled: number[][][];
  /** (4, 2, 15) scaled + base appended [pair][xy][feature+1] */
  scaledWithBase: number[][][];
  /** Index of most influential feature */
  maxIdx: number;
  /** Normalisation factor c per action output (8,) */
  c: number[];
}

function findExplanation(
  svAction: number[][],
  baseValue: number[],
  actionLow: number[],
  actionHigh: number[],
): ShapResult {
  const numActions = svAction.length; // 8
  const numFeats = svAction[0].length; // 14

  // sv_all_actions
  const svAllActions: number[] = new Array(numActions);
  for (let a = 0; a < numActions; a++) {
    let rowSum = 0;
    for (let f = 0; f < numFeats; f++) rowSum += svAction[a][f];
    svAllActions[a] = clamp(rowSum + baseValue[a], actionLow[a], actionHigh[a]);
  }

  // pos/neg totals
  const posTotal: number[] = new Array(numActions);
  const negTotal: number[] = new Array(numActions);
  for (let a = 0; a < numActions; a++) {
    let pos = 0;
    let neg = 0;
    for (let f = 0; f < numFeats; f++) {
      const v = svAction[a][f];
      if (v > 0) pos += v;
      else neg += v;
    }
    pos += baseValue[a] > 0 ? baseValue[a] : 0;
    neg += baseValue[a] < 0 ? baseValue[a] : 0;
    posTotal[a] = pos;
    negTotal[a] = neg;
  }

  // c
  const c: number[] = new Array(numActions);
  for (let a = 0; a < numActions; a++) {
    const denom = actionHigh[a] === 1 ? posTotal[a] : negTotal[a];
    c[a] = denom !== 0 ? svAllActions[a] / denom : 0;
  }

  // Reshape (8,14) -> (4,2,14), scale by c
  const numPairs = numActions / 2;
  const scaled: number[][][] = [];
  for (let p = 0; p < numPairs; p++) {
    const xRow: number[] = new Array(numFeats);
    const yRow: number[] = new Array(numFeats);
    const xIdx = p * 2;
    const yIdx = p * 2 + 1;
    for (let f = 0; f < numFeats; f++) {
      xRow[f] = svAction[xIdx][f] * c[xIdx];
      yRow[f] = svAction[yIdx][f] * c[yIdx];
    }
    scaled.push([xRow, yRow]);
  }

  // scaled_with_base: append base_scaled column
  // base_scaled[p][xy] = sum_f(sv_base * c_pairs[p][xy]) for each xy
  const scaledWithBase: number[][][] = [];
  for (let p = 0; p < numPairs; p++) {
    const xIdx = p * 2;
    const yIdx = p * 2 + 1;
    // base_scaled for x-row
    let baseX = 0;
    let baseY = 0;
    for (let f = 0; f < numFeats; f++) {
      // Python: base_scaled = (sv_base[None,None,:] * c_pairs[:,:,None]).sum(axis=2)
      // sv_base has shape (8,) — but it's broadcast as if it were (14,)
      // Wait, looking more carefully: sv_base shape is (8,) in Python.
      // The code does: sv_base[None, None, :] * c_pairs[:, :, None]
      // sv_base shape (8,) -> (1,1,8),  c_pairs shape (4,2) -> (4,2,1)
      // multiply -> (4,2,8), .sum(axis=2, keepdims=True) -> (4,2,1)
      // This doesn't match features... actually base_value is length 8, not 14.
      // So this sums over the 8 base values scaled by c.
      // But wait, the reshape means c_pairs is (4,2):
      //   c_pairs[p,0] = c[2p], c_pairs[p,1] = c[2p+1]
      // And sv_base is (8,), so sv_base[None,None,:] is (1,1,8)
      // c_pairs[:,:,None] is (4,2,1)
      // product is (4,2,8), sum over axis=2 -> (4,2)
      // So base_scaled[p,0] = sum(sv_base * c[2p]),  base_scaled[p,1] = sum(sv_base * c[2p+1])
      // This is just c[2p] * sum(sv_base) and c[2p+1] * sum(sv_base)
      // Actually no: sv_base[None,None,:] * c_pairs[:,:,None]
      // at [p, 0, k] = sv_base[k] * c_pairs[p,0]
      // sum over k: c_pairs[p,0] * sum_k(sv_base[k])
      baseX = 0; // will compute outside loop
      baseY = 0;
    }
    let svBaseSum = 0;
    for (let k = 0; k < baseValue.length; k++) svBaseSum += baseValue[k];
    baseX = c[xIdx] * svBaseSum;
    baseY = c[yIdx] * svBaseSum;

    const xRowExt = [...scaled[p][0], baseX];
    const yRowExt = [...scaled[p][1], baseY];
    scaledWithBase.push([xRowExt, yRowExt]);
  }

  // Euclidean distances, clipped to [0,1]
  const distsClipped: number[][] = [];
  for (let p = 0; p < numPairs; p++) {
    const row: number[] = new Array(numFeats);
    for (let f = 0; f < numFeats; f++) {
      row[f] = clamp(Math.hypot(scaled[p][0][f], scaled[p][1][f]), 0, 1);
    }
    distsClipped.push(row);
  }

  // Sum across pairs, find max feature
  const sumed: number[] = new Array(numFeats).fill(0);
  for (let f = 0; f < numFeats; f++) {
    for (let p = 0; p < numPairs; p++) sumed[f] += distsClipped[p][f];
  }
  let maxIdx = 0;
  for (let f = 1; f < numFeats; f++) {
    if (sumed[f] > sumed[maxIdx]) maxIdx = f;
  }

  return { distsClipped, scaled, scaledWithBase, maxIdx, c };
}

// ── Vessel drawing helpers ─────────────────────────────────────────────

function drawVesselBody(
  r: Renderer,
  ox: number,
  oy: number,
  scale: number,
  cx: number,
  cy: number,
): void {
  const hull = vesselHullPolygon();
  const tri = vesselTrianglePolygon();
  const moment = vesselMomentMarkerLine();

  const toScreen = (pts: [number, number][]): [number, number][] =>
    pts.map(([wx, wy]) => {
      const [px, py] = worldToPixels(wx, wy, scale, cx, cy);
      return [ox + px, oy + py];
    });

  r.drawPolygon(toScreen(hull), Colors.AGENT_BLUE);
  r.drawPolygon(toScreen(tri), Colors.BLACK);

  const [cPx, cPy] = worldToPixels(0, 0, scale, cx, cy);
  r.drawCircle(ox + cPx, oy + cPy, 3, Colors.BLACK);

  const momentPx = toScreen(moment);
  r.drawLine(
    momentPx[0][0], momentPx[0][1],
    momentPx[1][0], momentPx[1][1],
    Colors.BLACK, 1,
  );
}

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

  // Body frame (ned=false): rotate+translate shape by error
  const transformedHull = transformShape(xErr, yErr, psiErr, hull);
  const transformedTri = transformShape(xErr, yErr, psiErr, tri);

  const toScreen = (pts: [number, number][]): [number, number][] =>
    pts.map(([wx, wy]) => {
      const [px, py] = worldToPixels(wx, wy, scale, cx, cy);
      return [ox + px, oy + py];
    });

  const hullPx = toScreen(transformedHull);
  const triPx = toScreen(transformedTri);

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

  // Circle at transformed CoG
  const circleWorld = transformShape(xErr, yErr, psiErr, [[0, 0]]);
  const [circlePx, circlePy] = worldToPixels(
    circleWorld[0][0], circleWorld[0][1], scale, cx, cy,
  );
  r.drawCircle(ox + circlePx, oy + circlePy, 3, color);
}

// ── Drawing primitives (ported from _primitives.py) ────────────────────

function drawArrowPrimitive(
  r: Renderer,
  ox: number,
  oy: number,
  angle: number,
  arrowLength: number,
  color: string,
  startPx: number,
  startPy: number,
  headLen = 10,
  headWid = 10,
  lineWid = 2,
): void {
  // The Renderer.drawArrow already handles negative length, head sizing, etc.
  r.drawArrow(
    ox + startPx,
    oy + startPy,
    angle,
    arrowLength,
    color,
    headLen,
    headWid,
    lineWid,
  );
}

/**
 * Draw a curved arrow (arc with arrowhead) matching Python _draw_curved_arrow.
 */
function drawCurvedArrow(
  r: Renderer,
  ox: number,
  oy: number,
  angle: number,
  centerPx: number,
  centerPy: number,
  color: string,
  radius: number,
  lineWidth = 2,
  headLen = 10,
  headWid = 10,
  numPoints = 30,
): void {
  if (angle === 0) return;

  const sign = angle > 0 ? 1 : -1;
  const totalArcLen = radius * Math.abs(angle);

  let lineArcLen: number;
  let curHeadLen: number;
  let curHeadWid = headWid;

  if (totalArcLen > headLen) {
    lineArcLen = totalArcLen - headLen;
    curHeadLen = headLen;
  } else {
    lineArcLen = 0;
    curHeadLen = totalArcLen;
    curHeadWid *= totalArcLen / headLen;
  }

  const lineArcAngle = lineArcLen / radius;
  const headArcAngle = curHeadLen / radius;
  const startAngle = (3 * Math.PI) / 2;
  const tipAngle = startAngle + sign * (lineArcAngle + headArcAngle);

  const absCx = ox + centerPx;
  const absCy = oy + centerPy;

  // Draw the arc line
  if (lineArcLen > 0) {
    const pts: [number, number][] = [];
    for (let i = 0; i < numPoints; i++) {
      const t = i / (numPoints - 1);
      const theta = startAngle + sign * (lineArcAngle * t);
      pts.push([
        absCx + radius * Math.cos(theta),
        absCy + radius * Math.sin(theta),
      ]);
    }
    r.drawLines(pts, color, false, lineWidth);
  }

  // Arrowhead at tip
  const cosT = Math.cos(tipAngle);
  const sinT = Math.sin(tipAngle);
  const tipX = absCx + radius * cosT;
  const tipY = absCy + radius * sinT;

  const baseCenterX = tipX - curHeadLen * sign * (-sinT);
  const baseCenterY = tipY - curHeadLen * sign * cosT;

  const leftX = baseCenterX + (curHeadWid / 2) * sign * cosT;
  const leftY = baseCenterY + (curHeadWid / 2) * sign * sinT;
  const rightX = baseCenterX - (curHeadWid / 2) * sign * cosT;
  const rightY = baseCenterY - (curHeadWid / 2) * sign * sinT;

  r.drawPolygon(
    [
      [tipX, tipY],
      [leftX, leftY],
      [rightX, rightY],
    ],
    color,
  );
}

/**
 * Draw a distance measurement line with perpendicular end-markers.
 * Port of Utilities._draw_distance.
 */
function drawDistance(
  r: Renderer,
  ox: number,
  oy: number,
  startPx: number,
  startPy: number,
  length: number,
  angleRad: number,
  color: string,
  label: string,
  perpLen = 10,
  lineWidth = 2,
): void {
  const sx = ox + startPx;
  const sy = oy + startPy;
  const endX = sx + length * Math.cos(angleRad);
  const endY = sy + length * Math.sin(angleRad);

  // Main line
  r.drawLine(sx, sy, endX, endY, color, lineWidth);

  // Perpendicular markers
  const perpAngle = angleRad + Math.PI / 2;
  const cosP = Math.cos(perpAngle);
  const sinP = Math.sin(perpAngle);

  r.drawLine(
    sx + (perpLen / 2) * cosP, sy + (perpLen / 2) * sinP,
    sx - (perpLen / 2) * cosP, sy - (perpLen / 2) * sinP,
    color, lineWidth,
  );
  r.drawLine(
    endX + (perpLen / 2) * cosP, endY + (perpLen / 2) * sinP,
    endX - (perpLen / 2) * cosP, endY - (perpLen / 2) * sinP,
    color, lineWidth,
  );

  // Label text above the midpoint
  const midX = (sx + endX) / 2;
  const midY = (sy + endY) / 2;

  // Adjust position: if line is near-vertical, offset to the right
  const isVertical =
    Math.PI / 2 - Math.PI / 16 < Math.abs(angleRad) &&
    Math.abs(angleRad) < Math.PI / 2 + Math.PI / 16;

  if (isVertical) {
    r.drawText(label, midX + 10, midY, color, 18, "left", "middle");
  } else {
    r.drawText(label, midX, midY + 5, color, 18, "center", "top");
  }
}

/**
 * Draw an angle measurement arc with perpendicular end-markers.
 * Port of Utilities._draw_angle.
 */
function drawAngle(
  r: Renderer,
  ox: number,
  oy: number,
  centerPx: number,
  centerPy: number,
  angleRad: number,
  color: string,
  label: string,
  radius: number,
  lineWidth = 2,
  numPoints = 30,
): void {
  if (angleRad === 0) return;

  const sign = angleRad > 0 ? 1 : -1;
  const arcAngle = Math.abs(angleRad);
  const startAng = (3 * Math.PI) / 2;
  const endAng = startAng + sign * arcAngle;

  const absCx = ox + centerPx;
  const absCy = oy + centerPy;

  // Arc polyline
  const pts: [number, number][] = [];
  for (let i = 0; i < numPoints; i++) {
    const t = i / (numPoints - 1);
    const theta = startAng + sign * (arcAngle * t);
    pts.push([
      absCx + radius * Math.cos(theta),
      absCy + radius * Math.sin(theta),
    ]);
  }
  r.drawLines(pts, color, false, lineWidth);

  // Perpendicular markers at start and end (radial direction)
  const perpLen = 10;

  const startVx = Math.cos(startAng);
  const startVy = Math.sin(startAng);
  const sPx = absCx + radius * startVx;
  const sPy = absCy + radius * startVy;
  r.drawLine(
    sPx - (perpLen / 2) * startVx, sPy - (perpLen / 2) * startVy,
    sPx + (perpLen / 2) * startVx, sPy + (perpLen / 2) * startVy,
    color, lineWidth,
  );

  const endVx = Math.cos(endAng);
  const endVy = Math.sin(endAng);
  const ePx = absCx + radius * endVx;
  const ePy = absCy + radius * endVy;
  r.drawLine(
    ePx - (perpLen / 2) * endVx, ePy - (perpLen / 2) * endVy,
    ePx + (perpLen / 2) * endVx, ePy + (perpLen / 2) * endVy,
    color, lineWidth,
  );

  // Label at midpoint of arc
  const midAng = startAng + sign * (arcAngle / 2);
  const textX = absCx + radius * 1.2 * Math.cos(midAng);
  const textY = absCy + radius * 1.2 * Math.sin(midAng);
  r.drawText(label, textX, textY, color, 18, "center", "middle");
}

// ── Legend ──────────────────────────────────────────────────────────────

function drawLegend(
  r: Renderer,
  ox: number,
  oy: number,
  w: number,
  h: number,
): void {
  const items: [string, string][] = [
    [Colors.RED, "Main feature causing the desired total trust force/moment"],
    [
      Colors.DESIRED_LIGHT_YELLOW,
      "Estimated desired total trust force/moment based on main feature",
    ],
  ];

  const markerSize = 12;
  const spacing = 5;
  const padding = 5;
  const margin = 5;
  const fontSize = 12; // matches body + shap_bars legends (Python label_font size 12)
  const lineHeight = 16;

  // Measure actual text width
  const maxTextWidth = Math.max(...items.map(([, label]) => r.measureText(label, fontSize)));
  const boxW = padding * 2 + markerSize + spacing + maxTextWidth;
  const totalH = items.length * lineHeight + (items.length - 1) * spacing;
  const boxH = padding * 2 + totalH;

  const boxX = ox + w - margin - boxW;
  const boxY = oy + h - margin - boxH;

  r.drawRect(boxX, boxY, boxW, boxH, Colors.LEGEND_BOX);

  let yOff = boxY + padding;
  for (const [color, label] of items) {
    r.drawRect(
      boxX + padding,
      yOff + (lineHeight - markerSize) / 2,
      markerSize,
      markerSize,
      color,
    );
    r.drawText(
      label,
      boxX + padding + markerSize + spacing,
      yOff,
      Colors.BLACK,
      fontSize,
    );
    yOff += lineHeight + spacing;
  }
}

// ── Accuracy color helper ──────────────────────────────────────────────

function accColor(acc: number): string {
  if (acc > 0.75) return Colors.GREEN;
  if (acc > 0.5) return Colors.ORANGE;
  return Colors.RED;
}

// ── Main render function ───────────────────────────────────────────────

export function renderShapExplain(
  r: Renderer,
  frame: RenderFrame,
  ox: number,
  oy: number,
  w: number,
  h: number,
): void {
  // White background
  r.drawRect(ox, oy, w, h, Colors.WHITE);

  const scale = SCALE;
  const cx = w / 2;
  const cy = h / 2;

  const { x_tilde, y_tilde, psi_tilde, u_hat, v_hat, r_hat } = frame.vessel;
  const { tot_thrust, tot_angle, tot_angular_thrust } = frame.actuators;
  const shap = frame.shap;

  // ── Compute SHAP explanation ──────────────────────────────────────

  const result = findExplanation(
    shap.shap_values_action,
    shap.base_vectors,
    shap.action_low,
    shap.action_high,
  );

  const { distsClipped, scaled, scaledWithBase, maxIdx } = result;
  const idx = maxIdx;

  // ── Draw target (dashed outline at error position, gray) ──────────

  drawTarget(r, ox, oy, scale, cx, cy, x_tilde, y_tilde, psi_tilde, Colors.GRAY);

  // ── Draw vessel (solid hull at center) ────────────────────────────

  drawVesselBody(r, ox, oy, scale, cx, cy);

  // ── _draw_explanation4: force/moment arrows ───────────────────────

  // Per-thruster SHAP distances for the max feature
  const thrustersShap: number[] = [];
  for (let p = 0; p < 4; p++) thrustersShap.push(distsClipped[p][idx]);

  // Angles from arctan2 of scaled x,y components
  const anglesShap: number[] = [];
  for (let p = 0; p < 4; p++) {
    if (thrustersShap[p] === 0) {
      anglesShap.push(prevAngle1[p]);
    } else {
      anglesShap.push(
        (Math.atan2(scaled[p][1][idx], scaled[p][0][idx]) * 180) / Math.PI,
      );
    }
  }
  if (anglesShap[1] === 180) anglesShap[1] = -180;
  prevAngle1 = [...anglesShap];

  const vec: [number, number][] = thrustersShap.map((t, i) => [
    t,
    anglesShap[i],
  ]);

  // Total distances per thruster pair (sum across all features)
  const thrustersShapTot: number[] = [];
  for (let p = 0; p < 4; p++) {
    let s = 0;
    for (let f = 0; f < NUM_FEATURES; f++) s += distsClipped[p][f];
    thrustersShapTot.push(s);
  }

  // Total angles from scaledWithBase summed over features+base
  const anglesShap2: number[] = [];
  for (let p = 0; p < 4; p++) {
    if (thrustersShapTot[p] === 0) {
      anglesShap2.push(prevAngle2[p]);
    } else {
      // Sum over all columns (features+base) for x and y
      let sumX = 0;
      let sumY = 0;
      for (let f = 0; f < scaledWithBase[p][0].length; f++) {
        sumX += scaledWithBase[p][0][f];
        sumY += scaledWithBase[p][1][f];
      }
      anglesShap2.push((Math.atan2(sumY, sumX) * 180) / Math.PI);
    }
  }
  if (anglesShap2[1] === 180) anglesShap2[1] = -180;
  prevAngle2 = [...anglesShap2];

  // Combine per-thruster vectors into total force using combine_actuator_ref
  const vector = combineActuatorRef(vec, ACTUATOR_POS);

  // ── Draw estimated force arrow (light yellow) ─────────────────────

  const [centerPx, centerPy] = worldToPixels(0, 0, scale, cx, cy);

  const forceAngle = degreesToPygame(vector[1]);
  const forceLen = scalarToPygame(vector[0]);

  if (Math.abs(forceLen) > 1) {
    drawArrowPrimitive(
      r, ox, oy,
      forceAngle,
      forceLen,
      Colors.DESIRED_LIGHT_YELLOW,
      centerPx, centerPy,
    );
  }

  // ── Draw estimated moment (curved arrow, light yellow) ────────────

  const momentRadius = 30;
  const rawAngle = clamp(
    vector[2] / momentRadius,
    -2 * Math.PI,
    2 * Math.PI,
  );
  const drawAngleVal = scalarToPygame(rawAngle);

  if (Math.abs(drawAngleVal) > 0.5) {
    drawCurvedArrow(
      r, ox, oy,
      drawAngleVal,
      centerPx, centerPy,
      Colors.DESIRED_LIGHT_YELLOW,
      momentRadius,
    );
  }

  // ── Draw feature-specific annotation (red) ────────────────────────

  const vesselHalfWidth = VESSEL_BEAM / 2;

  if (idx === 0) {
    // x_tilde distance annotation
    drawDistance(
      r, ox, oy,
      centerPx + scalarToPygame(vesselHalfWidth) + EXPLAIN_OFFSET,
      centerPy,
      scalarToPygame(x_tilde),
      degreesToPygame(0),
      Colors.RED,
      `${x_tilde.toFixed(2)} m`,
    );
  } else if (idx === 1) {
    // y_tilde distance annotation
    drawDistance(
      r, ox, oy,
      centerPx,
      centerPy + scalarToPygame(VESSEL_LENGTH / 2) + EXPLAIN_OFFSET,
      scalarToPygame(y_tilde),
      degreesToPygame(90),
      Colors.RED,
      `${y_tilde.toFixed(2)} m`,
    );
  } else if (idx === 2) {
    // psi_tilde angle annotation
    const arcRadius =
      scalarToPygame(VESSEL_LENGTH / 2) + EXPLAIN_OFFSET;
    drawAngle(
      r, ox, oy,
      centerPx, centerPy,
      (psi_tilde * Math.PI) / 180,
      Colors.RED,
      `${psi_tilde.toFixed(0)} °`,
      arcRadius,
    );
  } else if (idx === 3) {
    // u_hat surge velocity arrow
    const uOffset =
      u_hat > 0
        ? centerPy - scalarToPygame(VESSEL_LENGTH / 2) - EXPLAIN_OFFSET
        : centerPy + scalarToPygame(VESSEL_LENGTH / 2) + EXPLAIN_OFFSET;
    drawArrowPrimitive(
      r, ox, oy,
      degreesToPygame(0),
      scalarToPygame(u_hat),
      Colors.RED,
      centerPx, uOffset,
    );
    // Label — surge is vertical arrow (degrees_to_pygame(0) = -pi/2)
    // Python _draw_arrow: vertical → label to the RIGHT at midpoint
    const angle0 = degreesToPygame(0);
    const tipSurgeX = ox + centerPx + scalarToPygame(u_hat) * Math.cos(angle0);
    const tipSurgeY = oy + uOffset + scalarToPygame(u_hat) * Math.sin(angle0);
    const midSurgeX = (ox + centerPx + tipSurgeX) / 2;
    const midSurgeY = (oy + uOffset + tipSurgeY) / 2;
    r.drawText(
      `${(u_hat / 2).toFixed(2)} m/s`,
      midSurgeX + 10, midSurgeY,
      Colors.RED, 18, "left", "middle",
    );
  } else if (idx === 4) {
    // v_hat sway velocity arrow
    const vOffsetX =
      v_hat > 0
        ? centerPx + scalarToPygame(vesselHalfWidth) + EXPLAIN_OFFSET
        : centerPx - scalarToPygame(vesselHalfWidth) - EXPLAIN_OFFSET;
    drawArrowPrimitive(
      r, ox, oy,
      degreesToPygame(90),
      scalarToPygame(v_hat),
      Colors.RED,
      vOffsetX, centerPy,
    );
    // Label — sway is horizontal arrow (degrees_to_pygame(90) = 0)
    // Python _draw_arrow: non-vertical → label ABOVE midpoint
    const angle90 = degreesToPygame(90);
    const tipSwayX = ox + vOffsetX + scalarToPygame(v_hat) * Math.cos(angle90);
    const tipSwayY = oy + centerPy + scalarToPygame(v_hat) * Math.sin(angle90);
    const midSwayX = (ox + vOffsetX + tipSwayX) / 2;
    const midSwayY = (oy + centerPy + tipSwayY) / 2;
    r.drawText(
      `${(v_hat / 2).toFixed(2)} m/s`,
      midSwayX, midSwayY - 18 - 5,
      Colors.RED, 18, "center", "top",
    );
  } else if (idx === 5) {
    // r_hat yaw rate (curved arrow)
    const rRadius =
      scalarToPygame(VESSEL_LENGTH / 2) + EXPLAIN_OFFSET;
    drawCurvedArrow(
      r, ox, oy,
      (r_hat * Math.PI) / 180,
      centerPx, centerPy,
      Colors.RED,
      rRadius,
    );
    // Label at arc midpoint
    const midAng = (3 * Math.PI) / 2 + ((r_hat * Math.PI) / 180) / 2;
    const labelR = rRadius * 1.3;
    r.drawText(
      `${((r_hat * 112.6) / (360 * 2)).toFixed(0)} °/s`,
      ox + centerPx + labelR * Math.cos(midAng),
      oy + centerPy + labelR * Math.sin(midAng),
      Colors.RED, 18, "center", "middle",
    );
  }

  // ── Accuracy metrics ──────────────────────────────────────────────

  const error0 = tot_thrust - vector[0];
  const error1 = ssa(tot_angle - vector[1]);
  const error2 = tot_angular_thrust - vector[2];

  const accRPM = (Math.SQRT2 + 1 - Math.abs(error0)) / (Math.SQRT2 + 1);
  const accAngleVal = (180 - Math.abs(error1)) / 180;
  const accMoment = (10.4 - Math.abs(error2)) / 10.4;
  const accTotal = (accRPM + accAngleVal + accMoment) / 3;

  // Python: _title_offset = 5
  const titleOffset = 5;
  const fontSize = 11;
  const lineH = 14;

  r.drawText(
    `Explained RPM: ${accRPM.toFixed(2)}`,
    ox + titleOffset, oy + titleOffset * 10,
    accColor(accRPM), fontSize,
  );
  r.drawText(
    `Explained angle: ${accAngleVal.toFixed(2)}`,
    ox + titleOffset, oy + titleOffset * 11 + lineH,
    accColor(accAngleVal), fontSize,
  );
  r.drawText(
    `Explained moment: ${accMoment.toFixed(2)}`,
    ox + titleOffset, oy + titleOffset * 12 + lineH * 2,
    accColor(accMoment), fontSize,
  );
  r.drawText(
    `Explained TOTAL: ${accTotal.toFixed(2)}`,
    ox + titleOffset, oy + titleOffset * 14 + lineH * 3,
    accColor(accTotal), fontSize,
  );

  // ── Legend ─────────────────────────────────────────────────────────

  drawLegend(r, ox, oy, w, h);

  // ── Title ─────────────────────────────────────────────────────────

  r.drawText(
    "Desired total trust force/moment explained in BODY-frame",
    ox + w - 5, oy + 5,
    Colors.GRAY, 11, "right",
  );
}
