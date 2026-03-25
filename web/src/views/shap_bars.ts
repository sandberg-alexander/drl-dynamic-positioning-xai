/**
 * SHAP horizontal bar chart visualization.
 * Faithful port of milliampere_xai/rendering/_shap_bars.py (ShapRender).
 *
 * Shows the top-3 most influential features + aggregated "Others" as four
 * horizontal stacked bars.  Each bar has 4 coloured segments representing
 * the 4 thruster pairs.
 */

import { Colors } from "../colors.ts";
import type { Renderer } from "../renderer.ts";
import type { RenderFrame } from "../types.ts";

// ── Constants matching Python ShapRender.__init__ ──────────────────────

const MAX_THRUSTER_RPM = 900;

const FEATURE_NAMES = [
  "x̃ᵇₜ [m]",
  "ỹᵇₜ [m]",
  "ψ̃ₜ[°]",
  "ûₜ [m/s]",
  "v̂ₜ [m/s]",
  "r̂ₜ [°/s]",
  "n_{ₓ₁,d,ₜ₋₁} [RPM]",
  "n_{ᵧ₁,d,ₜ₋₁} [RPM]",
  "n_{ₓ₂,d,ₜ₋₁} [RPM]",
  "n_{ᵧ₂,d,ₜ₋₁} [RPM]",
  "n_{ₓ₃,d,ₜ₋₁} [RPM]",
  "n_{ᵧ₃,d,ₜ₋₁} [RPM]",
  "n_{ₓ₄,d,ₜ₋₁} [RPM]",
  "n_{ᵧ₄,d,ₜ₋₁} [RPM]",
];

const BAR_COLORS = [
  Colors.TABLEAU_BLUE,
  Colors.TABLEAU_ORANGE,
  Colors.TABLEAU_GREEN,
  Colors.TABLEAU_RED,
];

const LEGEND_ITEMS_THRUST = [
  "n_{₁,d,ₜ} [RPM]",
  "n_{₂,d,ₜ} [RPM]",
  "n_{₃,d,ₜ} [RPM]",
  "n_{₄,d,ₜ} [RPM]",
];

const LEGEND_ITEMS_ANGLE = [
  "α_{d₁,ₜ} [°]",
  "α_{d₂,ₜ} [°]",
  "α_{d₃,ₜ} [°]",
  "α_{d₄,ₜ} [°]",
];

const NUM_BARS = 4; // top 3 + Others
const NUM_BAR_SEG = 4; // one segment per thruster pair
const MAX_SHAP_VALUE = 4;
const INCREMENT = 1;
const WINDOW_PADDING = 40;
const BAR_GAP = 25;
const LABEL_OFFSET = 10;
const NUM_FEATURES = 14;

// ── Helpers ────────────────────────────────────────────────────────────

/** Clamp v to [lo, hi]. */
function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Convert SHAP magnitude to pixel width within the bar area. */
function shapValueToPixels(value: number, barAllocationWidth: number): number {
  return (value / MAX_SHAP_VALUE) * barAllocationWidth;
}

/**
 * Compute the 4x14 distance matrix and normalisation factor `c`.
 *
 * Mirrors _draw_bars2 (and _find_explanation2) from the Python code:
 *   1. sv_all_actions = clip(sv_action.sum(axis=1) + sv_base, act_low, act_high)
 *   2. c = sv_all_actions / (pos_total or neg_total depending on act_high)
 *   3. reshape (8,14) -> (4,2,14), scale by c, Euclidean distance, clip to [0,1]
 */
function computeDistances(
  svAction: number[][],
  baseValue: number[],
  actionLow: number[],
  actionHigh: number[],
): number[][] {
  const numActions = svAction.length; // 8
  const numFeats = svAction[0].length; // 14

  // sv_all_actions = clip(sv_action.sum(axis=1) + sv_base, act_low, act_high)
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

  // c = sv_all_actions / denominator  (where denominator depends on act_high)
  const c: number[] = new Array(numActions);
  for (let a = 0; a < numActions; a++) {
    const denom = actionHigh[a] === 1 ? posTotal[a] : negTotal[a];
    c[a] = denom !== 0 ? svAllActions[a] / denom : 0;
  }

  // Reshape into pairs (4, 2, 14) and scale
  const numPairs = numActions / 2;
  // dists: shape (4, 14)
  const dists: number[][] = [];
  for (let p = 0; p < numPairs; p++) {
    const row: number[] = new Array(numFeats);
    const xIdx = p * 2;
    const yIdx = p * 2 + 1;
    for (let f = 0; f < numFeats; f++) {
      const scaledX = svAction[xIdx][f] * c[xIdx];
      const scaledY = svAction[yIdx][f] * c[yIdx];
      row[f] = clamp(Math.hypot(scaledX, scaledY), 0, 1);
    }
    dists.push(row);
  }

  return dists;
}

// ── Legend drawing ──────────────────────────────────────────────────────

function drawLegend(
  r: Renderer,
  ox: number,
  oy: number,
  w: number,
  h: number,
  items: string[],
  padding: number,
): void {
  const markerSize = 12;
  const spacing = 5;
  const legendPadding = 10;
  const fontSize = 12;
  const lineHeight = 14;

  const maxTextWidth = Math.max(...items.map((label) => r.measureText(label, fontSize)));
  const boxW = legendPadding * 2 + markerSize + spacing + maxTextWidth;
  const totalContentH =
    items.length * lineHeight + (items.length - 1) * spacing;
  const boxH = legendPadding * 2 + totalContentH;

  // Position: bottom-right with margin, then up by padding_bottom=60
  const margin = padding / 2;
  const paddingBottom = 60;
  const boxX = ox + w - margin - boxW;
  const boxY = oy + h - margin - paddingBottom - boxH;

  // Semi-transparent background
  r.drawRect(boxX, boxY, boxW, boxH, Colors.LEGEND_BOX);

  let yOff = boxY + legendPadding;
  for (let i = 0; i < items.length; i++) {
    const markerY = yOff + (lineHeight - markerSize) / 2;
    r.drawRect(
      boxX + legendPadding,
      markerY,
      markerSize,
      markerSize,
      BAR_COLORS[i],
    );
    r.drawText(
      items[i],
      boxX + legendPadding + markerSize + spacing,
      yOff,
      Colors.BLACK,
      fontSize,
    );
    yOff += lineHeight + spacing;
  }
}

// ── Main render function ───────────────────────────────────────────────

export function renderShapBars(
  r: Renderer,
  frame: RenderFrame,
  ox: number,
  oy: number,
  w: number,
  h: number,
  _mode: "thrust" | "angle",
): void {
  // White background
  r.drawRect(ox, oy, w, h, Colors.WHITE);

  const shap = frame.shap;
  const svAction = shap.shap_values_action; // (8, 14)
  const baseValue = shap.base_vectors; // (8,)
  const actionLow = shap.action_low; // (8,)
  const actionHigh = shap.action_high; // (8,)

  // ── Layout computation (mirrors Python __init__) ───────────────────

  // Approximate axis_padding_width: widest feature label + label_offset
  // Python uses pygame font metrics; we approximate with 12px * ~10 chars
  const axisPaddingWidth = 110 + LABEL_OFFSET;

  const barAllocationWidth = w - WINDOW_PADDING - axisPaddingWidth;

  // Bottom text height approximation (fontSize 13 + tick labels 12 + margins)
  const bottomTextHeight = 16;
  const tickLabelHeight = 14;
  const axisPaddingHeight =
    bottomTextHeight + 5 + tickLabelHeight + LABEL_OFFSET;

  const barAllocationHeight =
    (h - WINDOW_PADDING - axisPaddingHeight) / NUM_BARS;
  const barHeight = barAllocationHeight - BAR_GAP;

  // Origin (bottom-left of bar chart)
  const origoX = ox + WINDOW_PADDING / 2 + axisPaddingWidth;
  const origoY = oy + h - WINDOW_PADDING / 2 - axisPaddingHeight;

  // Top of y-axis
  const yAxisEndY = oy + WINDOW_PADDING / 2;

  // Right end of x-axis
  const xAxisEndX = ox + w - WINDOW_PADDING / 2;

  // ── Compute SHAP distances ────────────────────────────────────────

  const distsClipped = computeDistances(
    svAction,
    baseValue,
    actionLow,
    actionHigh,
  );

  // Sum across 4 thruster pairs for each feature -> (14,)
  const sumedFeatures: number[] = new Array(NUM_FEATURES).fill(0);
  for (let f = 0; f < NUM_FEATURES; f++) {
    for (let p = 0; p < 4; p++) {
      sumedFeatures[f] += distsClipped[p][f];
    }
  }

  // Find top-3 feature indices by total contribution (descending)
  const indices = Array.from({ length: NUM_FEATURES }, (_, i) => i);
  indices.sort((a, b) => sumedFeatures[b] - sumedFeatures[a]);
  const top3Idx = indices.slice(0, 3);

  const othersIdx = indices.slice(3);

  // Draw order: top3 + "Others" sentinel (-1)
  const drawOrder: number[] = [...top3Idx, -1];

  // ── Draw bars ─────────────────────────────────────────────────────

  for (let barNo = 0; barNo < drawOrder.length; barNo++) {
    const featIdx = drawOrder[barNo];
    const barY = oy + WINDOW_PADDING / 2 + barNo * barAllocationHeight;

    // Build 4 segment widths
    const segWidths: number[] = [];
    for (let seg = 0; seg < NUM_BAR_SEG; seg++) {
      let segVal: number;
      if (featIdx === -1) {
        // Others: sum across all residual features
        segVal = 0;
        for (const jj of othersIdx) {
          segVal += distsClipped[seg][jj];
        }
      } else {
        segVal = distsClipped[seg][featIdx];
      }
      segWidths.push(shapValueToPixels(segVal, barAllocationWidth));
    }

    // Draw stacked segments left-to-right from axis origin
    let x0 = origoX;
    for (let seg = 0; seg < NUM_BAR_SEG; seg++) {
      if (segWidths[seg] > 0.5) {
        r.drawRect(x0, barY, segWidths[seg], barHeight, BAR_COLORS[seg]);
      }
      x0 += segWidths[seg];
    }

    // Feature label (right-aligned to left of bar area)
    const label = featIdx === -1 ? "Others" : FEATURE_NAMES[featIdx];
    r.drawText(
      label,
      origoX - LABEL_OFFSET,
      barY + barHeight / 2,
      Colors.BLACK,
      12,
      "right",
      "middle",
    );
  }

  // ── Draw axes ─────────────────────────────────────────────────────

  // X-axis (horizontal)
  r.drawLine(origoX, origoY, xAxisEndX, origoY, Colors.BLACK, 2);
  // Y-axis (vertical)
  r.drawLine(origoX, origoY, origoX, yAxisEndY, Colors.BLACK, 2);

  // Tick marks and labels
  const numTicks = MAX_SHAP_VALUE / INCREMENT + 1;
  // Bottom text position — Python: bottom_text_rect.midbottom at (mid, h - padding/2)
  // tick_rect.midbottom at (tickPx, bottom_text_rect.midtop[1] - 5)
  const bottomTextY = oy + h - WINDOW_PADDING / 2;
  const tickLabelY = bottomTextY - bottomTextHeight - 10;

  for (let tick = 0; tick < numTicks; tick++) {
    const tickValue = tick * INCREMENT;
    const tickPx =
      WINDOW_PADDING / 2 + axisPaddingWidth +
      shapValueToPixels(tickValue, barAllocationWidth);
    const tickScreenX = ox + tickPx;

    // Tick mark
    r.drawLine(
      tickScreenX,
      origoY - 5,
      tickScreenX,
      origoY + 5,
      Colors.BLACK,
      2,
    );

    // Tick label
    const rpmLabel = `${tickValue * MAX_THRUSTER_RPM}`;
    r.drawText(rpmLabel, tickScreenX, tickLabelY, Colors.BLACK, 12, "center");
  }

  // Bottom text
  const bottomTextX = ox + (w + axisPaddingWidth) / 2;
  r.drawText(
    "Sum of SHAP based thrust components",
    bottomTextX,
    bottomTextY - bottomTextHeight / 2,
    Colors.BLACK,
    13,
    "center",
    "middle",
  );

  // ── Legend ─────────────────────────────────────────────────────────

  const legendItems =
    _mode === "thrust" ? LEGEND_ITEMS_THRUST : LEGEND_ITEMS_ANGLE;
  drawLegend(r, ox, oy, w, h, legendItems, WINDOW_PADDING);

  // ── Title (top-right, matching Python render_surface_title) ───────

  const title =
    _mode === "thrust"
      ? "Feature importance using thrust SHAP-values"
      : "SHAP-values azimuth angles";
  r.drawText(title, ox + w - 5, oy + 5, Colors.GRAY, 11, "right");
}
