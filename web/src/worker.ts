/**
 * OffscreenCanvas Web Worker for rendering.
 *
 * Receives RenderFrame payloads via postMessage from the main thread.
 * Always renders the latest frame (latest-frame-wins buffer).
 * Uses requestAnimationFrame for smooth, V-sync-aligned rendering.
 */

import { Colors } from "./colors.ts";
import { Renderer } from "./renderer.ts";
import type { RenderFrame, WorkerMessage } from "./types.ts";
import { renderBody } from "./views/body.ts";
import { renderNed } from "./views/ned.ts";
import { renderShapBars } from "./views/shap_bars.ts";
import { renderShapExplain } from "./views/shap_explain.ts";

let renderer: Renderer | null = null;
let canvas: OffscreenCanvas | null = null;
let latestFrame: RenderFrame | null = null;
let animating = false;

// Layout constants matching pygame compositor (975x975)
const SCREEN_WIDTH = 975;
const SCREEN_HEIGHT = 975;

// Sub-window positions and sizes
const NED_X = 25,
  NED_Y = 25,
  NED_W = 450,
  NED_H = 450;
const BODY_X = 25,
  BODY_Y = 500,
  BODY_W = 450,
  BODY_H = 450;
const SHAP_TOP_X = 500,
  SHAP_TOP_Y = 25,
  SHAP_TOP_W = 450,
  SHAP_TOP_H = 450;
const SHAP_BOTTOM_X = 500,
  SHAP_BOTTOM_Y = 500,
  SHAP_BOTTOM_W = 450,
  SHAP_BOTTOM_H = 450;

function renderLoop(): void {
  if (!renderer || !canvas || !latestFrame) {
    if (animating) requestAnimationFrame(renderLoop);
    return;
  }

  const frame = latestFrame;

  // Background
  const bgColor =
    sum(frame.shap.shap_values_value) < -2 ? Colors.RED : Colors.SCREEN_COLOR;
  renderer.beginFrame(bgColor);

  // Time step text (centered at bottom)
  renderer.drawText(
    `time step: ${frame.time_step}, time: ${frame.time_seconds} s`,
    SCREEN_WIDTH / 2,
    SCREEN_HEIGHT - 15,
    Colors.BLACK,
    11,
    "center",
    "middle",
  );

  // Render sub-windows (each clipped to its bounds, matching pygame Surface clipping)
  renderer.clipRect(NED_X, NED_Y, NED_W, NED_H);
  renderNed(renderer, frame, NED_X, NED_Y, NED_W, NED_H);
  renderer.resetClip();

  renderer.clipRect(BODY_X, BODY_Y, BODY_W, BODY_H);
  renderBody(renderer, frame, BODY_X, BODY_Y, BODY_W, BODY_H);
  renderer.resetClip();

  renderer.clipRect(SHAP_TOP_X, SHAP_TOP_Y, SHAP_TOP_W, SHAP_TOP_H);
  renderShapBars(
    renderer,
    frame,
    SHAP_TOP_X,
    SHAP_TOP_Y,
    SHAP_TOP_W,
    SHAP_TOP_H,
    "thrust",
  );
  renderer.resetClip();

  renderer.clipRect(SHAP_BOTTOM_X, SHAP_BOTTOM_Y, SHAP_BOTTOM_W, SHAP_BOTTOM_H);
  renderShapExplain(
    renderer,
    frame,
    SHAP_BOTTOM_X,
    SHAP_BOTTOM_Y,
    SHAP_BOTTOM_W,
    SHAP_BOTTOM_H,
  );
  renderer.resetClip();

  renderer.endFrame();

  if (animating) requestAnimationFrame(renderLoop);
}

function sum(arr: number[]): number {
  let s = 0;
  for (const v of arr) s += v;
  return s;
}

self.onmessage = (evt: MessageEvent<WorkerMessage>) => {
  const msg = evt.data;

  if (msg.type === "init") {
    canvas = msg.canvas;
    canvas.width = SCREEN_WIDTH;
    canvas.height = SCREEN_HEIGHT;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      console.error("Failed to get 2D context from OffscreenCanvas");
      return;
    }
    renderer = new Renderer(ctx);
    animating = true;
    requestAnimationFrame(renderLoop);
  } else if (msg.type === "frame") {
    // Latest-frame-wins: just overwrite the buffer
    latestFrame = msg.frame;
  } else if (msg.type === "resize") {
    if (canvas) {
      canvas.width = msg.width;
      canvas.height = msg.height;
    }
  }
};
