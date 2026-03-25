/**
 * Main thread entry point.
 *
 * Handles:
 * - WebSocket connection to the FastAPI server (with auto-reconnect)
 * - MessagePack decoding of incoming RenderFrame payloads
 * - Posting decoded frames to the render Worker via postMessage
 * - Keyboard events (keys 0-5) sent to the server as KeyboardInput
 * - Connection status indicator
 */

import { decode, encode } from "@msgpack/msgpack";
import type { KeyboardInput, RenderFrame, WorkerMessage } from "./types.ts";

// --- DOM setup ---
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const statusDot = document.getElementById("status-dot") as HTMLElement;
const statusText = document.getElementById("status-text") as HTMLElement;

// Transfer canvas to Worker
const offscreen = canvas.transferControlToOffscreen();
const worker = new Worker(new URL("./worker.ts", import.meta.url), {
  type: "module",
});
const initMsg: WorkerMessage = { type: "init", canvas: offscreen };
worker.postMessage(initMsg, [offscreen]);

// --- WebSocket connection ---
let ws: WebSocket | null = null;
let reconnectDelay = 500; // ms, exponential backoff
const MAX_RECONNECT_DELAY = 5000;

function getWsUrl(): string {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}/ws`;
}

function setStatus(connected: boolean): void {
  statusDot.className = connected ? "status-dot connected" : "status-dot";
  statusText.textContent = connected ? "Connected" : "Disconnected";
}

function connect(): void {
  const url = getWsUrl();
  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    console.log("WebSocket connected");
    setStatus(true);
    reconnectDelay = 500; // Reset backoff
  };

  ws.onmessage = (event: MessageEvent) => {
    const data = event.data as ArrayBuffer;
    const frame = decode(new Uint8Array(data)) as RenderFrame;
    const msg: WorkerMessage = { type: "frame", frame };
    worker.postMessage(msg);
  };

  ws.onclose = () => {
    console.log(`WebSocket disconnected, reconnecting in ${reconnectDelay}ms`);
    setStatus(false);
    ws = null;
    setTimeout(() => {
      reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY);
      connect();
    }, reconnectDelay);
  };

  ws.onerror = (err) => {
    console.error("WebSocket error:", err);
    ws?.close();
  };
}

connect();

// --- Keyboard input ---
const VALID_KEYS = new Set(["0", "1", "2", "3", "4", "5"]);

document.addEventListener("keydown", (event: KeyboardEvent) => {
  if (!VALID_KEYS.has(event.key)) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  const input: KeyboardInput = { key: parseInt(event.key, 10) };
  const packed = encode(input);
  ws.send(packed);
});
