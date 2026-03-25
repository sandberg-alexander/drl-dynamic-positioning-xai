/**
 * Canvas 2D renderer mirroring the Python Renderer protocol.
 *
 * All methods operate in pixel coordinates — coordinate transforms
 * (world-to-pixel) are applied before calling these methods.
 */

export class Renderer {
  private ctx: OffscreenCanvasRenderingContext2D;

  constructor(ctx: OffscreenCanvasRenderingContext2D) {
    this.ctx = ctx;
  }

  /** Set a clipping rectangle. All drawing is clipped until resetClip(). */
  clipRect(x: number, y: number, w: number, h: number): void {
    this.ctx.save();
    this.ctx.beginPath();
    this.ctx.rect(x, y, w, h);
    this.ctx.clip();
  }

  /** Remove the clipping rectangle set by clipRect(). */
  resetClip(): void {
    this.ctx.restore();
  }

  beginFrame(bgColor: string): void {
    const { width, height } = this.ctx.canvas;
    this.ctx.fillStyle = bgColor;
    this.ctx.fillRect(0, 0, width, height);
  }

  endFrame(): void {
    // Canvas 2D is immediate mode — no-op.
  }

  drawPolygon(
    points: [number, number][],
    color: string,
    fill = true,
    lineWidth = 1,
  ): void {
    if (points.length < 2) return;
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.closePath();
    if (fill) {
      ctx.fillStyle = color;
      ctx.fill();
    } else {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }
  }

  drawCircle(
    cx: number,
    cy: number,
    radius: number,
    color: string,
    fill = true,
    lineWidth = 1,
  ): void {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    if (fill) {
      ctx.fillStyle = color;
      ctx.fill();
    } else {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }
  }

  drawLine(
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    color: string,
    lineWidth = 1,
  ): void {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  }

  drawLines(
    points: [number, number][],
    color: string,
    closed = false,
    lineWidth = 1,
  ): void {
    if (points.length < 2) return;
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i][0], points[i][1]);
    }
    if (closed) ctx.closePath();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  }

  drawDashedLine(
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    color: string,
    dashLength = 10,
    gapLength = 5,
    lineWidth = 2,
  ): void {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.setLineDash([dashLength, gapLength]);
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
    ctx.setLineDash([]); // Reset
  }

  drawRect(
    x: number,
    y: number,
    width: number,
    height: number,
    color: string,
    fill = true,
    lineWidth = 1,
  ): void {
    const ctx = this.ctx;
    if (fill) {
      ctx.fillStyle = color;
      ctx.fillRect(x, y, width, height);
    } else {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.strokeRect(x, y, width, height);
    }
  }

  /** Measure text width in pixels at given font size. */
  measureText(text: string, fontSize = 11): number {
    this.ctx.font = `${fontSize}px "DejaVu Sans", sans-serif`;
    return this.ctx.measureText(text).width;
  }

  drawText(
    text: string,
    x: number,
    y: number,
    color: string,
    fontSize = 11,
    align: CanvasTextAlign = "left",
    baseline: CanvasTextBaseline = "top",
  ): void {
    const ctx = this.ctx;
    ctx.font = `${fontSize}px "DejaVu Sans", sans-serif`;
    ctx.fillStyle = color;
    ctx.textAlign = align;
    ctx.textBaseline = baseline;
    ctx.fillText(text, x, y);
  }

  drawArrow(
    startX: number,
    startY: number,
    angle: number,
    length: number,
    color: string,
    headLength = 10,
    headWidth = 10,
    lineWidth = 2,
  ): void {
    if (length < 0) {
      length = -length;
      angle += Math.PI;
    }

    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);

    let lineLen: number;
    let curHeadLen: number;
    let curHeadWidth = headWidth;

    if (length > headLength) {
      lineLen = length - headLength;
      curHeadLen = headLength;
    } else {
      lineLen = 0;
      curHeadLen = length;
      curHeadWidth *= length / headLength;
    }

    const lineEndX = startX + lineLen * cosA;
    const lineEndY = startY + lineLen * sinA;
    const tipX = lineEndX + curHeadLen * cosA;
    const tipY = lineEndY + curHeadLen * sinA;

    const perpX = -sinA;
    const perpY = cosA;
    const leftX = lineEndX + (curHeadWidth / 2) * perpX;
    const leftY = lineEndY + (curHeadWidth / 2) * perpY;
    const rightX = lineEndX - (curHeadWidth / 2) * perpX;
    const rightY = lineEndY - (curHeadWidth / 2) * perpY;

    if (lineLen > 0) {
      this.drawLine(startX, startY, lineEndX, lineEndY, color, lineWidth);
    }
    this.drawPolygon(
      [
        [tipX, tipY],
        [leftX, leftY],
        [rightX, rightY],
      ],
      color,
    );
  }
}
