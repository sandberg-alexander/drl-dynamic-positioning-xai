/**
 * Tests for coordinate transforms.
 * Verifies the TypeScript port matches the Python implementation.
 */

import { describe, expect, it } from "vitest";
import {
  R2,
  Viewport,
  applyTransform,
  degreesToCanvas,
  transformPoints,
} from "../transforms.ts";

describe("R2", () => {
  it("identity at psi=0 with scale=1", () => {
    const r = R2(0, 1);
    // sin(0)=0, cos(0)=1 => [0, 1, 1, 0]
    expect(r[0]).toBeCloseTo(0); // sin
    expect(r[1]).toBeCloseTo(1); // cos
    expect(r[2]).toBeCloseTo(1); // cos
    expect(r[3]).toBeCloseTo(0); // -sin
  });

  it("90 degrees rotation", () => {
    const r = R2(Math.PI / 2, 1);
    expect(r[0]).toBeCloseTo(1); // sin(pi/2) = 1
    expect(r[1]).toBeCloseTo(0); // cos(pi/2) = 0
    expect(r[2]).toBeCloseTo(0);
    expect(r[3]).toBeCloseTo(-1);
  });

  it("scales correctly", () => {
    const r = R2(0, 10);
    expect(r[1]).toBeCloseTo(10); // cos * scale
  });
});

describe("applyTransform", () => {
  it("translation only", () => {
    const r: [number, number, number, number] = [0, 1, 1, 0]; // identity-ish
    const t: [number, number] = [100, 200];
    const [x, y] = applyTransform(r, t, 0, 0);
    expect(x).toBeCloseTo(100);
    expect(y).toBeCloseTo(200);
  });

  it("rotation + translation", () => {
    const r = R2(0, 10);
    const t: [number, number] = [50, 50];
    const [x, y] = applyTransform(r, t, 1, 0);
    // r = [0, 10, 10, 0], so x = 0*1 + 10*0 + 50 = 50, y = 10*1 + 0*0 + 50 = 60
    expect(x).toBeCloseTo(50);
    expect(y).toBeCloseTo(60);
  });
});

describe("transformPoints", () => {
  it("transforms array of points", () => {
    const r = R2(0, 1);
    const t: [number, number] = [0, 0];
    const pts: [number, number][] = [
      [1, 0],
      [0, 1],
    ];
    const result = transformPoints(r, t, pts);
    expect(result).toHaveLength(2);
  });
});

describe("Viewport", () => {
  it("center maps to center pixel", () => {
    const vp = new Viewport(10, 200, 200);
    const [x, y] = vp.worldToPixels(0, 0);
    expect(x).toBeCloseTo(100);
    expect(y).toBeCloseTo(100);
  });

  it("metersToPixels scales correctly", () => {
    const vp = new Viewport(15, 100, 100);
    expect(vp.metersToPixels(2)).toBeCloseTo(30);
  });
});

describe("degreesToCanvas", () => {
  it("0 deg (north) -> -pi/2 (canvas up)", () => {
    expect(degreesToCanvas(0)).toBeCloseTo(-Math.PI / 2);
  });

  it("90 deg (east) -> 0 (canvas right)", () => {
    expect(degreesToCanvas(90)).toBeCloseTo(0);
  });
});
