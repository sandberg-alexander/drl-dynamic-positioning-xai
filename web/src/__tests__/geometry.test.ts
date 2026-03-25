/**
 * Tests for vessel geometry generation.
 */

import { describe, expect, it } from "vitest";
import {
  VESSEL_BEAM,
  VESSEL_LENGTH,
  vesselHullPolygon,
  vesselMomentMarkerLine,
  vesselTrianglePolygon,
} from "../geometry.ts";

describe("vesselHullPolygon", () => {
  it("returns 8 points", () => {
    const hull = vesselHullPolygon();
    expect(hull).toHaveLength(8);
  });

  it("is symmetric about x-axis", () => {
    const hull = vesselHullPolygon();
    // Top and bottom halves mirror across y=0
    // Point 0 (top-right) and 3 (bottom-right) share same x, opposite y
    expect(hull[0][0]).toBeCloseTo(hull[3][0]);
    expect(hull[0][1]).toBeCloseTo(-hull[3][1]);
  });

  it("stays within vessel bounds", () => {
    const hull = vesselHullPolygon();
    const hl = VESSEL_LENGTH / 2;
    const hb = VESSEL_BEAM / 2;
    for (const [x, y] of hull) {
      expect(Math.abs(x)).toBeLessThanOrEqual(hl + 0.001);
      expect(Math.abs(y)).toBeLessThanOrEqual(hb + 0.001);
    }
  });

  it("accepts custom dimensions", () => {
    const hull = vesselHullPolygon(10, 5, 1);
    expect(hull).toHaveLength(8);
    expect(hull[0][0]).toBeCloseTo(4); // 10/2 - 1
  });
});

describe("vesselTrianglePolygon", () => {
  it("returns 3 points", () => {
    const tri = vesselTrianglePolygon();
    expect(tri).toHaveLength(3);
  });

  it("tip is at bow", () => {
    const tri = vesselTrianglePolygon();
    expect(tri[0][0]).toBeCloseTo(VESSEL_LENGTH / 2);
    expect(tri[0][1]).toBeCloseTo(0);
  });
});

describe("vesselMomentMarkerLine", () => {
  it("returns 2 points", () => {
    const line = vesselMomentMarkerLine();
    expect(line).toHaveLength(2);
  });

  it("is centered at origin", () => {
    const line = vesselMomentMarkerLine();
    expect(line[0][0] + line[1][0]).toBeCloseTo(0);
  });
});
