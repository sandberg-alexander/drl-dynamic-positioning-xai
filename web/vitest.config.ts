import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    setupFiles: ["@vitest/web-worker"],
    include: ["src/__tests__/**/*.test.ts"],
  },
});
