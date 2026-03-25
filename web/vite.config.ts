import { defineConfig } from "vite";

export default defineConfig({
  build: {
    outDir: "dist",
  },
  server: {
    proxy: {
      "/ws": {
        target: "ws://localhost:8080",
        ws: true,
      },
    },
  },
});
