import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built bundle is served by viz/server.py from ./dist. Absolute base "/" — the
// static server maps "/" -> dist/index.html and "/assets/*" -> dist/assets/*.
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: { outDir: "dist", emptyOutDir: true },
  server: { port: 5173 },
});
