// Vite dev server for EventLab. The browser calls relative /api/... URLs; Vite forwards them to the FastAPI backend.
// Both stay on 127.0.0.1: the API can control the camera and has no login.
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const BACKEND = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: BACKEND,
        // backend not running: answer 502 with a JSON body, so the app can show "Offline" (src/api.ts)
        configure: (proxy) => {
          proxy.on("error", (_err, _req, res) => {
            if ("writeHead" in res && !res.headersSent) {
              res.writeHead(502, { "Content-Type": "application/json" });
              res.end(JSON.stringify({ detail: "backend unreachable" }));
            }
          });
        },
      },
    },
  },
});
