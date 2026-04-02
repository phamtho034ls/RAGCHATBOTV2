import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        // Tránh buffer SSE khi dev qua Vite — nếu không, cả response stream có thể tới một lần.
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes, req) => {
            if (
              req.url?.includes("/stream") &&
              proxyRes.headers["content-type"]?.includes("text/event-stream")
            ) {
              proxyRes.headers["x-accel-buffering"] = "no";
              delete proxyRes.headers["content-length"];
            }
          });
        },
      },
    },
  },
});
