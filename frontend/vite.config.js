import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The repo layout (section L of the plan) uses .js filenames (App.js, index.js),
// so tell Vite/esbuild that .js files in src/ contain JSX.
export default defineConfig({
  plugins: [react({ include: /\.(js|jsx)$/ })],
  esbuild: { loader: "jsx", include: /src\/.*\.jsx?$/, exclude: [] },
  optimizeDeps: { esbuildOptions: { loader: { ".js": "jsx" } } },
  server: { port: 3000 },
});
