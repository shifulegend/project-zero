import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// Phase 22.2: builds to dist/ as a single relocatable static bundle (assets
// under /assets/, referenced with relative paths) so it can be served from
// any path by the embedded C HTTP server (see src/api/static_assets.c) or
// from disk during development via --static-dir.
export default defineConfig({
  plugins: [svelte()],
  base: './',
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/v1': 'http://127.0.0.1:8080',
      '/health': 'http://127.0.0.1:8080',
      '/metrics': 'http://127.0.0.1:8080',
    },
  },
});
