import { unplugin as stylex } from '@stylexjs/unplugin';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

// The browser talks to /api on the dev server's own origin, and Vite forwards it to uvicorn:
// same origin means no CORS layer to configure, and the deployed build can use the same paths.
export default defineConfig({
  // StyleX keeps a dev CSS server alive, which holds the vitest process open for ten seconds
  // after the run; the unit tests are plain TypeScript and compile no styles.
  plugins: process.env.VITEST
    ? [react()]
    : [stylex.vite({ useCSSLayers: true }), react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
});
