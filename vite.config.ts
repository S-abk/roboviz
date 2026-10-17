// vite.config.ts
import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  server: {
    host: '0.0.0.0', // Listens on all local network interfaces (LAN)
    port: 5173,
    strictPort: true,
    cors: true,
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});
