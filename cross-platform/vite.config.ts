import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/postcss';
import { defineConfig } from 'vite';

// The interface is a plain client-side app. `npm run build` writes it to ui/,
// which local_worker.py serves itself - end users need neither Node.js nor npm.
export default defineConfig({
  plugins: [react()],
  css: { postcss: { plugins: [tailwindcss()] } },
  build: { outDir: 'ui', emptyOutDir: true },
  server: { host: 'localhost', port: 8765 },
});
