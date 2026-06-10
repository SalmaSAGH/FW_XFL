import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // Use relative paths for production builds (works with nginx subfolder)
  base: './',
  server: {
    port: 3000,
    proxy: {
      '/api': {
        // In Docker: proxy to dashboard (which proxies to server)
        // For local dev: use dashboard at 5001
        target: process.env.VITE_API_URL || 'http://localhost:5001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
