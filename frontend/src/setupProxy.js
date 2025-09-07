const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://backend:8000', // Using service name for Docker networking
      changeOrigin: true,
      secure: false,
      onProxyReq: (proxyReq, req, res) => {
        // Add CORS headers
        proxyReq.setHeader('Access-Control-Allow-Origin', 'http://localhost:3000');
        proxyReq.setHeader('Access-Control-Allow-Credentials', 'true');
      },
      onProxyRes: function(proxyRes, req, res) {
        proxyRes.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000';
        proxyRes.headers['Access-Control-Allow-Credentials'] = 'true';
      },
      logLevel: 'debug',
      // Add websocket support
      ws: true,
      // Don't verify SSL certificates
      rejectUnauthorized: false
    })
  );
};
