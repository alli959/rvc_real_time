/**
 * MorphVox Maintenance Worker
 *
 * Transparently proxies requests to the origin. If the origin is
 * unreachable or returns a gateway error (502-524), a branded
 * maintenance page is served instead.
 *
 * Deploy: wrangler deploy -c infra/cloudflare/wrangler.toml
 */

const MAINTENANCE_HTML = `<!DOCTYPE html>
<html>
<head>
  <title>MorphVox - Maintenance</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      color: white;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      margin: 0;
    }
    .container { text-align: center; max-width: 500px; padding: 40px; }
    h1 { font-size: 2.5rem; margin-bottom: 1rem; }
    p { color: #a0aec0; font-size: 1.1rem; line-height: 1.6; }
    .icon { font-size: 4rem; margin-bottom: 1rem; }
    .status {
      background: rgba(255,255,255,0.1);
      padding: 15px 30px;
      border-radius: 10px;
      margin-top: 20px;
      display: inline-block;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="icon">🔧</div>
    <h1>We'll Be Right Back</h1>
    <p>MorphVox is currently undergoing maintenance. We're working hard to improve your experience.</p>
    <div class="status">
      <strong>Status:</strong> Maintenance Mode
    </div>
  </div>
</body>
</html>`;

function maintenanceResponse() {
  return new Response(MAINTENANCE_HTML, {
    status: 503,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Retry-After": "60",
      "Cache-Control": "no-store",
    },
  });
}

export default {
  async fetch(request) {
    try {
      const response = await fetch(request);

      // Origin returned a gateway / server-down error — show maintenance
      if (response.status >= 502 && response.status <= 524) {
        return maintenanceResponse();
      }

      return response;
    } catch {
      // Origin is unreachable (DNS failure, connection refused, timeout, etc.)
      return maintenanceResponse();
    }
  },
};
