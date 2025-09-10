const path = require('path');
const fs = require('fs');

(async () => {
  // Use Playwright from Playwright MCP node_modules to avoid extra install
  const pwPath = 'C:\\mcp-servers\\playwright-mcp\\node_modules\\playwright';
  const { chromium } = require(pwPath);

  const outPath = path.join(process.cwd(), 'playwright-google.png');

  let browser;
  try {
    // Prefer system Edge to avoid downloading browsers
    browser = await chromium.launch({ channel: 'msedge', headless: true });
  } catch (e) {
    // Fallback to bundled Chromium (may require installed browsers)
    browser = await chromium.launch({ headless: true });
  }
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 }, ignoreHTTPSErrors: true });
  const page = await context.newPage();

  await page.goto('https://www.google.com', { waitUntil: 'domcontentloaded', timeout: 60000 });

  // Try to accept cookies if banner appears (best-effort)
  const candidates = [
    'button:has-text("I agree")',
    'button:has-text("Accept all")',
    'button:has-text("Принять все")',
    'button:has-text("Согласен")',
    '[aria-label="Accept all"]',
    '[aria-label="Принять все"]',
  ];
  for (const sel of candidates) {
    try {
      const btn = page.locator(sel).first();
      if (await btn.count()) {
        await btn.click({ timeout: 2000 });
        break;
      }
    } catch {}
  }

  await page.screenshot({ path: outPath, fullPage: false });
  const title = await page.title();
  await browser.close();

  console.log(JSON.stringify({ ok: true, title, screenshot: outPath }));
})();
