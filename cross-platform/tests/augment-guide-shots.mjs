// Makes the screenshots for the Augmentation Studio guide: public/augment-guide/<language>-<step>.jpg.
// Runs the real studio from a build (npm run build -> ui/) against a mocked worker. Every picture
// in it is drawn here as SVG: never use real training images, they must not end up on GitHub.
//
//   CHROME_PATH=... PLAYWRIGHT_MODULE=.../playwright-core/index.mjs node tests/augment-guide-shots.mjs ui
import { createServer } from 'node:http';
import { mkdir, readFile } from 'node:fs/promises';
import { extname, resolve } from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const root = resolve(process.argv[2] || 'ui');
const out = resolve(process.argv[3] || 'public/augment-guide');
await mkdir(out, { recursive: true });

const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.jpg': 'image/jpeg' };
const server = createServer(async (request, response) => {
  try {
    const path = resolve(root, '.' + (request.url === '/' ? '/index.html' : request.url.split('?')[0]));
    response.setHeader('Content-Type', types[extname(path)] || 'application/octet-stream');
    response.end(await readFile(path));
  } catch { response.statusCode = 404; response.end(); }
});
await new Promise((done) => server.listen(0, '127.0.0.1', done));
const url = `http://127.0.0.1:${server.address().port}/`;

// --- Drawings -------------------------------------------------------------------------------
const W = 1600, H = 900;
const names = { de: ['Heimspiel 12.10.', 'Auswärts 19.10.', 'Turnier 02.11.'], en: ['Home game 12 Oct', 'Away game 19 Oct', 'Tournament 2 Nov'], es: ['En casa 12/10', 'Fuera 19/10', 'Torneo 02/11'], fr: ['Domicile 12/10', 'Extérieur 19/10', 'Tournoi 02/11'] };
const halls = [
  { name: 'Heimspiel 12.10.', floor: ['#c98b4f', '#a86a35'], wall: '#3a3f4a', teams: ['#1f4fa0', '#f2f2f2'] },
  { name: 'Auswärts 19.10.', floor: ['#d9a564', '#b97c3d'], wall: '#4a3a3a', teams: ['#b3261e', '#1d1d1d'] },
  { name: 'Turnier 02.11.', floor: ['#7f9bb3', '#5e7b93'], wall: '#2f3a44', teams: ['#2e7d32', '#f2c230'] },
];
const court = [[260, 380], [1340, 380], [1560, 860], [40, 860]];
function hall(index) {
  const h = halls[index];
  return `<defs><linearGradient id="f" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${h.floor[0]}"/><stop offset="1" stop-color="${h.floor[1]}"/></linearGradient></defs>
<rect width="${W}" height="${H}" fill="${h.wall}"/><rect y="300" width="${W}" height="${H - 300}" fill="url(#f)"/>
<rect x="0" y="230" width="${W}" height="70" fill="#00000033"/>${Array.from({ length: 16 }, (_, i) => `<rect x="${30 + i * 98}" y="248" width="70" height="34" rx="4" fill="#ffffff22"/>`).join('')}
<polygon points="${court.map((p) => p.join(',')).join(' ')}" fill="none" stroke="#f3efe8" stroke-width="5"/>
<line x1="800" y1="380" x2="800" y2="860" stroke="#f3efe8" stroke-width="5"/><ellipse cx="800" cy="560" rx="150" ry="55" fill="none" stroke="#f3efe8" stroke-width="5"/>
<path d="M260 380 Q 330 520 160 620" fill="none" stroke="#f3efe8" stroke-width="4"/><path d="M1340 380 Q 1270 520 1440 620" fill="none" stroke="#f3efe8" stroke-width="4"/>
<rect x="150" y="170" width="10" height="210" fill="#ddd"/><rect x="120" y="165" width="80" height="50" fill="#fff" stroke="#333" stroke-width="3"/><ellipse cx="190" cy="225" rx="22" ry="7" fill="none" stroke="#e2692a" stroke-width="4"/>
<rect x="1440" y="170" width="10" height="210" fill="#ddd"/><rect x="1400" y="165" width="80" height="50" fill="#fff" stroke="#333" stroke-width="3"/><ellipse cx="1410" cy="225" rx="22" ry="7" fill="none" stroke="#e2692a" stroke-width="4"/>`;
}
// A simple figure standing with its feet at (x, y), height h.
function figure(x, y, h, shirt, shorts = '#222', skin = '#d7a17a') {
  const w = h * 0.36, head = h * 0.13;
  return `<g><rect x="${x - w * 0.32}" y="${y - h * 0.42}" width="${w * 0.24}" height="${h * 0.42}" rx="${w * 0.1}" fill="${skin}"/><rect x="${x + w * 0.08}" y="${y - h * 0.42}" width="${w * 0.24}" height="${h * 0.42}" rx="${w * 0.1}" fill="${skin}"/>
<rect x="${x - w / 2}" y="${y - h * 0.52}" width="${w}" height="${h * 0.16}" rx="${w * 0.12}" fill="${shorts}"/><rect x="${x - w / 2}" y="${y - h * 0.82}" width="${w}" height="${h * 0.34}" rx="${w * 0.2}" fill="${shirt}" stroke="#0003" stroke-width="2"/>
<circle cx="${x}" cy="${y - h * 0.82 - head}" r="${head}" fill="${skin}"/><rect x="${x - w * 0.12}" y="${y - h * 0.74}" width="${w * 0.24}" height="${h * 0.12}" fill="#fff8"/></g>`;
}
const svg = (body, width = W, height = H) => `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${body}</svg>`;
function thumb(name) {
  const [, hallIndex, team, , pale] = /^s(\d)-t(-?\d)-(\d+)(-p)?$/.exec(name) || [];
  const h = halls[Number(hallIndex) || 0];
  const shirt = team === '-1' ? '#8e44ad' : h.teams[Number(team) || 0];
  return svg(`<rect width="120" height="200" fill="${h.floor[0]}"/>${figure(60, 190, 175, shirt)}${pale ? '<rect x="0" y="0" width="120" height="70" fill="' + h.wall + '"/>' : ''}`, 120, 200);
}
function preview(index) {
  const hallIndex = index % 3, h = halls[hallIndex];
  const spots = [[420, 640, 230, 0], [620, 520, 170, 1], [900, 600, 210, 0], [1120, 720, 270, 1], [760, 780, 300, 0], [1280, 520, 165, 1], [520, 820, 320, 1], [1000, 480, 150, 2]];
  const people = spots.slice(0, 5 + (index % 3)).map(([x, y, size, team]) => {
    const shirt = team === 2 ? '#111' : h.teams[team];
    const colour = team === 2 ? '#f5c518' : team === 0 ? '#4aa3ff' : '#ff6b6b';
    const w = size * 0.4;
    return figure(x, y, size, shirt) + `<rect x="${x - w / 2 - 6}" y="${y - size - 6}" width="${w + 12}" height="${size + 12}" fill="none" stroke="${colour}" stroke-width="5"/>`;
  }).join('');
  return svg(hall(hallIndex) + people + '<circle cx="980" cy="430" r="13" fill="#e2692a"/><rect x="960" y="410" width="40" height="40" fill="none" stroke="#3ddc84" stroke-width="5"/>');
}

// --- Studio state ---------------------------------------------------------------------------
function studioFor(language) {
const sessions = halls.map((h, index) => ({
  id: `${names[language][index]}|cam${index + 1}`, images: Array.from({ length: [212, 168, 134][index] }, (_, i) => `cam${index + 1}-${String(i).padStart(4, '0')}.jpg`),
  split: ['train', 'valid', 'test'][index], width: W, height: H, plateFile: `plate-${index}.jpg`,
  polygon: court, polygonSource: index === 2 ? 'estimated' : 'clicked', teamCounts: { 0: 5, 1: 5 }, link: `link-${index}`,
}));
const instances = [];
halls.forEach((_, s) => {
  for (let i = 0; i < 16; i += 1) {
    const team = i < 8 ? 0 : i < 15 ? 1 : -1;
    const pale = i === 3 || i === 11;
    instances.push({ id: `s${s}-t${team}-${i}${pale ? '-p' : ''}`, session: sessions[s].id, role: 'player', team, uncertain: i === 6 || i === 13, complete: !pale, image: sessions[s].images[i] });
  }
});
return {
  source: '/Users/coach/Reco/Export/basketball-coco',
  config: {
    variations: 1800, cropShare: 0.6, zoom: [0.2, 0.7], outputSize: 384, seed: 1, minVisible: 0.3,
    roles: { ball: 'ball', hoop: 'static', player: 'player', referee: 'referee' },
    limits: { players: 5, referees: 3, balls: 1 }, jerseyVariation: true, yolo: false, anonymize: false,
  },
  analysis: { classes: ['ball', 'hoop', 'player', 'referee'], sessions, instances, splitWarning: null, log: [], dropped: 0 },
  report: { images: 1800, violations: {}, foundThenRegenerated: {}, rejectedWhileGenerating: { overlap: 14 }, labelsTooSmall: 3, real: { train: 212, valid: 168, test: 134 } },
  previews: Array.from({ length: 6 }, (_, i) => `preview-${i}.jpg`),
  generated: 1800,
  training: { mode: 'mixed', realWeight: 3 },
};
}
const status = {
  connected: true, selectedFolder: '/Users/coach/Reco/Basketball', folderName: 'Basketball', projectRoot: '/Users/coach/Reco/Basketball', sport: 'basketball',
  operation: 'ready', progress: 0, message: '', busy: false, error: null, log: '', mlReady: true, frames: [],
  modelLibrary: { packages: [], activePackageID: null },
};

// --- Shots ----------------------------------------------------------------------------------
const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH, headless: true });
for (const language of ['de', 'en', 'es', 'fr']) {
  const studio = studioFor(language);
  const page = await browser.newPage({ viewport: { width: 1180, height: 900 }, deviceScaleFactor: 1.5, colorScheme: 'light' });
  await page.addInitScript((lang) => {
    localStorage.setItem('reco-language-v1', lang);
    localStorage.setItem('reco-walkthrough-complete-v2', 'yes');
    localStorage.setItem('reco-field-nudge-dismissed-v1', 'yes');
    localStorage.setItem('reco-whats-new-seen-v1', '9.9.9');
    localStorage.setItem('reco-augment-guide-hidden-v1', 'yes');
  }, language);
  await page.route('**/api/**', async (route) => {
    const address = new URL(route.request().url());
    const path = address.pathname;
    if (path === '/api/events') return route.fulfill({ contentType: 'text/event-stream', body: '' });
    if (path === '/api/status') return route.fulfill({ json: status });
    if (path === '/api/augment/state') return route.fulfill({ json: studio });
    if (path === '/api/augment/file') {
      const kind = address.searchParams.get('kind'), name = (address.searchParams.get('name') || '').replace(/\.jpg$/, '');
      const body = kind === 'thumb' ? thumb(name) : kind === 'plate' ? svg(hall(Number(name.split('-')[1]) || 0)) : preview(Number(name.split('-')[1]) || 0);
      return route.fulfill({ contentType: 'image/svg+xml', body });
    }
    return route.fulfill({ json: {} });
  });
  await page.route('https://api.github.com/**', (route) => route.fulfill({ json: {} }));
  await page.goto(url);
  await page.locator('.header-actions button', { hasText: 'Augmentation Studio' }).click();
  await page.locator('.studio-previews img').first().waitFor({ state: 'attached' });
  await page.addStyleTag({ content: '[data-hint]::after,[data-hint]::before{display:none!important} *{caret-color:transparent!important} .studio *{scroll-margin-top:110px}' });
  await page.waitForTimeout(500);

  const pad = (b, p = 10) => ({ x: Math.max(0, b.x - p), y: Math.max(0, b.y - p), width: b.width + 2 * p, height: b.height + 2 * p });
  // targets: [selector, index] pairs; the shot covers all of them (each region fits the viewport).
  const shoot = async (step, ...targets) => {
    const locators = targets.map(([selector, index = 0]) => page.locator(selector).nth(index));
    await locators[0].evaluate((node) => node.scrollIntoView({ block: 'start' }));
    await page.waitForTimeout(250);
    const boxes = await Promise.all(locators.map((locator) => locator.boundingBox()));
    const x = Math.min(...boxes.map((b) => b.x)), y = Math.min(...boxes.map((b) => b.y));
    const clip = pad({ x, y, width: Math.max(...boxes.map((b) => b.x + b.width)) - x, height: Math.max(...boxes.map((b) => b.y + b.height)) - y });
    await page.screenshot({ path: `${out}/${language}-${step}.jpg`, type: 'jpeg', quality: 80, clip });
    console.log(`${language}-${step}`, Math.round(clip.width), 'x', Math.round(clip.height));
  };
  const cards = '.studio > .studio-card';
  // Lazy images below the fold would never load for a clip screenshot.
  await page.evaluate(() => Promise.all([...document.querySelectorAll('.studio img')].map((img) => { img.loading = 'eager'; return img.complete ? null : new Promise((done) => { img.onload = img.onerror = done; }); })));
  await page.waitForTimeout(300);

  await shoot(1, ['.studio .benchmark-heading'], [cards, 0]);
  await shoot(2, [cards, 0]);
  await shoot(3, [cards, 1]);
  await shoot(4, ['.studio-session header'], ['.studio-plate']);
  await shoot(5, ['.studio-teams']);
  await shoot(6, [cards, 3]);
  await shoot(7, [cards, 4]);
  await page.close();
}
await browser.close();
server.close();
