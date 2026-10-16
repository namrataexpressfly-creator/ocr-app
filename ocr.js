// ocr.js - local OCR (no external API). Uses tesseract.js + sharp.
//
// The image overlay can differ between sources (yellow / white / dark text, top-left
// corner or a full label, small or huge photos). Instead of one fixed preprocessing,
// we try several "variants" (region x colour-mask x page-segmentation mode), parse every
// result, and MERGE the best value for each field. We stop early once all fields are found.
const sharp = require('sharp');
const Tesseract = require('tesseract.js');
const path = require('path');

// Pool of tesseract workers so several images are processed in parallel.
const os = require('os');
const POOL_SIZE = Math.max(1, Number(process.env.OCR_CONCURRENCY) || Math.min(os.cpus().length, 6));
const pool = [];       // idle workers
let created = 0;
const waiters = [];    // callbacks waiting for a free worker

async function makeWorker() {
  const w = await Tesseract.createWorker('eng', 1, {
    langPath: path.join(__dirname, 'node_modules', '@tesseract.js-data', 'eng', '4.0.0_best_int'),
    gzip: true,
    cacheMethod: 'none',
  });
  await w.setParameters({ preserve_interword_spaces: '1', user_defined_dpi: '300' });
  w._psm = null;
  return w;
}

async function acquireWorker() {
  if (pool.length) return pool.pop();
  if (created < POOL_SIZE) { created++; return makeWorker(); }
  return new Promise((resolve) => waiters.push(resolve));
}

function releaseWorker(w) {
  const next = waiters.shift();
  if (next) next(w); else pool.push(w);
}

async function closeWorker() {
  const all = pool.splice(0);
  await Promise.all(all.map((w) => w.terminate()));
  created = 0;
}

async function downloadImage(url, tries = 3) {
  let lastErr;
  for (let i = 1; i <= tries; i++) {
    try {
      const res = await fetch(url, {
        headers: { 'User-Agent': 'Mozilla/5.0 (compatible; ocr-app)', Accept: 'image/*,*/*' },
        signal: AbortSignal.timeout(30000),
      });
      if (!res.ok) throw new Error('Download failed: HTTP ' + res.status);
      return Buffer.from(await res.arrayBuffer());
    } catch (e) {
      lastErr = e;
      await new Promise((r) => setTimeout(r, 500 * i));
    }
  }
  throw lastErr;
}

/* ---------------------------- preprocessing ---------------------------- */

const REGIONS = {
  tl: { x: 0, y: 0, w: 0.42, h: 0.45 },   // top-left overlay (layout A: yellow, Location/AWB/Length...)
  top: { x: 0, y: 0, w: 1, h: 0.6 },      // whole top part
  tl2: { x: 0, y: 0, w: 0.6, h: 0.22 },   // layout B: AWB / date / machine id (green, top-left)
  bl: { x: 0, y: 0.78, w: 0.65, h: 0.22 },// layout B: (L x B x H) and Weight (green, bottom-left)
  full: { x: 0, y: 0, w: 1, h: 1 },       // whole image
};

// Colour predicates -> "is this pixel text?"
const MASKS = {
  yellow: (r, g, b) => r > 120 && g > 100 && r - b > 60 && g - b > 40,
  white: (r, g, b) => r > 190 && g > 190 && b > 190,
  bright: (r, g, b) => 0.299 * r + 0.587 * g + 0.114 * b > 170,
  green: (r, g, b) => g > 110 && g - r > 45 && g - b > 45,
  dark: (r, g, b) => 0.299 * r + 0.587 * g + 0.114 * b < 90,
};

// Final scaling: tesseract works best with text ~30px high -> upscale small images, cap huge ones.
function targetWidth(w) {
  if (w < 1000) return Math.min(w * 2, 1800);
  return Math.min(w, 1800);
}

async function regionBuffer(buffer, region) {
  const meta = await sharp(buffer).rotate().metadata(); // honour EXIF rotation
  const reg = REGIONS[region];
  const width = Math.round(meta.width * reg.w);
  const height = Math.round(meta.height * reg.h);
  const left = Math.round(meta.width * reg.x);
  const top = Math.min(Math.round(meta.height * reg.y), meta.height - height);
  return { width, height, make: () => sharp(buffer).rotate().extract({ left, top, width, height }) };
}

async function preprocess(buffer, v) {
  const { width, height, make } = await regionBuffer(buffer, v.region);
  const tw = targetWidth(width);

  if (v.mask) {
    const { data, info } = await make().removeAlpha().raw().toBuffer({ resolveWithObject: true });
    const pred = MASKS[v.mask];
    const out = Buffer.alloc(info.width * info.height);
    for (let i = 0, p = 0; p < out.length; i += 3, p++) out[p] = pred(data[i], data[i + 1], data[i + 2]) ? 0 : 255;
    return sharp(out, { raw: { width: info.width, height: info.height, channels: 1 } })
      .resize({ width: tw })
      .median(3)
      .png()
      .toBuffer();
  }

  // plain greyscale (optionally inverted) with contrast stretch
  let p = make().greyscale().normalise().resize({ width: tw }).sharpen();
  if (v.invert) p = p.negate();
  return p.png().toBuffer();
}

// Ordered by likelihood. psm 6 = one block of text, 11 = sparse text (finds scattered lines).
const VARIANTS = [
  { name: 'yellow-tl', region: 'tl', mask: 'yellow', psm: '6' },
  { name: 'green-tl2', region: 'tl2', mask: 'green', psm: '6' },
  { name: 'green-bl', region: 'bl', mask: 'green', psm: '6' },
  { name: 'yellow-top', region: 'top', mask: 'yellow', psm: '6' },
  { name: 'white-tl', region: 'tl', mask: 'white', psm: '6' },
  { name: 'bright-top', region: 'top', mask: 'bright', psm: '6' },
  { name: 'gray-top', region: 'top', invert: false, psm: '6' },
  { name: 'gray-full-sparse', region: 'full', invert: false, psm: '11' },
  { name: 'inv-full-sparse', region: 'full', invert: true, psm: '11' },
  { name: 'dark-full', region: 'full', mask: 'dark', psm: '11' },
  { name: 'bright-full', region: 'full', mask: 'bright', psm: '11' },
];

/* -------------------------------- parsing ------------------------------- */

// fix typical OCR digit confusions inside a token that is supposed to be a number
function toNumberString(tok) {
  if (!tok) return '';
  const t = tok.replace(/[Oo]/g, '0').replace(/[Il|]/g, '1').replace(/S/g, '5').replace(/B/g, '8').replace(/,/g, '.');
  const m = t.match(/\d+(?:\.\d+)?/);
  return m ? m[0] : '';
}

const NUM = '([0-9OoIl|SB]+(?:[.,][0-9OoIl|SB]+)?)';
// label, optional "(cm)" / "(kg)" / "(cubic)" unit text, separator, number
function labelNumber(text, labelRe) {
  const re = new RegExp(labelRe + '\\s*(?:\\([^)\\n]{0,12}\\))?\\s*[:;=\\-]?\\s*' + NUM, 'i');
  const m = text.match(re);
  return m ? toNumberString(m[1]) : '';
}

// 'Hyderabad Medchal GW (Telangana)' -> 'Hyderabad_Medchal_GW (Telangana)'
function cleanLocation(s) {
  s = s.replace(/\s+/g, ' ').replace(/\s*_\s*/g, '_').trim();
  const m = s.match(/^(.*?)\s*(\(.*\))?$/);
  if (m && /GW/i.test(m[1])) s = m[1].trim().replace(/ /g, '_') + (m[2] ? ' ' + m[2] : '');
  return s;
}

function parseText(raw, hint = {}) {
  const text = (raw || '').replace(/\r/g, '');
  const lines = text.split('\n').map((l) => l.trim()).filter(Boolean);
  const afterColon = (s) => s.replace(/^[^:;]*[:;]\s*/, '').trim();

  // Location: "Location: xxx" (value may wrap onto the next line if it is short)
  let location = '';
  const li = lines.findIndex((l) => /locat|l0cat|iocat/i.test(l));
  if (li >= 0) {
    location = lines[li].replace(/^.*?(?:locat|l0cat|iocat)\w*\s*[:;\-]?\s*/i, '').trim();
    if (location.length < 4 && lines[li + 1] && !/awb|length|width|height|weight|time/i.test(lines[li + 1])) location = lines[li + 1];
  }
  if (!location) {
    const guess = lines.find((l) => /\bGW\b|_GW|\bDC[_ -]|\bHUB\b/i.test(l) && !/awb/i.test(l));
    if (guess) location = guess.replace(/^[^:;]*[:;]\s*/, '');
  }

  // AWB: SF-style (SF + 10-13 digits + 0-4 letters) anywhere, else the AWB line, else a long digit run.
  let awb = '';
  const sf = text.replace(/[ \t]/g, '').match(/\b(SF\d{10,14}[A-Z]{0,4})/i);
  if (sf) awb = sf[1].toUpperCase();
  if (!awb) {
    const awbLine = lines.find((l) => /awb|a\.w\.b|waybill/i.test(l)) || '';
    const cand = awbLine.replace(/^.*?(awb|waybill)\w*\s*(no\.?|number|#)?\s*[:;\-]?\s*/i, '');
    const m = cand.replace(/\s/g, '').replace(/[OoQ]/g, '0').replace(/[Il|]/g, '1').match(/([A-Z]{0,3}\d{8,}[A-Z]{0,4})/i);
    if (m) awb = m[1].toUpperCase();
  }
  if (!awb) {
    const m = text.replace(/[ \t]/g, '').match(/(?<!\d)(\d{10,14})(?!\d)/);
    if (m) awb = m[1];
  }

  const timeRe = /(\d{1,4}\s*[-\/.]\s*\d{1,2}\s*[-\/.]\s*\d{2,4}[\sT,]+\d{1,2}\s*:\s*\d{2}(?:\s*:\s*\d{2})?(?:\s*[AaPp][Mm])?)/;
  const tm = text.match(timeRe);
  let time = tm ? tm[1].replace(/\s*([-\/.:])\s*/g, '$1').replace(/[,T]+/, ' ') : '';
  if (!time) { const tl = lines.find((l) => /time|date/i.test(l)); if (tl) time = afterColon(tl); }

  // Layout B: "(L X B X H)  28.5cm X 27.2cm X 4.0cm"  (B = breadth = width)
  const dn = '([0-9OoIl|]+(?:[.,][0-9OoIl|]+)?)';
  const dm = text.match(new RegExp(dn + '\\s*c?m?\\s*[xX×*]\\s*' + dn + '\\s*c?m?\\s*[xX×*]\\s*' + dn + '\\s*c?m', 'i'));
  const machine = text.match(/machine\s*id\s*[:;\-]?\s*([A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*)/i);
  if (!location && machine) location = machine[1].toUpperCase().replace(/([A-Z]+)([0O]+\d*|\d+[0O]\d*)$/, (m0, p, n) => p + n.replace(/O/g, '0'));

  return {
    location: cleanLocation(location),
    awb,
    length: labelNumber(text, '(?:l[e3c]ngth|l[e3c]n[g9]th|len\\.?)') || (dm ? toNumberString(dm[1]) : ''),
    width: labelNumber(text, '(?:w[i1l]dth|w[i1l]dt[hn]|br[e3]adth)') || (dm ? toNumberString(dm[2]) : ''),
    height: labelNumber(text, '(?:h[e3c][i1l][g9]ht|h[e3c][i1l]ght)') || (dm ? toNumberString(dm[3]) : ''),
    rvol: labelNumber(text, '(?:r\\.?\\s*)?vol(?:ume|umetric)?\\.?'),
    weight: labelNumber(text, '(?<![a-z])w[e3c][i1l][g9]ht(?!\\w)'),
    time,
  };
}

const FIELDS = ['location', 'awb', 'length', 'width', 'height', 'rvol', 'weight', 'time'];
const CORE = ['awb', 'length', 'width', 'height', 'weight']; // must-have for "good enough"

function score(f) {
  return FIELDS.reduce((n, k) => n + (f[k] ? (CORE.includes(k) ? 2 : 1) : 0), 0);
}

// AWB from the file name, e.g. .../SF3723044239EXY_1785440461261.jpg  or  .../SF3675045089EXU_5430ad91.jpg
function awbFromUrl(url) {
  const m = String(url).match(/\/(SF\d{10,14}[A-Z]{0,4})(?:[_.\-])/i);
  return m ? m[1].toUpperCase() : '';
}

/* --------------------------------- main --------------------------------- */

// Adaptive ordering: variants that recently produced data go first, so a batch of
// same-layout images usually needs only 1-2 OCR passes per image instead of all of them.
const wins = Object.fromEntries(VARIANTS.map((v) => [v.name, 0]));
const orderedVariants = () => [...VARIANTS].sort((a, b) => wins[b.name] - wins[a.name]);

async function ocrBuffer(rawOriginal, url = '') {
  // Decode/shrink once (huge photos are slow to re-decode for every variant).
  const original = await sharp(rawOriginal).rotate().resize({ width: 2400, withoutEnlargement: true }).png({ compressionLevel: 1 }).toBuffer();
  const w = await acquireWorker();
  try { return await ocrWithWorker(w, original, url); } finally { releaseWorker(w); }
}

async function ocrWithWorker(w, original, url) {
  const results = [];
  const merged = {};
  const rawParts = [];

  for (const v of orderedVariants()) {
    let text = '';
    try {
      const img = await preprocess(original, v);
      if (w._psm !== v.psm) { await w.setParameters({ tessedit_pageseg_mode: v.psm }); w._psm = v.psm; }
      ({ data: { text } } = await w.recognize(img));
    } catch (e) {
      rawParts.push(`[${v.name}] error: ${e.message}`);
      continue;
    }
    const fields = parseText(text);
    results.push({ v, fields, s: score(fields) });
    rawParts.push(`[${v.name}] score=${score(fields)}\n${text.trim()}`);

    // merge: first variant to supply a field wins, but better-scoring variants are preferred
    let gave = false;
    for (const k of FIELDS) if (!merged[k] && fields[k]) { merged[k] = fields[k]; gave = true; }
    if (gave) wins[v.name]++;
    if (CORE.every((k) => merged[k]) && /\)\s*$|^[A-Z]{2,5}[-_][A-Z]+\d+$/.test(fields.location || merged.location || '') && merged.time) break; // everything found
  }

  // Prefer values from the single best variant when it has the field (more self-consistent).
  results.sort((a, b) => b.s - a.s);
  const best = {};
  for (const r of results) for (const k of FIELDS) if (!best[k] && r.fields[k]) best[k] = r.fields[k];
  const fields = Object.fromEntries(FIELDS.map((k) => [k, best[k] || merged[k] || '']));
  // Location: prefer a complete value (ends with ")") and then the longest one.
  const locs = results.map((r) => r.fields.location).filter(Boolean)
    .sort((a, b) => (/\)\s*$/.test(b) - /\)\s*$/.test(a)) || b.length - a.length);
  if (locs[0]) fields.location = locs[0];

  // The file name often carries the real AWB. Use it when OCR missed it or only read part of it.
  const urlAwb = awbFromUrl(url);
  if (urlAwb) {
    const digits = (fields.awb || '').replace(/\D/g, '');
    if (!fields.awb || urlAwb.includes(digits) || digits.includes(urlAwb.replace(/\D/g, ''))) fields.awb = urlAwb;
  }

  // Machine ID (layout B) is also in the URL path: .../DC_GAU-FSM01/2026/...  Use it to fix O/0 slips.
  const um = String(url).match(/\/DC_([A-Za-z]{2,5}[-_][A-Za-z]+\d+)\//);
  if (um && fields.location && /^[A-Z]{2,5}[-_][A-Z]+[0-9O]+$/.test(fields.location)) {
    const norm = (x) => x.toUpperCase().replace(/[O0]/g, '');
    if (norm(um[1]) === norm(fields.location)) fields.location = um[1].toUpperCase();
  }

  return { raw: rawParts.join('\n\n'), fields };
}

async function ocrImageUrl(url) {
  const original = await downloadImage(url);
  return ocrBuffer(original, url);
}

module.exports = { ocrImageUrl, ocrBuffer, closeWorker, parseText, awbFromUrl };
