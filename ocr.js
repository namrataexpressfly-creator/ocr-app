// ocr.js - local OCR (no external API). Uses tesseract.js + sharp.
// Works for different courier image formats: the photo is grey and the info
// text is printed in a colour (yellow, red, ...), so we keep only coloured pixels.
const sharp = require('sharp');
const Tesseract = require('tesseract.js');
const path = require('path');

let worker = null;

async function getWorker() {
  if (!worker) {
    worker = await Tesseract.createWorker('eng', 1, {
      // English data bundled via npm (@tesseract.js-data/eng) - fully offline
      langPath: path.join(__dirname, 'node_modules', '@tesseract.js-data', 'eng', '4.0.0_best_int'),
      gzip: true,
      cacheMethod: 'none',
    });
    await worker.setParameters({
      tessedit_pageseg_mode: '6', // single uniform block of text
      preserve_interword_spaces: '1',
      user_defined_dpi: '300',
    });
  }
  return worker;
}

async function closeWorker() {
  if (worker) { await worker.terminate(); worker = null; }
}

async function downloadImage(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error('Download failed: HTTP ' + res.status);
  return Buffer.from(await res.arrayBuffer());
}

// Turn coloured overlay text into black-on-white, crop to it and scale it to a good size.
async function preprocess(buffer) {
  const { data, info } = await sharp(buffer).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const W = info.width, H = info.height;

  // 1. mask of "coloured" pixels (grey photo pixels have R≈G≈B)
  const mask = new Uint8Array(W * H);
  for (let i = 0, p = 0; p < mask.length; i += 3, p++) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    if (max > 110 && max - min > 70) mask[p] = 1;
  }

  // 2. find the text block: rows / columns that contain enough coloured pixels
  const rowCount = new Uint32Array(H), colCount = new Uint32Array(W);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (mask[y * W + x]) { rowCount[y]++; colCount[x]++; }
  const minHits = 3;
  let top = rowCount.findIndex(c => c >= minHits);
  if (top < 0) throw new Error('No coloured text found in image');
  let bottom = H - 1; while (bottom > top && rowCount[bottom] < minHits) bottom--;
  let left = colCount.findIndex(c => c >= minHits);
  let right = W - 1; while (right > left && colCount[right] < minHits) right--;

  // 3. estimate text line height -> scale so letters are ~40px tall
  const lineHeights = [];
  let run = 0;
  for (let y = top; y <= bottom + 1; y++) {
    if (y <= bottom && rowCount[y] >= minHits) run++;
    else { if (run > 4) lineHeights.push(run); run = 0; }
  }
  lineHeights.sort((a, b) => a - b);
  const lineH = lineHeights[Math.floor(lineHeights.length / 2)] || 30;
  const scale = Math.max(0.4, Math.min(4, 40 / lineH));

  const pad = Math.round(lineH);
  top = Math.max(0, top - pad); left = Math.max(0, left - pad);
  bottom = Math.min(H - 1, bottom + pad); right = Math.min(W - 1, right + pad);
  const cw = right - left + 1, ch = bottom - top + 1;

  const out = Buffer.alloc(cw * ch);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
    out[y * cw + x] = mask[(y + top) * W + (x + left)] ? 0 : 255;
  }

  return sharp(out, { raw: { width: cw, height: ch, channels: 1 } })
    .resize({ width: Math.round(cw * scale), kernel: 'lanczos3' })
    .blur(scale > 1.5 ? 0.6 : 0.3) // smooth the jagged edges after scaling
    .threshold(160)
    .png()
    .toBuffer();
}

// 'Hyderabad Medchal GW (Telangana)' -> 'Hyderabad_Medchal_GW (Telangana)'
function cleanLocation(s) {
  s = s.replace(/\s+/g, ' ').replace(/\s*_\s*/g, '_').trim();
  const m = s.match(/^(.*?)\s*(\(.*\))?$/);
  if (m && /GW/i.test(m[1])) s = m[1].trim().replace(/ /g, '_') + (m[2] ? ' ' + m[2] : '');
  return s;
}

// Label variations seen so far:
//   Location: / LOCATION CODE :        AWB No: / AWB_NO : / AWB_No:
//   Length: / LENGTH(cm) :             R.Vol.: / Volume:
//   Weight: / WEIGHT(Kg) :             Time: / DATETIME :
//   ID : / Sorter_ID:
function parseText(raw) {
  const lines = raw.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  // Text after the label. OCR sometimes reads ':' as '.', '*' or drops it, so strip any of those.
  const value = (label) => {
    for (const l of lines) {
      const m = l.match(label);
      if (m) return l.slice(m.index + m[0].length).replace(/^[\s:;.*,'"=~_-]+/, '').trim();
    }
    return '';
  };
  const num = (s) => { const m = s.replace(/,/g, '.').match(/(\d+(?:\.\d+)?)/); return m ? m[1] : ''; };

  const awb = value(/^\W*AWB[\W_]*N[O0]\.?/i);
  const timeText = value(/^\W*(date\s*)?time/i);
  const time = timeText.match(/(\d{1,2}[-\/.]\d{1,2}[-\/.]\d{2,4})[\s.]+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)/i);

  return {
    location: cleanLocation(value(/^\W*location(\s*code)?/i)),
    awb: (awb.replace(/[OoQD]/g, '0').replace(/[Il|]/g, '1').replace(/\s/g, '').match(/(\d{8,})/) || [, ''])[1],
    length: num(value(/^\W*length\s*(\(\s*cm\s*\))?/i)),
    width: num(value(/^\W*width\s*(\(\s*cm\s*\))?/i)),
    height: num(value(/^\W*height\s*(\(\s*cm\s*\))?/i)),
    rvol: num(value(/^\W*(r\W*)?vol(ume)?\.?/i)),
    weight: num(value(/^\W*weight\s*(\(\s*kg\s*\))?/i)),
    time: time ? time[1] + ' ' + time[2] : timeText,
    machineId: value(/^\W*(sorter[\W_]*)?id\b/i),
  };
}

async function ocrImageUrl(url) {
  const original = await downloadImage(url);
  const clean = await preprocess(original);
  const w = await getWorker();
  const { data } = await w.recognize(clean);
  return { raw: data.text, fields: parseText(data.text) };
}

module.exports = { ocrImageUrl, closeWorker, parseText, preprocess };
