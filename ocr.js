// ocr.js - local OCR (no external API). Uses tesseract.js + sharp.
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

// Crop the top-left text overlay and turn yellow text into black-on-white.
async function preprocess(buffer) {
  const img = sharp(buffer);
  const meta = await img.metadata();
  const cropW = Math.round(meta.width * 0.42);
  const cropH = Math.round(meta.height * 0.45);

  const { data, info } = await sharp(buffer)
    .extract({ left: 0, top: 0, width: cropW, height: cropH })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const out = Buffer.alloc(info.width * info.height);
  for (let i = 0, p = 0; p < out.length; i += 3, p++) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    // yellow-ish: strong red & green, much weaker blue
    const isText = r > 120 && g > 100 && r - b > 60 && g - b > 40;
    out[p] = isText ? 0 : 255;
  }

  // scale down very large images a bit (text is huge) and smooth edges
  const targetW = Math.min(info.width, 1600);
  return sharp(out, { raw: { width: info.width, height: info.height, channels: 1 } })
    .resize({ width: targetW })
    .median(3)
    .png()
    .toBuffer();
}

function pick(re, text) {
  const m = text.match(re);
  return m ? m[1].trim() : '';
}

// 'Hyderabad Medchal GW (Telangana)' -> 'Hyderabad_Medchal_GW (Telangana)'
function cleanLocation(s) {
  s = s.replace(/\s+/g, ' ').replace(/\s*_\s*/g, '_').trim();
  const m = s.match(/^(.*?)\s*(\(.*\))?$/);
  if (m && /GW/i.test(m[1])) s = m[1].trim().replace(/ /g, '_') + (m[2] ? ' ' + m[2] : '');
  return s;
}

function parseText(raw) {
  const lines = raw.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const line = (key) => lines.find(l => key.test(l)) || '';
  const num = (s) => { const m = s.replace(/,/g, '.').match(/(\d+(?:\.\d+)?)/); return m ? m[1] : ''; };
  const afterColon = (s) => s.replace(/^[^:;]*[:;]\s*/, '').trim();

  const locLine = line(/locat/i);
  const awbLine = line(/awb/i);
  const timeLine = line(/time/i);

  return {
    location: cleanLocation(afterColon(locLine)),
    awb: (awbLine.replace(/[OoQ]/g, '0').replace(/[Il|]/g, '1').match(/(\d{8,})/) || [, ''])[1],
    length: num(afterColon(line(/length/i))),
    width: num(afterColon(line(/width/i))),
    height: num(afterColon(line(/height/i))),
    rvol: num(afterColon(line(/vol/i))),
    weight: num(afterColon(line(/weight/i))),
    time: pick(/(\d{2}[-\/.]\d{2}[-\/.]\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?)/, timeLine) || afterColon(timeLine),
  };
}

async function ocrImageUrl(url) {
  const original = await downloadImage(url);
  const clean = await preprocess(original);
  const w = await getWorker();
  const { data } = await w.recognize(clean);
  return { raw: data.text, fields: parseText(data.text) };
}

module.exports = { ocrImageUrl, closeWorker, parseText };
