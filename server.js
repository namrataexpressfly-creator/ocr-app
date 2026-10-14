// server.js - Upload Excel/CSV with image URLs, get Excel back with the text from each image.
// Run:  node server.js   then open http://localhost:3000
const express = require('express');
const multer = require('multer');
const ExcelJS = require('exceljs');
const path = require('path');
const { Readable } = require('stream');
const { ocrImageUrl } = require('./ocr');

const PORT = process.env.PORT || 3000;

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 20 * 1024 * 1024 } });
const jobs = {}; // jobId -> { total, done, status, file, error }

app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'index.html')));

// Read every http(s) URL from the uploaded sheet (any column, image-URL column preferred).
async function readUrls(file) {
  const wb = new ExcelJS.Workbook();
  const ext = path.extname(file.originalname).toLowerCase();
  if (ext === '.csv') await wb.csv.read(Readable.from(file.buffer.toString('utf8').replace(/^﻿/, '')));
  else await wb.xlsx.load(file.buffer);

  const ws = wb.worksheets[0];
  const urls = [];
  ws.eachRow((row) => {
    row.eachCell((cell) => {
      let v = cell.value;
      if (v && typeof v === 'object') v = v.hyperlink || v.text || '';
      v = String(v || '').trim();
      if (/^https?:\/\//i.test(v) && /\.(jpe?g|png|webp|bmp|gif|tiff?)(\?|$)/i.test(v)) urls.push(v);
    });
  });
  return urls;
}

async function runJob(id, urls, debug) {
  const job = jobs[id];
  const wb = new ExcelJS.Workbook();
  const ws = wb.addWorksheet('Extracted Data');
  const cols = [
    { header: 'Image URL', key: 'url', width: 60 },
    { header: 'Location', key: 'location', width: 34 },
    { header: 'AWB No', key: 'awb', width: 18 },
    { header: 'Length (cm)', key: 'length', width: 12 },
    { header: 'Width (cm)', key: 'width', width: 12 },
    { header: 'Height (cm)', key: 'height', width: 12 },
    { header: 'R.Vol (cubic)', key: 'rvol', width: 13 },
    { header: 'Weight (kg)', key: 'weight', width: 12 },
    { header: 'Time', key: 'time', width: 24 },
    { header: 'ID / Sorter ID', key: 'machineId', width: 14 },
    { header: 'Status', key: 'status', width: 20 },
  ];
  if (debug) cols.push({ header: 'Raw OCR Text', key: 'raw', width: 60 });
  ws.columns = cols;
  ws.getRow(1).font = { bold: true };
  ws.getColumn('awb').numFmt = '@'; // keep AWB as text so Excel doesn't turn it into 5.3E+13
  ws.getColumn('machineId').numFmt = '@'; // keep '04' as text

  const toNum = (s) => (s === '' || isNaN(Number(s)) ? s : Number(s));

  for (const url of urls) {
    let row = { url, status: 'OK' };
    try {
      const { raw, fields } = await ocrImageUrl(url);
      row = {
        ...row, ...fields,
        length: toNum(fields.length), width: toNum(fields.width), height: toNum(fields.height),
        rvol: toNum(fields.rvol), weight: toNum(fields.weight), raw,
      };
      if (!fields.awb && !fields.weight) row.status = 'Text not found';
    } catch (e) {
      row.status = 'Error: ' + e.message;
    }
    ws.addRow(row);
    job.done++;
  }

  // Keep the Excel in memory only - nothing is saved on disk.
  job.buffer = await wb.xlsx.writeBuffer();
  job.fileName = 'extracted_' + Date.now() + '.xlsx';
  job.status = 'done';
  // Free memory if it's never downloaded.
  setTimeout(() => delete jobs[id], 60 * 60 * 1000);
}

app.post('/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
    const urls = await readUrls(req.file);
    if (!urls.length) return res.status(400).json({ error: 'No image URLs found in the file' });

    const id = Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    jobs[id] = { total: urls.length, done: 0, status: 'running' };
    runJob(id, urls, req.body.debug === '1').catch((e) => {
      jobs[id].status = 'error';
      jobs[id].error = e.message;
    });
    res.json({ id, total: urls.length });
  } catch (e) {
    res.status(400).json({ error: 'Could not read file: ' + e.message });
  }
});

app.get('/status/:id', (req, res) => {
  const job = jobs[req.params.id];
  if (!job) return res.json({ status: 'unknown' });
  const { total, done, status, error } = job;
  res.json({ total, done, status, error });
});

app.get('/download/:id', (req, res) => {
  const job = jobs[req.params.id];
  if (!job || !job.buffer) return res.status(404).send('File not available (already downloaded or expired).');
  res.attachment(job.fileName);
  res.type('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
  res.send(Buffer.from(job.buffer));
  delete jobs[req.params.id]; // removed from memory after download
});

app.listen(PORT, () => console.log(`Image-to-Excel OCR running at http://localhost:${PORT}`));
