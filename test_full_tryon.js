// test_full_tryon.js — Full 4-step test with full URL logging and image saving
const fs = require('fs');
const path = require('path');

const SUPABASE_URL = 'https://fyqpifmrsftsfqibhwhy.supabase.co';
const ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5cXBpZm1yc2Z0c2ZxaWJod2h5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzMDYyOTgsImV4cCI6MjA3OTg4MjI5OH0.dydnFn3lqub7qMo9uFfn5yUyY4Wr_eQPnsbvHWHwMTk';
const FN = `${SUPABASE_URL}/functions/v1/mannequin-tryon`;
const HEADERS = { 'Content-Type': 'application/json', 'Authorization': `Bearer ${ANON_KEY}` };

// Same APPLY_ORDER as the app: top → layer → pants → shoes
const GARMENTS = [
  { label: 'top',   category: 'upper_body', imageUrl: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg' },
  { label: 'layer', category: 'upper_body', imageUrl: 'https://assets.burberry.com/is/image/Burberryltd/3DFB8EAD-C042-4E2C-B62D-9F3C1B6011DC' },
  { label: 'pants', category: 'lower_body', imageUrl: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/450251/item/usgoods_35_450251_3x4.jpg' },
  { label: 'shoes', category: 'shoes',      imageUrl: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/484330/item/goods_32_484330_3x4.jpg' },
];

async function invokeEdge(body) {
  const res = await fetch(FN, { method: 'POST', headers: HEADERS, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

async function saveImageFromUrl(url, filename) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not download ${url}`);
  fs.writeFileSync(filename, Buffer.from(await res.arrayBuffer()));
  console.log(`  → Saved ${filename}`);
}

async function pollUntilDone(predictionId, label) {
  for (let i = 0; i < 60; i++) {
    await new Promise(r => setTimeout(r, 2500));
    process.stdout.write(`  ↻ polling ${label} (${i+1})…\r`);
    const data = await invokeEdge({ action: 'poll', predictionId });
    if (!data.success) throw new Error(`Poll error: ${data.error}`);
    if (data.status === 'succeeded' && data.resultUrl) return data.resultUrl;
    if (data.status === 'failed') throw new Error(`Replicate prediction failed for ${label}`);
  }
  throw new Error(`Timeout for ${label}`);
}

async function run() {
  // Load base mannequin as base64
  const imgBuf = fs.readFileSync(path.join(__dirname, 'assets/images/mannequin_front.png'));
  let currentMannequin = 'data:image/png;base64,' + imgBuf.toString('base64');

  console.log('=== AIWardrobe Virtual Try-On Pipeline Test ===');
  console.log('Top → Layer → Pants → Shoes\n');

  for (let i = 0; i < GARMENTS.length; i++) {
    const g = GARMENTS[i];
    const total = GARMENTS.length;
    console.log(`[Step ${i+1}/${total}] Applying ${g.label.toUpperCase()}…`);

    const submitData = await invokeEdge({
      action: 'submit',
      mannequin_image: currentMannequin,
      garment: { image: g.imageUrl, type: g.category, label: g.label },
      step: i + 1,
      total,
    });

    if (!submitData.success) throw new Error(`Submit failed: ${submitData.error}`);
    const mode = submitData.mode ?? 'sync';
    console.log(`  mode=${mode}  method=${submitData.methodUsed}  predictionId=${submitData.predictionId || 'N/A'}`);

    let resultUrl;
    if (mode === 'sync') {
      resultUrl = submitData.resultUrl;
    } else {
      resultUrl = await pollUntilDone(submitData.predictionId, g.label);
    }

    // Log the FULL result URL
    console.log(`  ✓ DONE: ${resultUrl}`);

    // Save to disk
    if (resultUrl.startsWith('http')) {
      await saveImageFromUrl(resultUrl, `tryon_step${i+1}_${g.label}.jpg`);
      // Pass URL directly to next step (Replicate needs a URL, not base64)
      currentMannequin = resultUrl;
    } else {
      // base64 result — save and keep as base64
      const ext = resultUrl.startsWith('data:image/jpeg') ? 'jpg' : 'png';
      const b64 = resultUrl.split(',')[1];
      fs.writeFileSync(`tryon_step${i+1}_${g.label}.${ext}`, Buffer.from(b64, 'base64'));
      console.log(`  → Saved tryon_step${i+1}_${g.label}.${ext}`);
      currentMannequin = resultUrl;
    }
    console.log();
  }

  console.log('=== ALL 4 STEPS COMPLETE ===');
  console.log(`Final result: ${currentMannequin}`);
}

run().catch(e => { console.error('\n✗ FAILED:', e.message); process.exit(1); });
