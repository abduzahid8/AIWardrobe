// test_app_flow.js
// Mirrors exactly what AITryOnScreen does: submit → poll → next step
const fs = require('fs');
const path = require('path');

const SUPABASE_URL = 'https://fyqpifmrsftsfqibhwhy.supabase.co';
const ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5cXBpZm1yc2Z0c2ZxaWJod2h5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzMDYyOTgsImV4cCI6MjA3OTg4MjI5OH0.dydnFn3lqub7qMo9uFfn5yUyY4Wr_eQPnsbvHWHwMTk';
const FN = `${SUPABASE_URL}/functions/v1/mannequin-tryon`;
const HEADERS = { 'Content-Type': 'application/json', 'Authorization': `Bearer ${ANON_KEY}` };

// Same APPLY_ORDER as the app (top → layer → pants → shoes)
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

async function pollUntilDone(predictionId, label) {
  for (let i = 0; i < 60; i++) {
    await new Promise(r => setTimeout(r, 2500));
    process.stdout.write(`  ↻ polling ${label} (${i+1})…\r`);
    const data = await invokeEdge({ action: 'poll', predictionId });
    if (!data.success) throw new Error(`Poll error: ${data.error}`);
    if (data.status === 'succeeded' && data.resultUrl) return data.resultUrl;
    if (data.status === 'failed') throw new Error(`Prediction failed for ${label}`);
  }
  throw new Error(`Timeout for ${label}`);
}

async function run() {
  const imgBuf = fs.readFileSync(path.join(__dirname, 'assets/images/mannequin_front.png'));
  let currentMannequin = 'data:image/png;base64,' + imgBuf.toString('base64');
  const total = GARMENTS.length;

  console.log('Starting 4-step virtual try-on (same flow as the app)...\n');

  for (let i = 0; i < GARMENTS.length; i++) {
    const g = GARMENTS[i];
    console.log(`[Step ${i+1}/${total}] ${g.label} → Submitting…`);

    const data = await invokeEdge({
      action: 'submit',
      mannequin_image: currentMannequin,
      garment: { image: g.imageUrl, type: g.category, label: g.label },
      step: i + 1,
      total,
    });

    if (!data.success) throw new Error(`Submit failed: ${data.error}`);
    const mode = data.mode ?? 'sync';
    console.log(`  mode = ${mode}  methodUsed = ${data.methodUsed}`);

    if (mode === 'sync') {
      currentMannequin = data.resultUrl;
      console.log(`  ✓ NVIDIA result ready`);
    } else if (mode === 'async') {
      console.log(`  predictionId = ${data.predictionId}`);
      currentMannequin = await pollUntilDone(data.predictionId, g.label);
      console.log(`  ✓ Replicate result ready`);
    }
    console.log(`  resultUrl = ${currentMannequin.slice(0, 80)}...\n`);
  }

  console.log('\n=== FINAL RESULT URL ===');
  console.log(currentMannequin);

  // Save locally if it's a URL
  if (currentMannequin.startsWith('http')) {
    const res = await fetch(currentMannequin);
    const buf = Buffer.from(await res.arrayBuffer());
    fs.writeFileSync('tryon_final_result.jpg', buf);
    console.log('\n✓ Saved final image → tryon_final_result.jpg');
  }
}

run().catch(e => { console.error('\n✗ Test FAILED:', e.message); process.exit(1); });
