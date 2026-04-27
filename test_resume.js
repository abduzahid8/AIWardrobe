// test_resume.js — picks up from step 3 (pants) using the step 2 (layer) result
const fs = require('fs');
const path = require('path');

const SUPABASE_URL = 'https://fyqpifmrsftsfqibhwhy.supabase.co';
const ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5cXBpZm1yc2Z0c2ZxaWJod2h5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzMDYyOTgsImV4cCI6MjA3OTg4MjI5OH0.dydnFn3lqub7qMo9uFfn5yUyY4Wr_eQPnsbvHWHwMTk';
const FN = `${SUPABASE_URL}/functions/v1/mannequin-tryon`;
const HEADERS = { 'Content-Type': 'application/json', 'Authorization': `Bearer ${ANON_KEY}` };

// Start from the step 2 result (mannequin wearing top + layer)
const STEP2_RESULT = 'https://replicate.delivery/yhqm/GFnsO9UPghYIIdWfJ6DpN5V3kk6FQwulPcfRgD5oyjOxJBfs/out-0.png';

const REMAINING_GARMENTS = [
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
    if (data.status === 'failed') throw new Error(`Failed for ${label}`);
  }
  throw new Error(`Timeout for ${label}`);
}

async function run() {
  // Download and convert step2 result to base64
  console.log('Loading step 2 result (mannequin with top + layer)...');
  const step2Res = await fetch(STEP2_RESULT);
  const step2Buf = Buffer.from(await step2Res.arrayBuffer());
  fs.writeFileSync('tryon_step2_layer.jpg', step2Buf);
  let currentMannequin = 'data:image/jpeg;base64,' + step2Buf.toString('base64');
  console.log('Step 2 image saved → tryon_step2_layer.jpg\n');

  for (let i = 0; i < REMAINING_GARMENTS.length; i++) {
    const g = REMAINING_GARMENTS[i];
    const stepNum = i + 3;
    console.log(`[Step ${stepNum}/4] ${g.label} → Submitting…`);

    const data = await invokeEdge({
      action: 'submit',
      mannequin_image: currentMannequin,
      garment: { image: g.imageUrl, type: g.category, label: g.label },
      step: stepNum,
      total: 4,
    });

    if (!data.success) throw new Error(`Submit failed: ${data.error}`);
    console.log(`  mode=${data.mode}  predictionId=${data.predictionId}`);

    if (data.mode === 'sync') {
      currentMannequin = data.resultUrl;
    } else {
      currentMannequin = await pollUntilDone(data.predictionId, g.label);
    }
    console.log(`  ✓ ${g.label} done: ${currentMannequin.slice(0, 80)}...\n`);

    // Save intermediate
    const imgRes = await fetch(currentMannequin);
    const imgBuf = Buffer.from(await imgRes.arrayBuffer());
    fs.writeFileSync(`tryon_step${stepNum}_${g.label}.jpg`, imgBuf);
    console.log(`  Saved → tryon_step${stepNum}_${g.label}.jpg`);
  }

  console.log('\n=== FINAL RESULT ===');
  console.log(currentMannequin);
}

run().catch(e => { console.error('\n✗ FAILED:', e.message); process.exit(1); });
