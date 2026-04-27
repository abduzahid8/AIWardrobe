const fs = require('fs');
const path = require('path');

const SUPABASE_URL = 'https://fyqpifmrsftsfqibhwhy.supabase.co';
const ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5cXBpZm1yc2Z0c2ZxaWJod2h5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzMDYyOTgsImV4cCI6MjA3OTg4MjI5OH0.dydnFn3lqub7qMo9uFfn5yUyY4Wr_eQPnsbvHWHwMTk';
const FUNCTION_URL = `${SUPABASE_URL}/functions/v1/mannequin-tryon`;

const ITEMS = [
  {
    label: 'top',
    type: 'upper_body',
    imageUrl: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg'
  },
  {
    label: 'layer',
    type: 'upper_body',
    imageUrl: 'https://assets.burberry.com/is/image/Burberryltd/3DFB8EAD-C042-4E2C-B62D-9F3C1B6011DC'
  },
  {
    label: 'pants',
    type: 'lower_body',
    imageUrl: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/450251/item/usgoods_35_450251_3x4.jpg'
  },
  {
    label: 'shoes',
    type: 'shoes',
    imageUrl: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/484330/item/goods_32_484330_3x4.jpg'
  }
];

async function callTryOn(mannequinB64, item, step, total) {
  console.log(`[Step ${step}/${total}] Calling Edge Function for ${item.label}...`);
  
  const payload = {
    mannequin_image: mannequinB64,
    garment: {
      image: item.imageUrl,
      type: item.type,
      label: item.label
    },
    step: step,
    total: total
  };

  const response = await fetch(FUNCTION_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${ANON_KEY}`
    },
    body: JSON.stringify(payload)
  });

  const rawText = await response.text();
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${rawText}`);
  }

  let data;
  try {
    data = JSON.parse(rawText);
  } catch (e) {
    throw new Error(`Invalid JSON: ${rawText.slice(0, 100)}`);
  }
  
  if (!data.success) {
    throw new Error(`API Error: ${data.error}`);
  }

  return data.resultUrl; 
}

async function runTest() {
  try {
    // 1. Read base mannequin
    const mannequinPath = path.join(__dirname, 'assets/images/mannequin_front.png');
    const mannequinBuffer = fs.readFileSync(mannequinPath);
    let currentMannequin = `data:image/png;base64,${mannequinBuffer.toString('base64')}`;

    console.log('Base mannequin loaded. Starting sequential try-on...');

    // 2. Loop through all 4 slots
    for (let i = 0; i < ITEMS.length; i++) {
        const item = ITEMS[i];
        currentMannequin = await callTryOn(currentMannequin, item, i + 1, ITEMS.length);

        // Optionally save intermediate steps
        const base64Data = currentMannequin.replace(/^data:image\/\w+;base64,/, '');
        fs.writeFileSync(`tryon_step_${i+1}_${item.label}.png`, Buffer.from(base64Data, 'base64'));
        console.log(`Step ${i+1} saved to tryon_step_${i+1}_${item.label}.png`);
    }

    console.log('Successfully completed 4-item try on. Final image is tryon_step_4_shoes.png');
  } catch (err) {
    console.error('Test failed:', err);
  }
}

runTest();
