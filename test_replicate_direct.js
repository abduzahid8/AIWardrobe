const fs = require('fs');
const path = require('path');
const fetch = require('node-fetch');

const REPLICATE_TOKEN = process.env.REPLICATE_TOKEN || 'YOUR_REPLICATE_TOKEN_HERE';

const ITEMS = [
  { label: 'top', type: 'upper_body', imageUrl: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg' },
  { label: 'layer', type: 'upper_body', imageUrl: 'https://assets.burberry.com/is/image/Burberryltd/3DFB8EAD-C042-4E2C-B62D-9F3C1B6011DC' },
  { label: 'pants', type: 'lower_body', imageUrl: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/450251/item/usgoods_35_450251_3x4.jpg' },
  { label: 'shoes', type: 'shoes', imageUrl: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/484330/item/goods_32_484330_3x4.jpg' }
];

async function callReplicate(personImage, garmentImage, garmentType) {
    const category = ['lower_body', 'pants'].includes(garmentType.toLowerCase()) ? 'lower_body' : 'upper_body';
    let res = await fetch("https://api.replicate.com/v1/predictions", {
        method: "POST",
        headers: { "Authorization": `Token ${REPLICATE_TOKEN}`, "Content-Type": "application/json" },
        body: JSON.stringify({
            version: "0513734a452173b8173e907e3a59d19a36266e55b48528559432bd21c7d7e985",
            input: { human_img: personImage, garm_img: garmentImage, garment_des: "clothing", category, n_samples: 1, seed: 42 }
        })
    });
    let result = await res.json();
    for (let i=0; i<40; i++) {
        if (result.status === 'succeeded' || result.status === 'failed') break;
        await new Promise(r => setTimeout(r, 2000));
        let poll = await fetch(result.urls.get, { headers: { "Authorization": `Token ${REPLICATE_TOKEN}` } });
        result = await poll.json();
    }
    return Array.isArray(result.output) ? result.output[0] : result.output;
}

async function run() {
    console.log("Starting local test bypassing Supabase timeouts...");
    const buf = fs.readFileSync(path.join(__dirname, 'assets/images/mannequin_front.png'));
    let currentMannequin = "data:image/png;base64," + buf.toString('base64');
    
    for (let i=0; i<ITEMS.length; i++) {
        console.log(`Step ${i+1}: Applying ${ITEMS[i].label}`);
        try {
            currentMannequin = await callReplicate(currentMannequin, ITEMS[i].imageUrl, ITEMS[i].type);
        } catch (err) {
            console.error("Step failed:", err);
            return;
        }
    }
    console.log("FINAL GENERATION RESULT URL:", currentMannequin);
}
run();
