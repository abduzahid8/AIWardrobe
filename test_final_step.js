const fs = require('fs');
const TOKEN = process.env.REPLICATE_TOKEN || 'YOUR_REPLICATE_TOKEN_HERE';

// Poll the existing shoes prediction (already submitted, never timed out)
const SHOES_PRED_ID = '7kk6ny5f69rn80cxsdgbvtec40';
const STEP3_PANTS_URL = 'https://replicate.delivery/yhqm/Mor0D5EUJQJBLtIEQj2wVhVekQO9rBQrfyRfFFUpWTrEGEeZB/output.jpg';

async function run() {
    console.log('Polling existing shoes prediction:', SHOES_PRED_ID);

    let shoesUrl = null;
    for (let i = 0; i < 20; i++) {
        await new Promise(r => setTimeout(r, 2000));
        const poll = await fetch(`https://api.replicate.com/v1/predictions/${SHOES_PRED_ID}`, {
            headers: { 'Authorization': `Token ${TOKEN}` }
        });
        const data = await poll.json();
        console.log(`Poll ${i+1}: status=${data.status}`);
        if (data.status === 'succeeded') {
            shoesUrl = Array.isArray(data.output) ? data.output[0] : data.output;
            break;
        }
        if (data.status === 'failed') {
            // Re-submit shoes using step 3 result
            console.log('Shoes prediction failed, re-submitting...');
            const step3Buf = fs.readFileSync('./tryon_step3_pants.jpg');
            const personB64 = 'data:image/jpeg;base64,' + step3Buf.toString('base64');
            const res = await fetch('https://api.replicate.com/v1/predictions', {
                method: 'POST',
                headers: { 'Authorization': `Token ${TOKEN}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    version: '0513734a452173b8173e907e3a59d19a36266e55b48528559432bd21c7d7e985',
                    input: { human_img: personB64, garm_img: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/484330/item/goods_32_484330_3x4.jpg', garment_des: 'shoes', category: 'upper_body', n_samples: 1, seed: 42 }
                })
            });
            const newPred = await res.json();
            console.log('New prediction:', newPred.id, 'status:', newPred.status);
            // Continue polling new pred
            for (let j = 0; j < 30; j++) {
                await new Promise(r => setTimeout(r, 3000));
                const p = await fetch(`https://api.replicate.com/v1/predictions/${newPred.id}`, { headers: { 'Authorization': `Token ${TOKEN}` } });
                const d = await p.json();
                console.log(`  Repoll ${j+1}: ${d.status}`);
                if (d.status === 'succeeded') { shoesUrl = Array.isArray(d.output) ? d.output[0] : d.output; break; }
                if (d.status === 'failed') { console.error('Re-submit also failed:', d.error); return; }
            }
            break;
        }
    }

    if (!shoesUrl) { console.error('Could not get shoes result'); return; }

    console.log('\nShoes result URL:', shoesUrl);
    const imgRes = await fetch(shoesUrl);
    fs.writeFileSync('./tryon_step4_shoes.jpg', Buffer.from(await imgRes.arrayBuffer()));
    console.log('Saved → tryon_step4_shoes.jpg');
    console.log('\n=== ALL 4 STEPS COMPLETE ===');
    console.log('Step 1 top:   ./tryon_step1_top.jpg');
    console.log('Step 2 layer: ./tryon_step2_layer.jpg');
    console.log('Step 3 pants: ./tryon_step3_pants.jpg');
    console.log('Step 4 shoes: ./tryon_step4_shoes.jpg  ← FINAL');
}
run().catch(e => console.error('FAILED:', e.message));
