const fs = require('fs');
const TOKEN = process.env.REPLICATE_TOKEN || 'YOUR_REPLICATE_TOKEN_HERE';
const SUPABASE_URL = process.env.SUPABASE_URL || 'YOUR_SUPABASE_URL_HERE';
const ANON_KEY = process.env.SUPABASE_ANON_KEY || 'YOUR_SUPABASE_ANON_KEY_HERE';

async function run() {
    // Wait for rate limit to clear
    console.log('Waiting 15s for rate limit to reset...');
    await new Promise(r => setTimeout(r, 15000));

    // Load step 3 result (mannequin with top + layer + pants)
    const step3Buf = fs.readFileSync('./flux_step3_pants.jpg');
    const mannequin = 'data:image/jpeg;base64,' + step3Buf.toString('base64');

    console.log('Submitting shoes step via edge function...');
    const res = await fetch(`${SUPABASE_URL}/functions/v1/mannequin-tryon`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${ANON_KEY}` },
        body: JSON.stringify({
            action: 'submit',
            mannequin_image: mannequin,
            garment: { image: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/484330/item/goods_32_484330_3x4.jpg', type: 'shoes', label: 'shoes' },
            step: 4, total: 4
        })
    });
    const data = await res.json();
    console.log('Submit result:', JSON.stringify(data).slice(0, 200));
    if (!data.success) { console.error('Submit failed:', data.error); return; }

    let resultUrl;
    if (data.mode === 'sync') {
        resultUrl = data.resultUrl;
    } else {
        const predId = data.predictionId;
        console.log('Polling prediction:', predId);
        for (let i = 0; i < 30; i++) {
            await new Promise(r => setTimeout(r, 3000));
            const pollRes = await fetch(`${SUPABASE_URL}/functions/v1/mannequin-tryon`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${ANON_KEY}` },
                body: JSON.stringify({ action: 'poll', predictionId: predId })
            });
            const poll = await pollRes.json();
            process.stdout.write(`  poll ${i+1}: ${poll.status}\r`);
            if (poll.status === 'succeeded' && poll.resultUrl) { resultUrl = poll.resultUrl; break; }
            if (poll.status === 'failed') { console.error('\nFailed:', poll.error); return; }
        }
    }
    console.log('\n\n=== STEP 4 SHOES RESULT ===');
    console.log(resultUrl);
    const imgRes = await fetch(resultUrl);
    fs.writeFileSync('./flux_step4_shoes.jpg', Buffer.from(await imgRes.arrayBuffer()));
    console.log('Saved → flux_step4_shoes.jpg');
}
run().catch(e => console.error('FAILED:', e.message));
