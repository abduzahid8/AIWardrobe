// Debug: test Replicate pants prediction directly
const fs = require('fs');

const TOKEN = process.env.REPLICATE_TOKEN || 'YOUR_REPLICATE_TOKEN_HERE';

async function run() {
    // Use the step2 layer result as the person image
    const step2Buf = fs.readFileSync('./tryon_step2_layer.jpg');
    const personB64 = 'data:image/jpeg;base64,' + step2Buf.toString('base64');

    console.log('Submitting pants prediction to Replicate directly...');
    const res = await fetch('https://api.replicate.com/v1/predictions', {
        method: 'POST',
        headers: { 'Authorization': `Token ${TOKEN}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({
            version: '0513734a452173b8173e907e3a59d19a36266e55b48528559432bd21c7d7e985',
            input: {
                human_img: personB64,
                garm_img: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/450251/item/usgoods_35_450251_3x4.jpg',
                garment_des: 'pants clothing item',
                category: 'lower_body',
                n_samples: 1,
                seed: 42,
            }
        })
    });

    const prediction = await res.json();
    console.log('Prediction status:', prediction.status, 'id:', prediction.id);
    if (prediction.error) { console.error('Error:', prediction.error); return; }

    // Poll
    for (let i = 0; i < 30; i++) {
        await new Promise(r => setTimeout(r, 3000));
        const poll = await fetch(`https://api.replicate.com/v1/predictions/${prediction.id}`, {
            headers: { 'Authorization': `Token ${TOKEN}` }
        });
        const data = await poll.json();
        console.log(`Poll ${i+1}: status=${data.status}`);
        if (data.status === 'succeeded') {
            const url = Array.isArray(data.output) ? data.output[0] : data.output;
            console.log('SUCCESS:', url);
            const img = await fetch(url);
            fs.writeFileSync('./tryon_step3_pants_direct.jpg', Buffer.from(await img.arrayBuffer()));
            console.log('Saved → tryon_step3_pants_direct.jpg');
            return;
        }
        if (data.status === 'failed') { console.error('FAILED:', data.error); return; }
    }
}
run();
