const fs = require('fs');

const NVIDIA_KEY = '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';
const NVCF_ASSETS_URL = 'https://api.nvcf.nvidia.com/v2/nvcf/assets';

async function run() {
    const createRes = await fetch(NVCF_ASSETS_URL, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${NVIDIA_KEY}`,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        },
        body: JSON.stringify({ contentType: 'image/png', description: 'test' })
    });
    const { assetId, uploadUrl } = await createRes.json();
    console.log("Asset ID:", assetId);
    
    const buf = fs.readFileSync('assets/images/mannequin_front.png');
    
    console.log("Attempting PUT...");
    const putRes = await fetch(uploadUrl, {
        method: 'PUT',
        headers: {
            'Content-Type': 'image/png',
            'x-amz-meta-nvcf-asset-description': 'test'
        },
        body: buf
    });
    console.log("PUT status:", putRes.status);
    console.log("PUT body:", await putRes.text());
}
run();
