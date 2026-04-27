
const fs = require('fs');

const NVIDIA_KEY = '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';
const NVCF_ASSETS_URL = 'https://api.nvcf.nvidia.com/v2/nvcf/assets';

async function debugAssetUpload() {
    console.log('=== Debugging NVCF Asset Upload ===');

    const buf = fs.readFileSync('assets/images/mannequin_front.png');
    const mime = 'image/png';

    // Step 1: Create asset record
    console.log('Creating asset record...');
    const createRes = await fetch(NVCF_ASSETS_URL, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${NVIDIA_KEY}`,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        },
        body: JSON.stringify({ contentType: mime, description: 'test-mannequin' })
    });
    console.log('Create status:', createRes.status);
    const createBody = await createRes.text();
    console.log('Create body:', createBody.slice(0, 500));

    if (!createRes.ok) return;

    const { assetId, uploadUrl } = JSON.parse(createBody);
    console.log('\nAsset ID:', assetId);
    console.log('Upload URL:', uploadUrl?.slice(0, 100) + '...');

    // Step 2: Try PUT upload
    console.log('\nUploading to S3...');
    // Try with Content-Type header matching mime
    const putRes1 = await fetch(uploadUrl, {
        method: 'PUT',
        headers: { 'Content-Type': mime },
        body: buf
    });
    console.log('PUT with Content-Type result:', putRes1.status, putRes1.statusText);

    if (!putRes1.ok) {
        // Try without Content-Type (some presigned URLs reject extra headers)
        const putRes2 = await fetch(uploadUrl, {
            method: 'PUT',
            body: buf
        });
        console.log('PUT without Content-Type result:', putRes2.status, putRes2.statusText);
    } else {
        console.log('✅ Upload succeeded!');
    }

    // Clean up
    await fetch(`${NVCF_ASSETS_URL}/${assetId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${NVIDIA_KEY}` }
    });
}

debugAssetUpload().catch(e => console.error('FATAL:', e.message));
