
// Direct NVCF test from Node.js to see exact error message
// This bypasses the edge function to test NVCF directly
const https = require('https');
const fs = require('fs');

// We can't get the actual NVIDIA key here, but we can verify the NVCF endpoint format
// by testing with a dummy key: the error message will tell us if it's a 401 or other issue
async function testNVCFEndpoint() {
    console.log('Testing NVCF asset endpoint reachability...');
    try {
        const res = await fetch('https://api.nvcf.nvidia.com/v2/nvcf/assets', {
            method: 'POST',
            headers: {
                'Authorization': 'Bearer DUMMY_KEY_FOR_FORMAT_TEST',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
            body: JSON.stringify({ contentType: 'image/jpeg', description: 'test' })
        });
        const text = await res.text();
        console.log('NVCF Status:', res.status);
        console.log('NVCF Response:', text.slice(0, 300));
        console.log('\n→ 401 = endpoint works, key was wrong (expected with dummy key)');
        console.log('→ Network error = endpoint not reachable from Deno edge');
    } catch (e) {
        console.error('Network error:', e.message);
    }
}

testNVCFEndpoint();
