const fs = require('fs');
const fetch = require('node-fetch');

async function testNVIDIA() {
    const NVIDIA_KEY = "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";
    
    // I need a perfectly 1024x1024 image. I will just pass an empty base64 or a small base64, 
    // wait I don't have canvas installed here to easily generate a 1024x1024 image.
    // The mannequin image is 800x1328! 
    // Wait, the previous sizes error said: `Input should be ... 800 ... 1328 ...` !
    // Both 800 and 1328 are valid magic sizes!
}
