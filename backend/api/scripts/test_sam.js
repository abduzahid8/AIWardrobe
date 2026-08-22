import 'dotenv/config';
import fs from 'fs';
import replicateService from './services/replicate.js';

const test = async () => {
    try {
        console.log("Loading test image...");
        // A simple tiny image, just to test the API connection and parameters
        const tinyImageBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

        console.log("Calling isolateItem with Grounded SAM for 'Jacket'...");
        const result = await replicateService.isolateItem(tinyImageBase64, "Jacket");

        console.log("Success! Output starts with:", result.image.substring(0, 50));
        console.log("Processing time:", result.processingTimeMs, "ms");
    } catch (e) {
        console.log("Error:", e.message);
    }
}
test();
