import 'dotenv/config';
import openaiService from './services/openai.js';

const test = async () => {
    try {
        console.log("Loading image...");
        // A simple 10x10 transparent PNG base64 string
        const tinyImageBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

        console.log("Calling editImage...");
        const result = await openaiService.editImage(tinyImageBase64, "A beautiful studio background");
        console.log("Success! Output starts with:", result.substring(0, 50));
    } catch (e) {
        console.log("Error:", e.message);
    }
}
test();
