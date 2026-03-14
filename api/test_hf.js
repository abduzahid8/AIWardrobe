import "dotenv/config";
import fetch from 'node-fetch';

async function query() {
    const HF_TOKEN = process.env.HF_TOKEN;

    // Test base SDXL on Hugging Face Serverless
    try {
        const response = await fetch(
            "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
            {
                headers: {
                    Authorization: `Bearer ${HF_TOKEN}`,
                    "Content-Type": "application/json"
                },
                method: "POST",
                body: JSON.stringify({ inputs: "professional studio ghost mannequin product photography of a coat, white background" }),
            }
        );
        if (!response.ok) {
            console.log("Failed:", await response.text());
        } else {
            console.log("Success! Received an image blob of type", response.headers.get("content-type"));
        }
    } catch (e) {
        console.log("Error:", e.message);
    }
}
query();
