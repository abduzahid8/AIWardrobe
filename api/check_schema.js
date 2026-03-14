import "dotenv/config";
import fetch from "node-fetch";

async function run() {
    const token = process.env.REPLICATE_API_TOKEN; // users token from .env

    try {
        const res = await fetch(`https://api.replicate.com/v1/models/stability-ai/sdxl/versions/7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc`, {
            headers: { "Authorization": `Token ${token}` }
        });
        const data = await res.json();
        console.log("SCHEMA:", JSON.stringify(data.openapi_schema.components.schemas.Input, null, 2));
    } catch (e) {
        console.log(`❌ error ${e.message}`);
    }
}
run();
