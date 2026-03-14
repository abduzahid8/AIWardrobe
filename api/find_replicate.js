import "dotenv/config";
import fetch from "node-fetch";

async function run() {
    const token = process.env.REPLICATE_API_TOKEN; // users token from .env

    const models = [
        "lucataco/ip-adapter-sdxl",
        "lucataco/sdxl-controlnet",
        "stability-ai/sdxl",
        "tencent/hunyuan3d-1.0",
        "cuuupid/idm-vton"
    ];

    for (const m of models) {
        try {
            const res = await fetch(`https://api.replicate.com/v1/models/${m}`, {
                headers: { "Authorization": `Token ${token}` }
            });
            const data = await res.json();
            if (data.latest_version) {
                console.log(`✅ ${m}: ${data.latest_version.id}`);
            } else {
                console.log(`❌ ${m}: not found or no version`);
            }
        } catch (e) {
            console.log(`❌ ${m}: error ${e.message}`);
        }
    }
}
run();
