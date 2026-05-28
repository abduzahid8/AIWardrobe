import "dotenv/config";
import fetch from "node-fetch";

async function run() {
    const token = process.env.REPLICATE_API_TOKEN;

    try {
        const res = await fetch(`https://api.replicate.com/v1/models?search=ip-adapter`, {
            headers: { "Authorization": `Token ${token}` }
        });
        const data = await res.json();
        const results = data.results.slice(0, 50).filter(m => m.name.includes("ip-adapter") || m.name.includes("sdxl")).map(m => m.owner + "/" + m.name + " (" + m.description + ")");
        console.log("Found:", JSON.stringify(results, null, 2));
    } catch (e) {
        console.log(`❌ error ${e.message}`);
    }
}
run();
