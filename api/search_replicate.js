import "dotenv/config";
import fetch from "node-fetch";

async function run() {
    const token = process.env.REPLICATE_API_TOKEN;

    try {
        const res = await fetch(`https://api.replicate.com/v1/models?query=ip-adapter`, {
            headers: { "Authorization": `Token ${token}` }
        });
        const data = await res.json();
        const results = data.results.slice(0, 10).map(m => m.owner + "/" + m.name);
        console.log("Found:", results);
    } catch (e) {
        console.log(`❌ error ${e.message}`);
    }
}
run();
