import 'dotenv/config';
import replicateService from './services/replicate.js';

const test = async () => {
    try {
        console.log("=== Ghost Mannequin Pipeline Test ===\n");

        // A simple 10x10 transparent PNG for connection testing
        const tinyImageBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

        // Test 1: isolateItem (Grounded SAM)
        console.log("1. Testing Grounded SAM isolateItem...");
        try {
            const sam = await replicateService.isolateItem(tinyImageBase64, "Jacket");
            console.log("   ✅ Grounded SAM OK:", sam.image.substring(0, 30) + "...");
        } catch (e) {
            console.log("   ❌ Grounded SAM:", e.message);
        }

        // Test 2: generateGhostMannequin
        console.log("\n2. Testing Ghost Mannequin generation...");
        try {
            const gm = await replicateService.generateGhostMannequin(tinyImageBase64, "Jacket");
            console.log("   ✅ Ghost Mannequin OK:", gm.image.substring(0, 30) + "...");
        } catch (e) {
            console.log("   ❌ Ghost Mannequin:", e.message);
        }

        console.log("\n=== Tests Complete ===");
    } catch (e) {
        console.log("Fatal error:", e.message);
    }
}
test();
