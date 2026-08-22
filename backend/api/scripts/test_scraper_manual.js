import { scrapeProduct } from './scraper.js';

const TEST_URLS = [
    "https://www.zara.com/us/en/woman-dresses-l1066.html",
    "https://www.massimodutti.com/us/men/collection/shirts-n1426",
    "https://www.ralphlauren.com/men-clothing-polo-shirts"
];

const runTests = async () => {
    console.log("Starting Scraper Tests...\n");

    for (const url of TEST_URLS) {
        console.log(`Testing: ${url}`);
        try {
            const result = await scrapeProduct(url);
            console.log("Result:", JSON.stringify(result, null, 2));
        } catch (error) {
            console.error("Failed:", error.message);
        }
        console.log("-".repeat(50));
    }
};

runTests();
