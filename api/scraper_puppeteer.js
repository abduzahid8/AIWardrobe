import puppeteer from 'puppeteer';

export const scrapeProductPuppeteer = async (url) => {
    let browser;
    try {
        browser = await puppeteer.launch({
            headless: "new", // or true
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });
        const page = await browser.newPage();

        // Set a real User-Agent
        await page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');

        console.log(`Navigating to ${url}...`);
        await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });

        // Wait for some content (optional, but good for heavily JS sites)
        // await page.waitForSelector('h1', { timeout: 5000 }).catch(() => {});

        const data = await page.evaluate(() => {
            const getMeta = (prop) => document.querySelector(`meta[property="${prop}"]`)?.content;
            const getJsonLd = () => {
                const scripts = document.querySelectorAll('script[type="application/ld+json"]');
                for (const script of scripts) {
                    try {
                        const json = JSON.parse(script.innerText);
                        if (json['@type'] === 'Product' || json['@type'] === 'ItemPage') {
                            return json;
                        }
                    } catch (e) { }
                }
                return null;
            };

            const json = getJsonLd();
            const title = json?.name || getMeta('og:title') || document.title;
            let image = json?.image || (Array.isArray(json?.image) ? json.image[0] : null) || getMeta('og:image');

            if (!image) {
                const img = document.querySelector('img');
                image = img ? img.src : '';
            }

            return { title, image };
        });

        return {
            success: true,
            data: {
                ...data,
                url
            }
        };

    } catch (error) {
        console.error("Puppeteer Error:", error.message);
        return {
            success: false,
            error: "Could not scrape with Puppeteer."
        };
    } finally {
        if (browser) await browser.close();
    }
};

// Self-test if run directly
if (process.argv[1] === import.meta.url.substring(7)) { // simple check
    (async () => {
        const url = "https://www.zara.com/us/en/woman-dresses-l1066.html";
        console.log(await scrapeProductPuppeteer(url));
    })();
}
