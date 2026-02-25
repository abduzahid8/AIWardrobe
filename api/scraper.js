import * as cheerio from 'cheerio';
import axios from 'axios';
import { scrapeProductPuppeteer } from './scraper_puppeteer.js';

const HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
  'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
  'Accept-Language': 'en-US,en;q=0.9',
  'Accept-Encoding': 'gzip, deflate, br',
  'Connection': 'keep-alive',
  'Upgrade-Insecure-Requests': '1',
  'Sec-Fetch-Dest': 'document',
  'Sec-Fetch-Mode': 'navigate',
  'Sec-Fetch-Site': 'none',
  'Sec-Fetch-User': '?1',
  'Cache-Control': 'max-age=0'
};

export const scrapeProduct = async (url) => {
  try {
    const domain = new URL(url).hostname;

    // Use Puppeteer for hard-to-scrape domains
    if (domain.includes('zara') || domain.includes('massimodutti') || domain.includes('ralphlauren')) {
      console.log(`Using Puppeteer for ${domain}...`);
      return await scrapeProductPuppeteer(url);
    }

    // 1. Fetch HTML with enhanced headers
    const { data } = await axios.get(url, { headers: HEADERS });

    // 2. Load into Cheerio
    const $ = cheerio.load(data);
    let title = "";
    let image = "";

    // Try extracting from JSON-LD (common for e-commerce)
    $('script[type="application/ld+json"]').each((_, script) => {
      try {
        const json = JSON.parse($(script).html());
        if (json['@type'] === 'Product' || json['@type'] === 'ItemPage') {
          title = json.name || title;
          image = json.image || (Array.isArray(json.image) ? json.image[0] : image);
        }
      } catch (e) {
        // ignore parsing errors
      }
    });

    // 4. Fallback to Open Graph
    if (!title) title = $('meta[property="og:title"]').attr('content') || $('title').text();
    if (!image) image = $('meta[property="og:image"]').attr('content');

    // 5. Fallback to common selectors if still empty
    if (!image) image = $('img').first().attr('src');
    if (!title) title = $('h1').first().text();

    // Clean up
    title = title ? title.trim() : "Unknown Item";

    // Ensure image is absolute URL if relative
    if (image && image.startsWith('//')) {
      image = 'https:' + image;
    } else if (image && image.startsWith('/')) {
      const origin = new URL(url).origin;
      image = origin + image;
    }

    return {
      success: true,
      data: {
        title,
        image,
        url
      }
    };

  } catch (error) {
    // Return more specific error for 403
    if (error.response && error.response.status === 403) {
      console.error(`Access Denied (403) for ${url}`);
      return {
        success: false,
        error: "Access Denied by retailer (Bot Protection). Try a different product or store."
      };
    }

    console.error("Scraping Error:", error.message);
    return {
      success: false,
      error: "Could not parse this link. Try another one."
    };
  }
};