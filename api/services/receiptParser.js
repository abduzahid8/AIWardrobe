/**
 * Receipt Parser Service
 * Extracts clothing purchase information from email receipts
 * Supports major fashion retailers and marketplaces
 */

const patterns = {
    // Common clothing item keywords
    clothingKeywords: [
        'shirt', 'tshirt', 't-shirt', 'blouse', 'top', 'sweater', 'hoodie', 'cardigan',
        'pants', 'jeans', 'trousers', 'shorts', 'skirt', 'dress', 'suit',
        'jacket', 'coat', 'blazer', 'parka', 'bomber', 'denim',
        'shoes', 'sneakers', 'boots', 'heels', 'sandals', 'flats',
        'bag', 'handbag', 'backpack', 'purse', 'tote',
        'belt', 'scarf', 'hat', 'cap', 'gloves', 'socks',
        'underwear', 'bra', 'lingerie', 'swimwear', 'activewear', 'leggings'
    ],

    // Price patterns
    price: /(?:USD|RUB|₽|\$|€|£)\s*(\d+(?:[.,]\d{2})?)|(\d+(?:[.,]\\d{2})?)\s*(?:USD|RUB|₽|\$|€|£)/gi,

    // Date patterns
    date: /(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})|(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})|([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})/gi,

    // Color patterns
    colors: [
        'black', 'white', 'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
        'brown', 'gray', 'grey', 'beige', 'navy', 'maroon', 'burgundy', 'cream', 'ivory',
        'khaki', 'olive', 'teal', 'turquoise', 'gold', 'silver', 'charcoal',
        'черный', 'белый', 'красный', 'синий', 'зеленый', 'желтый' // Russian colors
    ],

    // Material patterns
    materials: [
        'cotton', 'polyester', 'wool', 'silk', 'linen', 'denim', 'leather',
        'suede', 'velvet', 'cashmere', 'fleece', 'nylon', 'spandex', 'rayon'
    ],

    // Retailer patterns
    retailers: {
        'zara.com': 'Zara',
        'hm.com': 'H&M',
        'uniqlo.com': 'Uniqlo',
        'gap.com': 'Gap',
        'nordstrom.com': 'Nordstrom',
        'amazon.com': 'Amazon',
        'asos.com': 'ASOS',
        'shein.com': 'SHEIN',
        'wildberries.ru': 'Wildberries',
        'ozon.ru': 'Ozon',
        'lamoda.ru': 'Lamoda',
        'yoox.com': 'YOOX',
        'farfetch.com': 'Farfetch',
        'net-a-porter.com': 'Net-A-Porter'
    }
};

/**
 * Parse email content to extract receipt information
 * @param {Object} email - Email object with subject, body, from
 * @returns {Object[]} Array of detected clothing items
 */
function parseReceipt(email) {
    const { subject = '', body = '', from = '', html = '' } = email;
    const fullText = `${subject} ${body} ${html}`.toLowerCase();

    // Detect retailer
    const retailer = detectRetailer(from, subject, fullText);

    // Extract date
    const purchaseDate = extractDate(fullText) || new Date().toISOString();

    // Extract clothing items
    const items = extractClothingItems(fullText, html);

    // Extract prices
    const prices = extractPrices(fullText);

    // Match items with prices (best effort)
    const enrichedItems = items.map((item, index) => ({
        ...item,
        retailer,
        purchaseDate,
        price: prices[index] || null,
        source: 'email_receipt'
    }));

    return enrichedItems;
}

/**
 * Detect retailer from email metadata
 */
function detectRetailer(from, subject, text) {
    // Check from email domain
    for (const [domain, name] of Object.entries(patterns.retailers)) {
        if (from.toLowerCase().includes(domain)) {
            return name;
        }
    }

    // Check subject/body for retailer name
    for (const name of Object.values(patterns.retailers)) {
        if (text.includes(name.toLowerCase())) {
            return name;
        }
    }

    return 'Unknown Retailer';
}

/**
 * Extract purchase date from text
 */
function extractDate(text) {
    const matches = text.match(patterns.date);
    if (matches && matches.length > 0) {
        try {
            return new Date(matches[0]).toISOString();
        } catch (e) {
            return null;
        }
    }
    return null;
}

/**
 * Extract clothing items from text
 */
function extractClothingItems(text, html) {
    const items = [];
    const lines = text.split('\n');

    lines.forEach(line => {
        // Check if line contains clothing keywords
        const matchedKeyword = patterns.clothingKeywords.find(keyword =>
            line.includes(keyword)
        );

        if (matchedKeyword) {
            // Extract full item description (the line or sentence)
            const description = extractItemDescription(line, html);

            // Extract attributes
            const color = extractColor(line);
            const material = extractMaterial(line);
            const size = extractSize(line);

            items.push({
                itemType: capitalizeFirst(matchedKeyword),
                description,
                color,
                material,
                size,
                rawText: line.trim()
            });
        }
    });

    return items;
}

/**
 * Extract clean item description
 */
function extractItemDescription(line, html) {
    // Remove common noise words
    let clean = line
        .replace(/qty:?\s*\d+/gi, '')
        .replace(/quantity:?\s*\d+/gi, '')
        .replace(/size:?\s*[smlxSMLX\d]+/gi, '')
        .replace(/color:?\s*\w+/gi, '')
        .trim();

    // Try to extract product name from HTML if available
    if (html) {
        const productNameMatch = html.match(/<td[^>]*class="product-name"[^>]*>(.*?)<\/td>/i);
        if (productNameMatch) {
            return productNameMatch[1].replace(/<[^>]*>/g, '').trim();
        }
    }

    return clean.substring(0, 200); // Limit length
}

/**
 * Extract color from text
 */
function extractColor(text) {
    for (const color of patterns.colors) {
        if (text.includes(color)) {
            return capitalizeFirst(color);
        }
    }
    return null;
}

/**
 * Extract material from text
 */
function extractMaterial(text) {
    for (const material of patterns.materials) {
        if (text.includes(material)) {
            return capitalizeFirst(material);
        }
    }
    return null;
}

/**
 * Extract size from text
 */
function extractSize(text) {
    const sizeMatch = text.match(/size:?\s*([smlxSMLX\d]+)/i);
    if (sizeMatch) {
        return sizeMatch[1].toUpperCase();
    }

    // Common size patterns
    const commonSizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL'];
    for (const size of commonSizes) {
        if (text.toUpperCase().includes(size)) {
            return size;
        }
    }

    return null;
}

/**
 * Extract prices from text
 */
function extractPrices(text) {
    const matches = text.match(patterns.price);
    if (!matches) return [];

    return matches.map(priceStr => {
        // Extract numeric value
        const numMatch = priceStr.match(/(\d+(?:[.,]\d{2})?)/);
        if (numMatch) {
            return parseFloat(numMatch[1].replace(',', '.'));
        }
        return null;
    }).filter(p => p !== null);
}

/**
 * Capitalize first letter
 */
function capitalizeFirst(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Check if email is likely a clothing purchase receipt
 */
function isClothingReceipt(email) {
    const { subject = '', body = '', from = '' } = email;
    const fullText = `${subject} ${body}`.toLowerCase();

    // Check for receipt keywords
    const receiptKeywords = ['order', 'receipt', 'purchase', 'confirmation', 'invoice', 'заказ', 'покупка'];
    const hasReceiptKeyword = receiptKeywords.some(keyword => fullText.includes(keyword));

    if (!hasReceiptKeyword) return false;

    // Check for clothing keywords
    const hasClothingKeyword = patterns.clothingKeywords.some(keyword =>
        fullText.includes(keyword)
    );

    // Check if from known retailer
    const fromKnownRetailer = Object.keys(patterns.retailers).some(domain =>
        from.toLowerCase().includes(domain)
    );

    return hasClothingKeyword || fromKnownRetailer;
}

module.exports = {
    parseReceipt,
    isClothingReceipt,
    patterns
};
