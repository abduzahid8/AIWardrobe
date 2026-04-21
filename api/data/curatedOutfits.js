// Curated Outfit Database
// Pre-made outfit combinations with real fashion products

/**
 * @typedef {Object} OutfitItem
 * @property {string} type
 * @property {string} name
 * @property {string} image
 * @property {string} [brand]
 * @property {string} color
 */

/**
 * @typedef {Object} CuratedOutfit
 * @property {string} id
 * @property {string[]} occasion
 * @property {string[]} style
 * @property {string[]} season
 * @property {OutfitItem[]} items
 * @property {string} mainImage
 * @property {string} stylingTips
 * @property {string} description
 */

export const curatedOutfits = [
    // CASUAL DATE OUTFITS
    {
        id: 'casual_date_1',
        occasion: ['date', 'dinner', 'romantic'],
        style: ['casual', 'elegant', 'feminine'],
        season: ['spring', 'summer', 'fall'],
        items: [
            {
                type: 'dress',
                name: 'Black Midi Dress',
                image: 'https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400&h=500&fit=crop',
                brand: 'Zara',
                color: 'black'
            },
            {
                type: 'heels',
                name: 'Strappy Heels',
                image: 'https://images.unsplash.com/photo-1543163521-1bf539c55dd2?w=400&h=500&fit=crop',
                color: 'nude'
            },
            {
                type: 'bag',
                name: 'Mini Crossbody Bag',
                image: 'https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=400&h=500&fit=crop',
                color: 'black'
            }
        ],
        mainImage: 'https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=800&h=1200&fit=crop',
        stylingTips: 'Keep jewelry minimal. Add a light cardigan for cooler evenings.',
        description: 'Timeless elegance with a black midi dress and classic accessories'
    },

    // COFFEE CASUAL
    {
        id: 'coffee_casual_1',
        occasion: ['coffee', 'casual', 'brunch'],
        style: ['relaxed', 'comfortable', 'chic'],
        season: ['spring', 'fall'],
        items: [
            {
                type: 'jeans',
                name: 'High-Waisted Jeans',
                image: 'https://images.unsplash.com/photo-1542272454315-4c01d7abdf4a?w=400&h=500&fit=crop',
                brand: 'Levi\'s',
                color: 'blue'
            },
            {
                type: 'top',
                name: 'White T-Shirt',
                image: 'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&h=500&fit=crop',
                color: 'white'
            },
            {
                type: 'jacket',
                name: 'Denim Jacket',
                image: 'https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400&h=500&fit=crop',
                color: 'blue'
            },
            {
                type: 'sneakers',
                name: 'White Sneakers',
                image: 'https://images.unsplash.com/photo-1460353581641-37baddab0fa2?w=400&h=500&fit=crop',
                color: 'white'
            }
        ],
        mainImage: 'https://images.unsplash.com/photo-1542272454315-4c01d7abdf4a?w=800&h=1200&fit=crop',
        stylingTips: 'Roll up sleeves for a relaxed vibe. Add sunglasses.',
        description: 'Classic denim-on-denim with casual sneakers for effortless style'
    },

    // INTERVIEW PROFESSIONAL
    {
        id: 'interview_professional_1',
        occasion: ['interview', 'work', 'professional'],
        style: ['formal', 'professional', 'confident'],
        season: ['all'],
        items: [
            {
                type: 'blazer',
                name: 'Navy Blazer',
                image: 'https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=400&h=500&fit=crop',
                brand: 'H&M',
                color: 'navy'
            },
            {
                type: 'shirt',
                name: 'White Button-Down',
                image: 'https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400&h=500&fit=crop',
                color: 'white'
            },
            {
                type: 'pants',
                name: 'Black Trousers',
                image: 'https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=400&h=500&fit=crop',
                color: 'black'
            },
            {
                type: 'shoes',
                name: 'Black Pumps',
                image: 'https://images.unsplash.com/photo-1543163521-1bf539c55dd2?w=400&h=500&fit=crop',
                color: 'black'
            }
        ],
        mainImage: 'https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=800&h=1200&fit=crop',
        stylingTips: 'Keep makeup natural. Simple stud earrings. Minimal fragrance.',
        description: 'Professional confidence in a classic navy blazer and tailored trousers'
    },

    // PARTY NIGHT OUT
    {
        id: 'party_night_1',
        occasion: ['party', 'night', 'celebration'],
        style: ['bold', 'glamorous', 'trendy'],
        season: ['all'],
        items: [
            {
                type: 'dress',
                name: 'Sequin Mini Dress',
                image: 'https://images.unsplash.com/photo-1566174053879-31528523f8ae?w=400&h=500&fit=crop',
                brand: 'Zara',
                color: 'silver'
            },
            {
                type: 'heels',
                name: 'Strappy Heels',
                image: 'https://images.unsplash.com/photo-1543163521-1bf539c55dd2?w=400&h=500&fit=crop',
                color: 'black'
            },
            {
                type: 'clutch',
                name: 'Evening Clutch',
                image: 'https://images.unsplash.com/photo-1566150905458-1bf1fc113f0d?w=400&h=500&fit=crop',
                color: 'black'
            }
        ],
        mainImage: 'https://images.unsplash.com/photo-1566174053879-31528523f8ae?w=800&h=1200&fit=crop',
        stylingTips: 'Statement earrings. Bold lip color. Confidence!',
        description: 'Show-stopping glamour in a sequin dress perfect for dancing'
    },

    // GYM ATHLETIC
    {
        id: 'gym_athletic_1',
        occasion: ['gym', 'workout', 'athletic'],
        style: ['sporty', 'comfortable', 'functional'],
        season: ['all'],
        items: [
            {
                type: 'leggings',
                name: 'High-Waist Leggings',
                image: 'https://images.unsplash.com/photo-1506629082955-511b1aa562c8?w=400&h=500&fit=crop',
                brand: 'Nike',
                color: 'black'
            },
            {
                type: 'top',
                name: 'Sports Bra',
                image: 'https://images.unsplash.com/photo-1571731956672-f2b94d7dd0cb?w=400&h=500&fit=crop',
                color: 'gray'
            },
            {
                type: 'sneakers',
                name: 'Running Shoes',
                image: 'https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&h=500&fit=crop',
                brand: 'Adidas',
                color: 'white'
            }
        ],
        mainImage: 'https://images.unsplash.com/photo-1506629082955-511b1aa562c8?w=800&h=1200&fit=crop',
        stylingTips: 'Tie hair up. Bring water bottle. Comfortable socks.',
        description: 'Performance-ready activewear for your best workout'
    },

    // MINIMALIST CASUAL
    {
        id: 'minimalist_casual_1',
        occasion: ['casual', 'everyday', 'shopping'],
        style: ['minimalist', 'neutral', 'comfortable'],
        season: ['all'],
        items: [
            {
                type: 'sweater',
                name: 'Beige Sweater',
                image: 'https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=400&h=500&fit=crop',
                color: 'beige'
            },
            {
                type: 'pants',
                name: 'Wide-Leg Trousers',
                image: 'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?w=400&h=500&fit=crop',
                color: 'cream'
            },
            {
                type: 'sneakers',
                name: 'Minimalist Sneakers',
                image: 'https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400&h=500&fit=crop',
                color: 'white'
            }
        ],
        mainImage: 'https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=800&h=1200&fit=crop',
        stylingTips: 'Add a tote bag. Simple gold jewelry. Natural makeup.',
        description: 'Effortless minimalism in neutral tones for everyday elegance'
    },

    // SUMMER BEACH
    {
        id: 'summer_beach_1',
        occasion: ['beach', 'vacation', 'summer'],
        style: ['breezy', 'relaxed', 'colorful'],
        season: ['summer'],
        items: [
            {
                type: 'dress',
                name: 'Flowy Maxi Dress',
                image: 'https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=400&h=500&fit=crop',
                color: 'floral'
            },
            {
                type: 'sandals',
                name: 'Flat Sandals',
                image: 'https://images.unsplash.com/photo-1603487742131-4160ec999306?w=400&h=500&fit=crop',
                color: 'tan'
            },
            {
                type: 'hat',
                name: 'Straw Hat',
                image: 'https://images.unsplash.com/photo-1521369909029-2afed882baee?w=400&h=500&fit=crop',
                color: 'natural'
            }
        ],
        mainImage: 'https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=800&h=1200&fit=crop',
        stylingTips: 'Sunscreen is a must! Add sunglasses. Keep it breezy.',
        description: 'Vacation-ready in a flowing maxi dress and sun protection'
    },

    // FORMAL EVENING
    {
        id: 'formal_evening_1',
        occasion: ['formal', 'gala', 'wedding'],
        style: ['elegant', 'sophisticated', 'luxurious'],
        season: ['all'],
        items: [
            {
                type: 'gown',
                name: 'Evening Gown',
                image: 'https://images.unsplash.com/photo-1566174053879-31528523f8ae?w=400&h=500&fit=crop',
                brand: 'Designer',
                color: 'navy'
            },
            {
                type: 'heels',
                name: 'High Heels',
                image: 'https://images.unsplash.com/photo-1543163521-1bf539c55dd2?w=400&h=500&fit=crop',
                color: 'silver'
            },
            {
                type: 'clutch',
                name: 'Elegant Clutch',
                image: 'https://images.unsplash.com/photo-1566150905458-1bf1fc113f0d?w=400&h=500&fit=crop',
                color: 'gold'
            }
        ],
        mainImage: 'https://images.unsplash.com/photo-1566174053879-31528523f8ae?w=800&h=1200&fit=crop',
        stylingTips: 'Updo hairstyle. Statement jewelry. Evening makeup.',
        description: 'Red carpet-worthy elegance for special occasions'
    }
];

// Helper function to match outfits based on occasion and style
export function matchOutfits(occasion, styleKeywords = [], limit = 5) {
    // Score each outfit
    const scoredOutfits = curatedOutfits.map(outfit => {
        let score = 0;

        // Occasion match (highest weight)
        if (outfit.occasion.includes(occasion.toLowerCase())) {
            score += 10;
        }

        // Style keyword matches
        styleKeywords.forEach(keyword => {
            if (outfit.style.some(s => s.includes(keyword.toLowerCase()))) {
                score += 3;
            }
            if (outfit.description.toLowerCase().includes(keyword.toLowerCase())) {
                score += 1;
            }
        });

        return { outfit, score };
    });

    // Sort by score and return top matches
    return scoredOutfits
        .sort((a, b) => b.score - a.score)
        .slice(0, limit)
        .map(item => item.outfit);
}
