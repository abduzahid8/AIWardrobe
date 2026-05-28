/**
 * Style Rules — Fashion wisdom synthesized from the brand's style guide transcripts.
 * Used to enhance AI stylist responses and outfit scoring.
 */

import { styleCategories } from "./styleCategories.js";

/**
 * Style Rules — Fashion wisdom synthesized from the brand's style guide transcripts.
 * Categorized by Formality, Item Type, and Seasonality.
 */

export const styleRules = {
    categories: `
        ## Specific Style Categories (DNA)
        ${Object.values(styleCategories).map(cat => `
        ### ${cat.name}
        - **Philosophy**: ${cat.philosophy}
        - **Palette**: ${cat.palette.join(', ')}
        - **Proportions**: ${cat.proportions}
        - **Rules**: ${cat.rules.join(' ')}
        - **Avoid**: ${cat.avoid.join(', ')}
        `).join('\n')}
    `,
    core: `
        ## Core Fashion Principles
        - **The Gray Trousers Rule**: Gray trousers are the most universal. Match them with ANY top (Blazer, Shirt, Knitwear). Avoid black trousers for general versatility.
        - **The Sandwich Rule**: Match the color of your upper body (shirt/blazer) with your shoes to create a balanced "sandwich" look.
        - **Proportions**: A wide lapel requires a larger tie/bow tie. High-waisted trousers create a better silhouette and hide body imperfections better than low-waist.
        - **The 10-Item Capsule**: You only need 10 base items (Navy Blazer, Gray Trousers, Blue Striped Shirt, Brown Loafers, etc.) to create hundreds of combinations.
    `,
    // ... (rest remains the same but will be included in getAllStyleRules)
    formality: `
        ## Formality Hierarchy
        - **Level 1: Classic Formal**: Suit (Jacket + Pants from same fabric), Shirt, Tie, Oxfords. No suede shoes.
        - **Level 2: Smart Casual High**: Sport Jacket (textured/patterned), Trousers, Shirt, Knit Tie.
        - **Level 3: Smart Casual Medium**: Sport Jacket, Knitwear (Turtleneck/Polo), Trousers, Loafers.
        - **Level 4: Smart Casual Low**: Drawstring/Elastic waist wool trousers, Knitwear, Minimalist Sneakers.
        - **Level 5: Casual**: Chinos/Slacks, Polo, Sneakers. No shorts in the city.
    `,
    items: `
        ## Item-Specific Knowledge
        - **Navy Blazer**: The "Swiss Army Knife" of jackets. Use metal buttons for the ultimate classic look. Matches gray, beige, and white pants perfectly.
        - **Sport Jackets**: Should be textured (Tweed, Flannel). Patterns like Gunclub check or Prince of Wales are ideal. No striped jackets separately (striped = suit only).
        - **Knitwear**: Knitted (woven) T-shirts/Polos are superior to plain cotton. Use plain cotton only for gym/sports.
        - **Footwear**:
            - Loafers (Dark Brown Leather): Most universal shoe for Smart Casual/Classic.
            - Minimalist Sneakers (White Leather): Allowed with Knitwear, NOT with Shirts or Suits.
            - Socks: Match socks to trousers or top, NEVER to shoes. Use long socks or "invisible" footies. No ankle socks showing.
        - **Accessories**: Match watch size to wrist size. No backpacks with suits—use leather bags. Only wear wristwatches with tuxedos if they are extremely minimal (or none).
    `,
    antiPatterns: `
        ## Mistakes to Avoid (Anti-Patterns)
        - **No Mixing Styles**: Don't wear sneakers with formal suits or ties with t-shirts.
        - **No Shorts in the City**: Use light trousers (Linen/Tropical Wool) instead.
        - **No Square-Toe Shoes**: Use round or almond shapes.
        - **No Puffy Oversized Jackets with Suits**: Use wool coats or wool puffers.
        - **No Graphic T-shirts**: Use monochrome plain high-quality t-shirts for a refined look.
    `,
    seasonality: `
        ## Seasonal Transitions
        - **Summer**: Use Linen and Tropical Wool (220-240g). Tropical wool is breathable and keeps its shape better than linen.
        - **Winter**: Use Wool ties, Flannel trousers, and shoes with high soles (to isolate from cold ground).
    `
};

export { styleCategories };

// Helper to get all rules as a single string
export const getAllStyleRules = () => Object.values(styleRules).join('\n');
