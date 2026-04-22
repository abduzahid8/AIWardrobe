/**
 * Mixed Outfit Service — AI outfit generation combining wardrobe + shop items
 *
 * Creates complete outfits by intelligently mixing:
 * - User's existing wardrobe items
 * - Shop recommendations to fill gaps
 */

import { supabase } from '../../lib/supabase';
import { createLogger } from '../../utils/logger';

const logger = createLogger('MixedOutfit');

export interface ShopItem {
    id: string;
    name?: string;
    brand?: string;
    price?: number;
    category: string;
    subCategory?: string;
    primaryColor?: string;
    image: string | number;
    imageUrl?: string;
    type?: string;
    shopUrl?: string;
}

export interface MixedOutfitItem {
    id: string;
    type: string;
    color: string;
    image?: string | number;
    recommendation: string;
    isShopItem: boolean;
    shopUrl?: string;
    price?: number;
    brand?: string;
    name?: string;
}

export interface MixedOutfit {
    id: string;
    occasion: string;
    style: string;
    description: string;
    confidence: number;
    matchScore: number;
    items: MixedOutfitItem[];
    stylingTips: string[];
    mainImage?: string | number;
    wardrobeItemCount: number;
    shopItemCount: number;
}

export interface GenerateMixedOutfitsParams {
    wardrobeItems: any[];
    shopItems: ShopItem[];
    occasion?: string;
    stylePreferences?: string;
    weather?: { temp: number; condition: string };
    limit?: number;
}

/**
 * Generate outfits mixing wardrobe items with shop recommendations
 */
export async function generateMixedOutfits(
    params: GenerateMixedOutfitsParams
): Promise<{ success: boolean; outfits: MixedOutfit[]; error?: string }> {
    const {
        wardrobeItems = [],
        shopItems = [],
        occasion = 'casual',
        stylePreferences = 'modern',
        weather,
        limit = 3
    } = params;

    try {
        // Format wardrobe items for the edge function
        const formattedWardrobe = wardrobeItems.map(item => ({
            id: item.id || item.uniqueId,
            type: item.type || item.itemType || item.category,
            color: item.color || item.primaryColor,
            description: item.description || item.name,
            image: item.image || item.imageUrl,
            style: item.style,
        }));

        // Format shop items
        const formattedShop = shopItems.map(item => ({
            id: item.id,
            name: item.name,
            brand: item.brand,
            price: item.price,
            type: item.type || item.category || item.subCategory,
            color: item.primaryColor || 'neutral',
            image: item.image || item.imageUrl,
            shopUrl: item.shopUrl,
            isShopItem: true,
        }));

        logger.info(`🎨 Generating mixed outfits: ${formattedWardrobe.length} wardrobe + ${formattedShop.length} shop items`);

        // Call Supabase Edge Function
        const { data, error } = await supabase.functions.invoke('generate-outfits', {
            body: {
                occasion,
                stylePreferences,
                wardrobeItems: formattedWardrobe,
                shopItems: formattedShop,
                weather,
                limit
            }
        });

        if (error) {
            console.error('Edge function error:', error);
            throw error;
        }

        if (data && data.success && data.outfits) {
            logger.info(`✅ Received ${data.outfits.length} outfits from AI`);
            // Process and enhance the outfits
            const processedOutfits: MixedOutfit[] = data.outfits.map((outfit: any, outfitIdx: number) => {
                logger.debug(`Processing outfit ${outfitIdx}: ${outfit.id}, items: ${outfit.items?.length || 0}`);
                
                // Handle edge case where items might be missing or malformed
                const rawItems = outfit.items || [];
                
                const items: MixedOutfitItem[] = rawItems.map((item: any, itemIdx: number) => {
                    // Debug logging
                    logger.debug(`  Item ${itemIdx}: id=${item.id}, type=${item.type}, hasImage=${!!item.image}`);
                    
                    // Find the original item to get the image - try multiple ID formats
                    const wardrobeMatch = wardrobeItems.find(w => {
                        const wId = String(w.id || w.uniqueId || '');
                        const itemId = String(item.id || '');
                        return wId === itemId || wId.includes(itemId) || itemId.includes(wId);
                    });
                    
                    const shopMatch = shopItems.find(s => {
                        const sId = String(s.id || '');
                        const itemId = String(item.id || '');
                        return sId === itemId;
                    });

                    // Determine if this is a shop item
                    const isShopItem = item.isShopItem === true || !!shopMatch || (!wardrobeMatch && !!shopMatch);

                    // Resolve image - prioritize the original source
                    const resolvedImage = item.image 
                        || wardrobeMatch?.image 
                        || wardrobeMatch?.imageUrl 
                        || shopMatch?.image;

                    logger.debug(`    Resolved: image=${!!resolvedImage}, isShopItem=${isShopItem}, match=${wardrobeMatch ? 'wardrobe' : shopMatch ? 'shop' : 'none'}`);

                    return {
                        id: item.id || `item_${Date.now()}_${itemIdx}`,
                        type: item.type || wardrobeMatch?.type || shopMatch?.type || 'clothing',
                        color: item.color || wardrobeMatch?.color || wardrobeMatch?.primaryColor || shopMatch?.primaryColor || 'neutral',
                        recommendation: item.recommendation || 'Great choice for this outfit',
                        isShopItem,
                        image: resolvedImage, // May be undefined - UI will show placeholder
                        shopUrl: item.shopUrl || shopMatch?.shopUrl,
                        price: item.price || shopMatch?.price,
                        brand: item.brand || shopMatch?.brand,
                        name: item.name || shopMatch?.name || wardrobeMatch?.name || item.type || 'Item',
                    };
                }); // Keep all items, even without images - UI will handle placeholders

                logger.debug(`  Final items count: ${items.length}/${rawItems.length}`);

                // Strip duplicate-category items from AI response (e.g. 2 outerwear items)
                const seenTypes = new Set<string>();
                const dedupedItems = items.filter((item) => {
                    const cat = classifyItemType(item.type || '');
                    if (seenTypes.has(cat)) return false;
                    seenTypes.add(cat);
                    return true;
                });
                items.length = 0;
                items.push(...dedupedItems);

                // If no items resolved, pick one per category from wardrobe as fallback
                if (items.length === 0 && wardrobeItems.length > 0) {
                    logger.debug('  Using fallback wardrobe items');
                    const fallbackSeen = new Set<string>();
                    const fallbackItems = wardrobeItems
                        .filter((w: any) => {
                            const cat = classifyItemType(w.type || w.category || '');
                            if (fallbackSeen.has(cat)) return false;
                            fallbackSeen.add(cat);
                            return true;
                        })
                        .slice(0, 4)
                        .map((w: any, idx: number) => ({
                            id: w.id || `fallback_${idx}`,
                            type: w.type || 'clothing',
                            color: w.color || w.primaryColor || 'neutral',
                            image: w.image || w.imageUrl,
                            recommendation: 'From your wardrobe',
                            isShopItem: false,
                            name: w.name || w.type || 'Item',
                        }));
                    items.push(...fallbackItems);
                }

                const wardrobeCount = items.filter(i => !i.isShopItem).length;
                const shopCount = items.filter(i => i.isShopItem).length;

                return {
                    id: outfit.id || `outfit_${Date.now()}_${Math.random()}`,
                    occasion: outfit.occasion || 'Casual',
                    style: outfit.style || stylePreferences,
                    description: outfit.description || `A ${stylePreferences} outfit for you`,
                    confidence: outfit.confidence || 0.85,
                    matchScore: outfit.confidence || outfit.matchScore || 0.78,
                    items,
                    stylingTips: outfit.stylingTips || ['Mix and match to personalize this look'],
                    mainImage: items[0]?.image,
                    wardrobeItemCount: wardrobeCount,
                    shopItemCount: shopCount,
                };
            });

            return { success: true, outfits: processedOutfits };
        }

        return { success: false, outfits: [], error: 'No outfits returned from AI' };

    } catch (err: any) {
        console.error('Mixed outfit generation error:', err);

        // Return fallback with local matching
        const fallbackOutfits = generateLocalMixedOutfits(
            wardrobeItems,
            shopItems,
            stylePreferences,
            limit
        );

        return { success: true, outfits: fallbackOutfits };
    }
}

/**
 * Classify a clothing type string into a macro category.
 * Keeps outerwear (jackets/coats) separate from tops (shirts/t-shirts).
 */
function classifyItemType(type: string): 'outerwear' | 'top' | 'bottom' | 'shoes' | 'other' {
    const t = (type || '').toLowerCase();
    if (t.includes('jacket') || t.includes('coat') || t.includes('zip') ||
        t.includes('sweater') || t.includes('pullover') || t.includes('hoodie') ||
        t.includes('cardigan') || t.includes('vest') || t.includes('puffer') ||
        t.includes('outerwear')) return 'outerwear';
    if (t.includes('polo') || t.includes('shirt') || t.includes('t-shirt') ||
        t.includes('tee') || t.includes('blouse') || t.includes('top')) return 'top';
    if (t.includes('pant') || t.includes('trouser') || t.includes('jeans') ||
        t.includes('bottom')) return 'bottom';
    if (t.includes('shoe') || t.includes('sneaker') || t.includes('boot') ||
        t.includes('loafer')) return 'shoes';
    return 'other';
}

/**
 * Local fallback when AI service is unavailable
 */
function generateLocalMixedOutfits(
    wardrobeItems: any[],
    shopItems: ShopItem[],
    stylePreferences: string,
    limit: number
): MixedOutfit[] {
    const categorize = classifyItemType;

    // Group by category
    const wardrobeTops = wardrobeItems.filter(i => categorize(i.type) === 'top');
    const wardrobeBottoms = wardrobeItems.filter(i => categorize(i.type) === 'bottom');
    const wardrobeShoes = wardrobeItems.filter(i => categorize(i.type) === 'shoes');

    const shopTops = shopItems.filter(i => categorize(i.type || i.category) === 'top');
    const shopBottoms = shopItems.filter(i => categorize(i.type || i.category) === 'bottom');
    const shopShoes = shopItems.filter(i => categorize(i.type || i.category) === 'shoes');

    const outfits: MixedOutfit[] = [];

    for (let i = 0; i < limit; i++) {
        const items: MixedOutfitItem[] = [];

        // Try wardrobe first, fall back to shop
        const top = wardrobeTops[i % wardrobeTops.length] || shopTops[i % shopTops.length];
        const bottom = wardrobeBottoms[i % wardrobeBottoms.length] || shopBottoms[i % shopBottoms.length];
        const shoes = wardrobeShoes[i % wardrobeShoes.length] || shopShoes[i % shopShoes.length];

        if (top) {
            items.push({
                id: top.id,
                type: top.type || 'top',
                color: top.color || top.primaryColor || 'neutral',
                image: top.image || top.imageUrl,
                recommendation: 'Core piece for this style',
                isShopItem: !wardrobeTops.includes(top),
                price: top.price,
                brand: top.brand,
                name: top.name,
            });
        }

        if (bottom) {
            items.push({
                id: bottom.id,
                type: bottom.type || 'bottom',
                color: bottom.color || bottom.primaryColor || 'neutral',
                image: bottom.image || bottom.imageUrl,
                recommendation: 'Pairs well with the top',
                isShopItem: !wardrobeBottoms.includes(bottom),
                price: bottom.price,
                brand: bottom.brand,
                name: bottom.name,
            });
        }

        if (shoes) {
            items.push({
                id: shoes.id,
                type: shoes.type || 'shoes',
                color: shoes.color || shoes.primaryColor || 'neutral',
                image: shoes.image || shoes.imageUrl,
                recommendation: 'Completes the look',
                isShopItem: !wardrobeShoes.includes(shoes),
                price: shoes.price,
                brand: shoes.brand,
                name: shoes.name,
            });
        }

        const wardrobeCount = items.filter(i => !i.isShopItem).length;
        const shopCount = items.filter(i => i.isShopItem).length;

        outfits.push({
            id: `local_${Date.now()}_${i}`,
            occasion: 'Everyday',
            style: stylePreferences,
            description: `A ${stylePreferences} outfit combining your wardrobe with curated shop picks.`,
            confidence: 0.75,
            matchScore: 0.78,
            items,
            stylingTips: ['Add accessories to personalize this look', 'Mix textures for visual interest'],
            mainImage: items[0]?.image,
            wardrobeItemCount: wardrobeCount,
            shopItemCount: shopCount,
        });
    }

    return outfits;
}

/**
 * Calculate match percentage between wardrobe and shop items
 */
export function calculateOutfitMatchScore(
    outfit: MixedOutfit,
    stylePreference: string
): number {
    const baseScore = outfit.confidence || 0.78;
    const wardrobeRatio = outfit.items.length > 0
        ? outfit.wardrobeItemCount / outfit.items.length
        : 0;

    // Boost score if outfit uses more wardrobe items
    // and matches the requested style
    const styleMatch = outfit.style.toLowerCase().includes(stylePreference.toLowerCase())
        ? 0.1
        : 0;

    return Math.min(0.99, baseScore + (wardrobeRatio * 0.05) + styleMatch);
}
