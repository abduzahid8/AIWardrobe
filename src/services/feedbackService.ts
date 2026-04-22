/**
 * 🧠 AI Feedback Service
 * 
 * Collects user corrections to improve AI detection accuracy.
 * Stores corrections in Supabase for model retraining.
 */

import { supabase } from '../lib/supabase';
import { createLogger } from '../utils/logger';

const logger = createLogger('Feedback');

// Types for feedback
export interface DetectionCorrection {
    id?: string;
    original_type: string;
    corrected_type: string;
    category: string;
    image_hash?: string;
    confidence: number;
    created_at?: string;
    user_id?: string;
}

export interface CorrectionStats {
    total_corrections: number;
    most_confused: { from: string; to: string; count: number }[];
    accuracy_improvement: number;
}

// Comprehensive clothing types for correction picker (100+ options)
export const CLOTHING_TYPE_OPTIONS = {
    upper_clothes: [
        // T-Shirts
        'T-Shirt', 'V-Neck T-Shirt', 'Crew Neck T-Shirt', 'Graphic Tee', 'Pocket Tee',
        'Long Sleeve T-Shirt', 'Oversized T-Shirt', 'Henley T-Shirt',
        // Shirts
        'Polo Shirt', 'Button-Down Shirt', 'Dress Shirt', 'Oxford Shirt', 'Flannel Shirt',
        'Linen Shirt', 'Denim Shirt', 'Camp Collar Shirt',
        // Blouses & Tops
        'Blouse', 'Silk Blouse', 'Crop Top', 'Tank Top', 'Camisole', 'Tube Top',
        'Halter Top', 'Off-Shoulder Top', 'Wrap Top',
        // Sweaters & Knitwear
        'Sweater', 'Crewneck Sweater', 'V-Neck Sweater', 'Turtleneck Sweater',
        'Cable Knit Sweater', 'Cashmere Sweater', 'Wool Sweater', 'Cotton Sweater',
        'Cardigan', 'Long Cardigan', 'Cropped Cardigan',
        // Hoodies & Sweatshirts
        'Hoodie', 'Zip-Up Hoodie', 'Pullover Hoodie', 'Cropped Hoodie',
        'Sweatshirt', 'Crewneck Sweatshirt', 'Oversized Sweatshirt',
        // Jackets
        'Denim Jacket', 'Trucker Jacket', 'Leather Jacket', 'Bomber Jacket',
        'Varsity Jacket', 'Fleece Jacket', 'Track Jacket', 'Windbreaker',
        'Blazer', 'Sport Coat', 'Suit Jacket', 'Corduroy Jacket',
        // Coats
        'Coat', 'Trench Coat', 'Puffer Jacket', 'Down Jacket', 'Parka',
        'Peacoat', 'Overcoat', 'Wool Coat', 'Rain Jacket',
        // Vests
        'Vest', 'Puffer Vest', 'Fleece Vest', 'Denim Vest', 'Suit Vest'
    ],
    pants: [
        // Jeans
        'Jeans', 'Skinny Jeans', 'Slim Jeans', 'Straight Jeans', 'Bootcut Jeans',
        'Wide-Leg Jeans', 'Mom Jeans', 'Dad Jeans', 'Boyfriend Jeans',
        'High-Waisted Jeans', 'Low-Rise Jeans', 'Cropped Jeans', 'Ripped Jeans',
        'Black Jeans', 'White Jeans', 'Light Wash Jeans', 'Dark Wash Jeans',
        // Casual Pants
        'Chinos', 'Khakis', 'Cargo Pants', 'Joggers', 'Sweatpants', 'Track Pants',
        'Corduroy Pants', 'Linen Pants', 'Cotton Pants',
        // Dress Pants
        'Dress Pants', 'Slacks', 'Trousers', 'Suit Pants', 'Pleated Pants',
        // Athletic
        'Leggings', 'Yoga Pants', 'Athletic Pants', 'Training Pants',
        // Shorts
        'Shorts', 'Denim Shorts', 'Chino Shorts', 'Cargo Shorts', 'Gym Shorts',
        'Running Shorts', 'Board Shorts', 'Swim Trunks'
    ],
    dress: [
        'Dress', 'Casual Dress', 'Summer Dress', 'Sundress', 'Shirt Dress',
        'T-Shirt Dress', 'Wrap Dress', 'Bodycon Dress', 'A-Line Dress',
        'Fit and Flare Dress', 'Slip Dress', 'Sweater Dress',
        'Cocktail Dress', 'Evening Gown', 'Prom Dress', 'Formal Dress',
        'Maxi Dress', 'Midi Dress', 'Mini Dress',
        'Off-Shoulder Dress', 'Halter Dress', 'Strapless Dress',
        'Jumpsuit', 'Romper', 'Overalls'
    ],
    skirt: [
        'Skirt', 'Mini Skirt', 'Midi Skirt', 'Maxi Skirt',
        'Pencil Skirt', 'A-Line Skirt', 'Pleated Skirt', 'Wrap Skirt',
        'Denim Skirt', 'Leather Skirt', 'Tulle Skirt', 'Tennis Skirt',
        'Flared Skirt', 'Tiered Skirt', 'Slit Skirt'
    ],
    shoes: [
        // Sneakers
        'Sneakers', 'Running Shoes', 'Basketball Shoes', 'Tennis Shoes',
        'High-Top Sneakers', 'Low-Top Sneakers', 'Canvas Sneakers', 'Chunky Sneakers',
        'Athletic Shoes', 'Training Shoes', 'Walking Shoes',
        // Boots
        'Boots', 'Ankle Boots', 'Chelsea Boots', 'Combat Boots', 'Hiking Boots',
        'Riding Boots', 'Knee-High Boots', 'Rain Boots', 'Work Boots',
        'Cowboy Boots', 'Desert Boots', 'Chukka Boots',
        // Dress Shoes
        'Dress Shoes', 'Oxford Shoes', 'Derby Shoes', 'Brogues', 'Loafers',
        'Penny Loafers', 'Tassel Loafers', 'Monk Strap Shoes',
        // Casual
        'Slip-Ons', 'Boat Shoes', 'Espadrilles', 'Moccasins',
        // Sandals
        'Sandals', 'Flip Flops', 'Slides', 'Gladiator Sandals', 'Sport Sandals',
        // Heels
        'Heels', 'High Heels', 'Stilettos', 'Block Heels', 'Wedges',
        'Kitten Heels', 'Platform Heels', 'Pumps', 'Mules',
        // Flats
        'Flats', 'Ballet Flats', 'Pointed Flats', 'Loafer Flats'
    ],
    accessories: [
        'Hat', 'Baseball Cap', 'Snapback', 'Dad Hat', 'Bucket Hat',
        'Beanie', 'Fedora', 'Beret', 'Sun Hat', 'Visor',
        'Scarf', 'Wool Scarf', 'Silk Scarf', 'Bandana',
        'Belt', 'Leather Belt', 'Canvas Belt', 'Chain Belt',
        'Bag', 'Shoulder Bag', 'Crossbody Bag', 'Tote Bag', 'Backpack',
        'Clutch', 'Handbag', 'Messenger Bag', 'Duffel Bag',
        'Watch', 'Sunglasses', 'Glasses', 'Tie', 'Bow Tie',
        'Bracelet', 'Necklace', 'Earrings', 'Ring', 'Gloves'
    ]
};

/**
 * Submit a correction for a misclassified item
 */
export const submitCorrection = async (
    originalType: string,
    correctedType: string,
    category: string,
    confidence: number = 0,
    imageHash?: string
): Promise<boolean> => {
    try {
        const correction: DetectionCorrection = {
            original_type: originalType.toLowerCase(),
            corrected_type: correctedType.toLowerCase(),
            category: category.toLowerCase(),
            confidence,
            image_hash: imageHash,
            created_at: new Date().toISOString()
        };

        logger.info('📝 Submitting correction', correction);

        const { error } = await supabase
            .from('detection_corrections')
            .insert(correction);

        if (error) {
            console.error('Error submitting correction:', error);
            // Fallback: store locally if Supabase fails
            await storeLocalCorrection(correction);
            return true;
        }

        logger.info('✅ Correction submitted successfully');
        return true;
    } catch (error) {
        console.error('Failed to submit correction:', error);
        return false;
    }
};

/**
 * Store correction locally if Supabase is unavailable
 */
const storeLocalCorrection = async (correction: DetectionCorrection): Promise<void> => {
    try {
        const AsyncStorage = require('@react-native-async-storage/async-storage').default;
        const existing = await AsyncStorage.getItem('pending_corrections');
        const corrections = existing ? JSON.parse(existing) : [];
        corrections.push(correction);
        await AsyncStorage.setItem('pending_corrections', JSON.stringify(corrections));
        logger.info('💾 Correction stored locally for later sync');
    } catch (e) {
        console.error('Failed to store local correction:', e);
    }
};

/**
 * Sync pending local corrections to Supabase
 */
export const syncPendingCorrections = async (): Promise<number> => {
    try {
        const AsyncStorage = require('@react-native-async-storage/async-storage').default;
        const pending = await AsyncStorage.getItem('pending_corrections');
        if (!pending) return 0;

        const corrections: DetectionCorrection[] = JSON.parse(pending);
        if (corrections.length === 0) return 0;

        const { error } = await supabase
            .from('detection_corrections')
            .insert(corrections);

        if (!error) {
            await AsyncStorage.removeItem('pending_corrections');
            logger.info(`✅ Synced ${corrections.length} pending corrections`);
            return corrections.length;
        }
        return 0;
    } catch (e) {
        console.error('Failed to sync corrections:', e);
        return 0;
    }
};

/**
 * Get correction statistics
 */
export const getCorrectionStats = async (): Promise<CorrectionStats | null> => {
    try {
        const { data, error } = await supabase
            .from('detection_corrections')
            .select('original_type, corrected_type')
            .order('created_at', { ascending: false })
            .limit(1000);

        if (error || !data) return null;

        // Calculate most confused pairs
        const confusionMap: Record<string, number> = {};
        data.forEach(c => {
            const key = `${c.original_type}→${c.corrected_type}`;
            confusionMap[key] = (confusionMap[key] || 0) + 1;
        });

        const mostConfused = Object.entries(confusionMap)
            .sort(([, a], [, b]) => b - a)
            .slice(0, 5)
            .map(([pair, count]) => {
                const [from, to] = pair.split('→');
                return { from, to, count };
            });

        return {
            total_corrections: data.length,
            most_confused: mostConfused,
            accuracy_improvement: Math.min(data.length * 0.1, 15) // Rough estimate
        };
    } catch (e) {
        console.error('Failed to get stats:', e);
        return null;
    }
};

/**
 * Get options for a specific category
 */
export const getTypeOptionsForCategory = (category: string): string[] => {
    const cat = category.toLowerCase();
    if (cat.includes('shirt') || cat.includes('top') || cat.includes('jacket') || cat.includes('sweater')) {
        return CLOTHING_TYPE_OPTIONS.upper_clothes;
    }
    if (cat.includes('pants') || cat.includes('jeans') || cat.includes('shorts')) {
        return CLOTHING_TYPE_OPTIONS.pants;
    }
    if (cat.includes('dress')) {
        return CLOTHING_TYPE_OPTIONS.dress;
    }
    if (cat.includes('skirt')) {
        return CLOTHING_TYPE_OPTIONS.skirt;
    }
    if (cat.includes('shoe') || cat.includes('boot') || cat.includes('sneaker')) {
        return CLOTHING_TYPE_OPTIONS.shoes;
    }
    // Return all options for unknown categories
    return [
        ...CLOTHING_TYPE_OPTIONS.upper_clothes,
        ...CLOTHING_TYPE_OPTIONS.pants,
        ...CLOTHING_TYPE_OPTIONS.dress,
        ...CLOTHING_TYPE_OPTIONS.shoes
    ];
};
