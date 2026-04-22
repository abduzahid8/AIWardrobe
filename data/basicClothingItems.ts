/**
 * Legacy static "basic clothing" catalog — now a stub.
 *
 * Previously this file bundled raw PNGs from a local `basic_clothing/`
 * directory into the IPA. That directory was scratch content and has
 * been removed. The real catalog now lives in the Supabase
 * `shop_catalog` table and is fetched via `src/hooks/useShopCatalog.ts`.
 *
 * We keep the export + type so existing callers continue to compile.
 * New callers should use `useShopCatalog()` directly.
 */

export interface BasicClothingItem {
    id: string;
    name: string;
    category: 'top' | 'bottom' | 'shoes' | 'outerwear' | 'accessory';
    subCategory: string;
    primaryColor: string;
    colorHex: string;
    pattern: string;
    material: string;
    seasons: Array<'spring' | 'summer' | 'fall' | 'winter'>;
    occasions: Array<'casual' | 'work' | 'formal' | 'sport' | 'date' | 'travel'>;
    /** Remote URL. Historically this was a bundled require(...) asset. */
    image: string;
    description: string;
}

export const BASIC_CLOTHING_ITEMS: BasicClothingItem[] = [];
