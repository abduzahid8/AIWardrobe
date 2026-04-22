import type { ShopCatalogItem } from '../features/try-on/types';
import { BASIC_CLOTHING_ITEMS } from './basicClothingItems';

const SHOP_PRICES: Record<string, number> = {
    'basic-1': 39.9,
    'basic-2': 149.0,
    'basic-3': 119.0,
    'basic-4': 145.0,
    'basic-5': 129.0,
    'basic-6': 125.0,
    'basic-7': 135.0,
    'basic-8': 139.0,
    'basic-9': 149.0,
    'basic-10': 89.9,
    'basic-11': 95.9,
};

function categoryToGarmentType(category: string): ShopCatalogItem['garmentType'] {
    if (category === 'bottom') return 'lower_body';
    if (category === 'shoes') return 'shoes';
    return 'upper_body';
}

function itemBrand(itemId: string, category: string): string {
    if (category === 'shoes' || itemId === 'basic-11') return 'ZARA';
    return 'ZARA';
}

const LOCAL_BASIC_ITEMS: ShopCatalogItem[] = BASIC_CLOTHING_ITEMS
    .filter((item) => item.category !== 'accessory')
    .map((item) => ({
        id: item.id,
        brand: itemBrand(item.id, item.category),
        name: item.name,
        price: SHOP_PRICES[item.id] ?? 99.0,
        currency: 'USD',
        garmentType: categoryToGarmentType(item.category),
        description: item.description,
        imageUrl: item.image,
    }));

/**
 * Curated classic menswear — Oxfords, chinos, blazers, loafers, trenches, etc.
 * Exported so it can be appended to the live Supabase catalog as well as used
 * as fallback content when the live catalog is empty, guaranteeing the Shop
 * tab always surfaces a deep bench of men's classic staples.
 */
export const CLASSIC_MENS_ITEMS: ShopCatalogItem[] = [
    // ── Shirts ─────────────────────────────────────────────────────────
    {
        id: 'classic-m-shirt-01',
        brand: 'Uniqlo',
        name: 'Oxford Slim-Fit Shirt',
        price: 39.9,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'Button-down collar 100% cotton oxford in light blue',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg',
    },
    {
        id: 'classic-m-shirt-02',
        brand: 'Uniqlo',
        name: 'Easy Care Broadcloth Shirt',
        price: 29.9,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'Wrinkle-resistant slim-fit broadcloth dress shirt in blue',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/448264/item/goods_67_448264_3x4.jpg',
    },
    {
        id: 'classic-m-shirt-03',
        brand: 'Uniqlo',
        name: 'Premium Linen Long Sleeve Shirt',
        price: 39.9,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'Breathable 100% linen shirt with classic collar',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/455957/item/usgoods_57_455957_3x4.jpg',
    },
    {
        id: 'classic-m-shirt-04',
        brand: 'Uniqlo U',
        name: 'Cotton-Linen Long-Sleeve Shirt',
        price: 49.9,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'Relaxed-fit cotton-linen overshirt in natural tones',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/447764/item/goods_30_447764_3x4.jpg',
    },

    // ── Polos & Knits ──────────────────────────────────────────────────
    {
        id: 'classic-m-polo-01',
        brand: 'Uniqlo',
        name: 'AIRism Pique Short-Sleeve Polo',
        price: 29.9,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'Classic pique polo with AIRism comfort in navy',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/424217/item/goods_09_424217_3x4.jpg',
    },
    {
        id: 'classic-m-polo-02',
        brand: 'Uniqlo',
        name: 'Dry Pique Striped Polo',
        price: 24.9,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'Dry pique short-sleeve polo in wide stripe',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/485565/item/usgoods_55_485565_3x4.jpg',
    },
    {
        id: 'classic-m-knit-01',
        brand: 'Uniqlo',
        name: 'Merino Crewneck Sweater',
        price: 39.9,
        currency: 'USD',
        garmentType: 'upper_body',
        description: '100% extra-fine merino wool crewneck in blue',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/469395/item/goods_67_469395_3x4.jpg',
    },
    {
        id: 'classic-m-knit-02',
        brand: 'Uniqlo',
        name: 'Extra Fine Merino Turtleneck',
        price: 49.9,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'Fine-gauge merino wool turtleneck sweater in navy',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/450538/item/goods_69_450538_3x4.jpg',
    },
    {
        id: 'classic-m-knit-03',
        brand: 'Uniqlo',
        name: 'Merino Turtleneck Sweater',
        price: 49.9,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'Fine-gauge merino wool turtleneck in camel',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/429067/item/goods_03_429067_3x4.jpg',
    },
    {
        id: 'classic-m-knit-04',
        brand: 'Uniqlo',
        name: 'Cable Crew Neck Sweater',
        price: 49.9,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'Chunky cable-knit crewneck sweater in navy',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/451685/item/goods_69_451685_3x4.jpg',
    },
    {
        id: 'classic-m-knit-05',
        brand: 'Uniqlo',
        name: 'Premium Lambswool V-Neck Cardigan',
        price: 49.9,
        currency: 'USD',
        garmentType: 'upper_body',
        description: '100% premium lambswool v-neck cardigan in off-white',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/450542/item/goods_01_450542_3x4.jpg',
    },

    // ── Blazers & Outerwear ────────────────────────────────────────────
    {
        id: 'classic-m-blazer-01',
        brand: 'Uniqlo',
        name: 'AirSense Blazer',
        price: 99.9,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'Wool-like ultra-light two-button blazer in navy',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/448034/item/usgoods_09_448034_3x4.jpg',
    },
    {
        id: 'classic-m-coat-01',
        brand: 'Burberry',
        name: 'Long Kensington Heritage Trench Coat',
        price: 2850.0,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'Honey cotton gabardine double-breasted trench coat',
        imageUrl: 'https://assets.burberry.com/is/image/Burberryltd/3DFB8EAD-C042-4E2C-B62D-9F3C1B6011DC',
    },
    {
        id: 'classic-m-coat-02',
        brand: 'Uniqlo',
        name: 'Wool Cashmere Chesterfield Coat',
        price: 199.9,
        currency: 'USD',
        garmentType: 'upper_body',
        description: '90% wool 10% cashmere single-breasted overcoat in brown',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/449890/item/goods_69_449890_3x4.jpg',
    },
    {
        id: 'classic-m-jacket-01',
        brand: 'Uniqlo',
        name: 'Denim Trucker Jacket',
        price: 59.9,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'Relaxed-fit stretch denim trucker jacket in mid-wash',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/484402/item/usgoods_64_484402_3x4.jpg',
    },

    // ── Trousers & Chinos ──────────────────────────────────────────────
    {
        id: 'classic-m-chino-01',
        brand: 'Uniqlo',
        name: 'Slim-Fit Chino Pants',
        price: 39.9,
        currency: 'USD',
        garmentType: 'lower_body',
        description: 'Supima cotton stretch slim-fit chinos in brown',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/450251/item/usgoods_35_450251_3x4.jpg',
    },
    {
        id: 'classic-m-trouser-01',
        brand: 'Uniqlo U',
        name: 'Wool-Blend Tailored Pants',
        price: 69.9,
        currency: 'USD',
        garmentType: 'lower_body',
        description: 'Relaxed-fit wool-blend tailored trousers in khaki',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/443403/item/goods_36_443403_3x4.jpg',
    },
    {
        id: 'classic-m-trouser-02',
        brand: 'Uniqlo',
        name: 'Smart Ankle Pants (Wool-Like)',
        price: 49.9,
        currency: 'USD',
        garmentType: 'lower_body',
        description: 'Two-way stretch wool-look tapered smart ankle pants in charcoal',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/455492/item/usgoods_38_455492_3x4.jpg',
    },
    {
        id: 'classic-m-jean-01',
        brand: 'Uniqlo',
        name: 'Straight Jeans',
        price: 39.9,
        currency: 'USD',
        garmentType: 'lower_body',
        description: 'Classic straight-leg denim in mid blue wash',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/479816/item/usgoods_63_479816_3x4.jpg',
    },
    {
        id: 'classic-m-jean-02',
        brand: 'Uniqlo',
        name: 'Selvedge Straight Jeans',
        price: 59.9,
        currency: 'USD',
        garmentType: 'lower_body',
        description: 'Authentic selvedge denim in dark navy with straight cut',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/485737/item/usgoods_69_485737_3x4.jpg',
    },
    {
        id: 'classic-m-short-01',
        brand: 'Uniqlo',
        name: 'Chino Shorts (7")',
        price: 29.9,
        currency: 'USD',
        garmentType: 'lower_body',
        description: '100% cotton twill chino shorts with regular fit',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/458245/item/goods_09_458245_3x4.jpg',
    },

    // ── Shoes ──────────────────────────────────────────────────────────
    {
        id: 'classic-m-shoe-01',
        brand: 'Uniqlo : C',
        name: 'Combination Sneaker',
        price: 69.9,
        currency: 'USD',
        garmentType: 'shoes',
        description: 'White leather with beige suede panels on gum rubber sole',
        imageUrl: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/484330/item/goods_32_484330_3x4.jpg',
    },
];

export const INSPO_MENS_SHOP_ITEMS: ShopCatalogItem[] = [
    ...LOCAL_BASIC_ITEMS,
    ...CLASSIC_MENS_ITEMS,
];
