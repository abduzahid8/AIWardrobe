export interface InspoShopItem {
    id: string;
    brand: string;
    name: string;
    price: number;
    category: string;
    image: ReturnType<typeof require>;
}

export const INSPO_SHOP_ITEMS: InspoShopItem[] = [
    { id: 'shop-inspo-1', brand: 'ZARA', name: 'Oversized Blazer', price: 129.00, category: 'tops', image: require('../pictures/shop/image copy.png') },
    { id: 'shop-inspo-2', brand: 'ZARA', name: 'Wide Leg Trousers', price: 89.90, category: 'bottoms', image: require('../pictures/shop/image copy 2.png') },
    { id: 'shop-inspo-3', brand: 'ZARA', name: 'Structured Jacket', price: 69.90, category: 'tops', image: require('../pictures/shop/image copy 3.png') },
    { id: 'shop-inspo-4', brand: 'ZARA', name: 'Slim Fit Jeans', price: 15.90, category: 'bottoms', image: require('../pictures/shop/image copy 4.png') },
    { id: 'shop-inspo-5', brand: 'ZARA', name: 'Ribbed Knit Top', price: 35.90, category: 'tops', image: require('../pictures/shop/image copy 5.png') },
    { id: 'shop-inspo-6', brand: 'ZARA', name: 'Leather Ankle Boots', price: 99.90, category: 'shoes', image: require('../pictures/shop/image copy 6.png') },
    { id: 'shop-inspo-7', brand: 'ZARA', name: 'Satin Mini Dress', price: 59.90, category: 'tops', image: require('../pictures/shop/image.png') },
    { id: 'shop-inspo-8', brand: 'ZARA', name: 'Brown Pants', price: 79.90, category: 'bottoms', image: require('../pictures/shop/Brown-pants-with_line.png') },
    { id: 'shop-inspo-9', brand: 'ZARA', name: 'Brown Loafers', price: 89.90, category: 'shoes', image: require('../pictures/shop/Brown_loafers.png.png') },
    { id: 'shop-inspo-10', brand: 'ZARA', name: 'Grey Loafers', price: 95.90, category: 'shoes', image: require('../pictures/shop/Grey_loafers_loropiana.png') },
    { id: 'shop-inspo-11', brand: 'ZARA', name: 'High Waist Trousers', price: 69.90, category: 'bottoms', image: require('../pictures/shop/highweist_trousers_whte.png') },
];
