/**
 * Flash Sales Types
 * Type definitions for Maison Safqa-style flash sale events
 */

export type FlashSaleStatus = 'upcoming' | 'active' | 'ended';

export interface FlashSaleEvent {
    id: string;
    title: string;
    brand: string;
    brandLogo?: string;
    description: string;
    heroImage: string;
    startTime: Date;
    endTime: Date;
    status: FlashSaleStatus;
    discountPercentage: number;
    itemCount: number;
    isExclusive?: boolean;
    subscriberCount?: number;
}

export interface FlashSaleProduct {
    id: string;
    eventId: string;
    name: string;
    brand: string;
    originalPrice: number;
    salePrice: number;
    currency: string;
    imageUrl: string;
    productUrl: string;
    affiliateUrl: string;
    category: string;
    color?: string;
    size?: string[];
    stockStatus: 'in_stock' | 'low_stock' | 'sold_out';
    stockCount?: number;
    rating?: number;
    reviewCount?: number;
}

export interface FlashSaleSubscription {
    eventId: string;
    userId: string;
    subscribedAt: Date;
    notified: boolean;
}

export interface FlashSalesState {
    activeEvents: FlashSaleEvent[];
    upcomingEvents: FlashSaleEvent[];
    endedEvents: FlashSaleEvent[];
    subscribedEventIds: string[];
    loading: boolean;
    error: string | null;
}
