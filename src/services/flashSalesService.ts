/**
 * Flash Sales Service
 * Manages Maison Safqa-style flash sale events for premium brand excess inventory
 * Helps luxury brands convert stock into revenue through exclusive time-limited sales
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { FlashSaleEvent, FlashSaleProduct, FlashSaleStatus } from '../types/flashSales';
import { createLogger } from '../utils/logger';

const logger = createLogger('FlashSales');

const API_URL = process.env.EXPO_PUBLIC_API_URL || 'https://aiwardrobe-api.onrender.com';

// ============================================
// MOCK DATA (Replace with real API in production)
// ============================================

const MOCK_FLASH_EVENTS: FlashSaleEvent[] = [
    {
        id: 'flash-1',
        title: 'Massimo Dutti Exclusive',
        brand: 'Massimo Dutti',
        brandLogo: 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Massimo_Dutti_logo.svg/200px-Massimo_Dutti_logo.svg.png',
        description: 'Premium Italian-inspired collection at up to 70% off. Limited pieces from the latest season.',
        heroImage: 'https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=800',
        startTime: new Date(Date.now() - 2 * 60 * 60 * 1000), // Started 2 hours ago
        endTime: new Date(Date.now() + 22 * 60 * 60 * 1000), // Ends in 22 hours
        status: 'active',
        discountPercentage: 70,
        itemCount: 45,
        isExclusive: true,
        subscriberCount: 1247,
    },
    {
        id: 'flash-2',
        title: 'Zara Premium Selection',
        brand: 'Zara',
        brandLogo: 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Zara_Logo.svg/200px-Zara_Logo.svg.png',
        description: 'Selected pieces from the Zara Studio line. Architectural silhouettes and luxury fabrics.',
        heroImage: 'https://images.unsplash.com/photo-1445205170230-053b83016050?w=800',
        startTime: new Date(Date.now() + 4 * 60 * 60 * 1000), // Starts in 4 hours
        endTime: new Date(Date.now() + 28 * 60 * 60 * 1000),
        status: 'upcoming',
        discountPercentage: 60,
        itemCount: 78,
        isExclusive: false,
        subscriberCount: 892,
    },
    {
        id: 'flash-3',
        title: 'COS Minimalist Edit',
        brand: 'COS',
        brandLogo: 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/39/COS_Logo.svg/200px-COS_Logo.svg.png',
        description: 'Clean lines and timeless pieces. Scandinavian design philosophy at exclusive prices.',
        heroImage: 'https://images.unsplash.com/photo-1483985988355-763728e1935b?w=800',
        startTime: new Date(Date.now() + 12 * 60 * 60 * 1000), // Starts in 12 hours
        endTime: new Date(Date.now() + 36 * 60 * 60 * 1000),
        status: 'upcoming',
        discountPercentage: 55,
        itemCount: 32,
        isExclusive: true,
        subscriberCount: 634,
    },
    {
        id: 'flash-4',
        title: 'MAX&Co. Flash Deal',
        brand: 'MAX&Co.',
        description: 'Modern femininity meets Italian craftsmanship. Ending soon!',
        heroImage: 'https://images.unsplash.com/photo-1469334031218-e382a71b716b?w=800',
        startTime: new Date(Date.now() - 20 * 60 * 60 * 1000),
        endTime: new Date(Date.now() + 4 * 60 * 60 * 1000), // Ends in 4 hours
        status: 'active',
        discountPercentage: 65,
        itemCount: 28,
        isExclusive: false,
        subscriberCount: 456,
    },
];

const MOCK_FLASH_PRODUCTS: FlashSaleProduct[] = [
    {
        id: 'fp-1',
        eventId: 'flash-1',
        name: 'Wool Blend Structured Blazer',
        brand: 'Massimo Dutti',
        originalPrice: 395,
        salePrice: 118,
        currency: 'USD',
        imageUrl: 'https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=400',
        productUrl: 'https://example.com/blazer',
        affiliateUrl: 'https://example.com/blazer?ref=aiwardrobe',
        category: 'tops',
        color: 'Navy',
        size: ['S', 'M', 'L'],
        stockStatus: 'in_stock',
        stockCount: 12,
        rating: 4.8,
        reviewCount: 67,
    },
    {
        id: 'fp-2',
        eventId: 'flash-1',
        name: 'Italian Leather Belt',
        brand: 'Massimo Dutti',
        originalPrice: 125,
        salePrice: 45,
        currency: 'USD',
        imageUrl: 'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400',
        productUrl: 'https://example.com/belt',
        affiliateUrl: 'https://example.com/belt?ref=aiwardrobe',
        category: 'accessories',
        color: 'Brown',
        size: ['32', '34', '36', '38'],
        stockStatus: 'low_stock',
        stockCount: 4,
        rating: 4.9,
        reviewCount: 134,
    },
    {
        id: 'fp-3',
        eventId: 'flash-1',
        name: 'Cashmere Crew Neck Sweater',
        brand: 'Massimo Dutti',
        originalPrice: 285,
        salePrice: 99,
        currency: 'USD',
        imageUrl: 'https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=400',
        productUrl: 'https://example.com/sweater',
        affiliateUrl: 'https://example.com/sweater?ref=aiwardrobe',
        category: 'tops',
        color: 'Camel',
        size: ['XS', 'S', 'M', 'L', 'XL'],
        stockStatus: 'in_stock',
        stockCount: 23,
        rating: 4.7,
        reviewCount: 89,
    },
    {
        id: 'fp-4',
        eventId: 'flash-1',
        name: 'Slim Fit Chino Trousers',
        brand: 'Massimo Dutti',
        originalPrice: 195,
        salePrice: 68,
        currency: 'USD',
        imageUrl: 'https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=400',
        productUrl: 'https://example.com/chinos',
        affiliateUrl: 'https://example.com/chinos?ref=aiwardrobe',
        category: 'bottoms',
        color: 'Beige',
        size: ['28', '30', '32', '34', '36'],
        stockStatus: 'in_stock',
        stockCount: 18,
        rating: 4.6,
        reviewCount: 156,
    },
    {
        id: 'fp-5',
        eventId: 'flash-4',
        name: 'Silk Blend Midi Dress',
        brand: 'MAX&Co.',
        originalPrice: 345,
        salePrice: 121,
        currency: 'USD',
        imageUrl: 'https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400',
        productUrl: 'https://example.com/dress',
        affiliateUrl: 'https://example.com/dress?ref=aiwardrobe',
        category: 'dresses',
        color: 'Burgundy',
        size: ['XS', 'S', 'M', 'L'],
        stockStatus: 'low_stock',
        stockCount: 3,
        rating: 4.8,
        reviewCount: 42,
    },
    {
        id: 'fp-6',
        eventId: 'flash-4',
        name: 'Leather Crossbody Bag',
        brand: 'MAX&Co.',
        originalPrice: 265,
        salePrice: 93,
        currency: 'USD',
        imageUrl: 'https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=400',
        productUrl: 'https://example.com/bag',
        affiliateUrl: 'https://example.com/bag?ref=aiwardrobe',
        category: 'accessories',
        color: '#0A1931',
        stockStatus: 'in_stock',
        stockCount: 15,
        rating: 4.5,
        reviewCount: 78,
    },
];

// ============================================
// FLASH SALES SERVICE
// ============================================

class FlashSalesService {
    private subscribedEventIds: string[] = [];
    private cachedEvents: FlashSaleEvent[] = [];

    constructor() {
        this.loadSubscriptions();
    }

    private async loadSubscriptions() {
        try {
            const stored = await AsyncStorage.getItem('flashSalesSubscriptions');
            if (stored) {
                this.subscribedEventIds = JSON.parse(stored);
            }
        } catch (error) {
            console.error('Failed to load flash sale subscriptions:', error);
        }
    }

    private async saveSubscriptions() {
        try {
            await AsyncStorage.setItem(
                'flashSalesSubscriptions',
                JSON.stringify(this.subscribedEventIds)
            );
        } catch (error) {
            console.error('Failed to save subscriptions:', error);
        }
    }

    /**
     * Determine event status based on current time
     */
    private getEventStatus(event: FlashSaleEvent): FlashSaleStatus {
        const now = new Date();
        const start = new Date(event.startTime);
        const end = new Date(event.endTime);

        if (now < start) return 'upcoming';
        if (now > end) return 'ended';
        return 'active';
    }

    /**
     * Get all flash sale events
     */
    async getAllEvents(): Promise<FlashSaleEvent[]> {
        // In production, fetch from API
        // For now, use mock data and update status
        const events = MOCK_FLASH_EVENTS.map(event => ({
            ...event,
            status: this.getEventStatus(event),
        }));

        this.cachedEvents = events;
        return events;
    }

    /**
     * Get currently active flash sales
     */
    async getActiveEvents(): Promise<FlashSaleEvent[]> {
        const events = await this.getAllEvents();
        return events.filter(e => e.status === 'active');
    }

    /**
     * Get upcoming flash sales
     */
    async getUpcomingEvents(): Promise<FlashSaleEvent[]> {
        const events = await this.getAllEvents();
        return events.filter(e => e.status === 'upcoming');
    }

    /**
     * Get ended flash sales (for discovery)
     */
    async getEndedEvents(): Promise<FlashSaleEvent[]> {
        const events = await this.getAllEvents();
        return events.filter(e => e.status === 'ended');
    }

    /**
     * Get a specific event by ID
     */
    async getEventById(eventId: string): Promise<FlashSaleEvent | null> {
        const events = await this.getAllEvents();
        return events.find(e => e.id === eventId) || null;
    }

    /**
     * Get products for a specific flash sale event
     */
    async getEventProducts(eventId: string): Promise<FlashSaleProduct[]> {
        // In production, fetch from API
        return MOCK_FLASH_PRODUCTS.filter(p => p.eventId === eventId);
    }

    /**
     * Subscribe to event notifications
     */
    async subscribeToEvent(eventId: string): Promise<boolean> {
        if (!this.subscribedEventIds.includes(eventId)) {
            this.subscribedEventIds.push(eventId);
            await this.saveSubscriptions();

            // Track subscription analytics
            this.trackEventAction(eventId, 'subscribe');
            return true;
        }
        return false;
    }

    /**
     * Unsubscribe from event notifications
     */
    async unsubscribeFromEvent(eventId: string): Promise<boolean> {
        const index = this.subscribedEventIds.indexOf(eventId);
        if (index > -1) {
            this.subscribedEventIds.splice(index, 1);
            await this.saveSubscriptions();
            return true;
        }
        return false;
    }

    /**
     * Check if user is subscribed to an event
     */
    isSubscribed(eventId: string): boolean {
        return this.subscribedEventIds.includes(eventId);
    }

    /**
     * Get all subscribed event IDs
     */
    getSubscribedEventIds(): string[] {
        return [...this.subscribedEventIds];
    }

    /**
     * Calculate time remaining for an event
     */
    getTimeRemaining(event: FlashSaleEvent): {
        hours: number;
        minutes: number;
        seconds: number;
        isEnding: boolean;
        totalSeconds: number;
    } {
        const now = new Date();
        const targetTime = event.status === 'upcoming'
            ? new Date(event.startTime)
            : new Date(event.endTime);

        const diff = Math.max(0, targetTime.getTime() - now.getTime());
        const totalSeconds = Math.floor(diff / 1000);
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;

        return {
            hours,
            minutes,
            seconds,
            isEnding: hours < 2 && event.status === 'active',
            totalSeconds,
        };
    }

    /**
     * Format time remaining as string
     */
    formatTimeRemaining(event: FlashSaleEvent): string {
        const { hours, minutes, seconds } = this.getTimeRemaining(event);

        if (hours > 24) {
            const days = Math.floor(hours / 24);
            return `${days}d ${hours % 24}h`;
        }

        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        }

        return `${minutes}m ${seconds}s`;
    }

    /**
     * Get featured event (most prominent active or upcoming)
     */
    async getFeaturedEvent(): Promise<FlashSaleEvent | null> {
        const active = await this.getActiveEvents();
        if (active.length > 0) {
            // Return the one ending soonest
            return active.sort((a, b) =>
                new Date(a.endTime).getTime() - new Date(b.endTime).getTime()
            )[0];
        }

        const upcoming = await this.getUpcomingEvents();
        if (upcoming.length > 0) {
            // Return the one starting soonest
            return upcoming.sort((a, b) =>
                new Date(a.startTime).getTime() - new Date(b.startTime).getTime()
            )[0];
        }

        return null;
    }

    /**
     * Track user action for analytics
     */
    private async trackEventAction(eventId: string, action: string) {
        try {
            await fetch(`${API_URL}/api/analytics/flash-sale`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    eventId,
                    action,
                    timestamp: new Date().toISOString(),
                }),
            }).catch(() => { }); // Non-blocking
        } catch (error) {
            // Silent fail for analytics
        }
    }

    /**
     * Track product view
     */
    async trackProductView(product: FlashSaleProduct) {
        logger.info(`Product viewed: ${product.name}`);
        this.trackEventAction(product.eventId, 'product_view');
    }

    /**
     * Get affiliate link for flash sale product
     */
    getAffiliateLink(product: FlashSaleProduct): string {
        return product.affiliateUrl || product.productUrl;
    }
}

// Export singleton instance
export const flashSalesService = new FlashSalesService();
export default flashSalesService;
